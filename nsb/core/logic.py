from abc import ABCMeta, abstractmethod
import functools
import graphlib
from collections import defaultdict

import numpy as np
from sklearn.neighbors import BallTree
import scipy.integrate as si

import astropy.units as u
from astropy.coordinates import AltAz, SkyCoord
import nsb.core.utils as utils


class Frame:
    def __init__(self, location, obstime, target, rotation, obswl, **kwargs):
        """
        Frame class holding information about observatory, observation time, observation wavelength and telescope rotation.

        Parameters
        ----------
        location : astropy.coordinates.EarthLocation
            EarthLocation of the observer
        obstime : astropy.time.Time
            Time of the observation
        target : astropy.coordinates.SkyCoord
            Location of the observation center
        rotation : astropy.coordinates.Angle
            Rotation of the telescope around its axis compared to nominal AltAz frame
        obswl : astropy.units.Quantity
            Array of wavelengths for simulation
        """
        self.AltAz = AltAz(obstime=obstime, location=location)
        self.location = location
        self.target = target.transform_to(self.AltAz)
        self.time = obstime
        self.obswl = obswl
        self.conf = kwargs

        self.telframe = self.target.skyoffset_frame(rotation=rotation)


class Model(metaclass=ABCMeta):
    def __init__(self, layers):
        """
        A Model takes a linked layer object to create a computational graph, which can then be executed.

        Parameters
        ----------
        layers : Layer
            A linked layer object representing the computation
        """
        self.layers = layers

    def compile(self, integrated=True):
        """
        Creates computational graph, sorts it by computation order and iterates through
        layers, compiling them in computational order.

        Parameters
        ----------
        integrated : bool, optional
            If true, results for this model are integrated over wavelengths at the end
            to calculate a total rate, by default True
        """
        # Change setting dependent if method is spectral or integrated
        self.integrated = integrated
        # Build computational and physical graph
        self.c_graph = utils.create_computational_graph(self.layers)
        self.p_graph = utils.create_physical_graph(self.layers)
        # Compile all layers comprising the model
        ts = graphlib.TopologicalSorter(self.p_graph).static_order()
        for layer in ts:
            layer.compile()

    def summary(self):
        """
        Returns graph in physical order (emitter>atmosphere>telescope)

        Returns
        -------
        Dictionary
            Dicttionary representing physical propagation of light.
        """
        return self.p_graph

    def predict(self, frame):
        """
        Computes layers in computational order for an observational frame.

        Parameters
        ----------
        frame : Frame
            Observational frame

        Returns
        -------
        numpy.array
            Rates per pixel in p.e./s, either integrated over all wavelengths if self.integrated=True
            or per wavelength specified in frame.obswl
        """
        ts = graphlib.TopologicalSorter(utils.reverse_graph(self.c_graph))
        data_dict = defaultdict(list)

        x_a, y_a = self.layers.camera.pix_pos[:, 1], self.layers.camera.pix_pos[:, 0]
        frame.pix_coord = SkyCoord(
            x_a, y_a, unit="rad", frame=frame.telframe
        ).transform_to(frame.AltAz)
        frame.pix_radii = self.layers.camera.pix_rad

        results = []
        for node in tuple(ts.static_order()):
            res = node(frame, data_dict.pop(node, None))
            if type(res) != list:
                res = [res]
            if len(self.c_graph[node]) == 0:
                results.extend(res)
            else:
                for j in self.c_graph[node]:
                    data_dict[j].extend(res)

        if self.integrated:
            for r in results:
                r.weight = si.simpson(r.weight, x=frame.obswl.to(u.nm).value, axis=1)

        comb = functools.reduce(lambda a, b: a + b, results)
        return comb


class PhotonMap:
    def __init__(self, layer, radius):
        """
        A photon map connects two layers with different propagation direction.

        Parameters
        ----------
        layer : Layer
            A (linked) layer
        radius : astropy.coordinates.Angle
            The maximum angle between forward/backward rays under which they get connected.
        """
        self.mode = "backward"
        self.layer = layer
        self.parents = layer.parents
        self.radius = radius

        # Set forwards & backwards options to photonmap function
        self.forward = self.photonmap
        self.backward = self.photonmap

        self.call_forward = self.photonmap
        self.call_backward = self.photonmap

    def compile(self):
        return None

    def photonmap(self, frame, rays):
        """
        Function connecting forward & backward layer together.

        Parameters
        ----------
        frame : Frame
            Observational frame
        rays : Rays
            A list of rays with their direction specified

        Returns
        -------
        Rays
            Weighted rays after connecting forward&backward rays
        """
        forward = [x for x in rays if x.direction == "forward"]
        backward = [x for x in rays if x.direction == "backward"]

        forward = functools.reduce(lambda a, b: a + b, forward)
        backward = functools.reduce(lambda a, b: a + b, backward)

        balltree = self.generate_map(forward)
        lengths, ind = self.query_map(balltree, backward)
        new_rays = backward.repeat(lengths)
        ind_rays = forward[ind]
        new_rays.source = ind_rays.source
        f_weight = self.layer.evaluate(frame, new_rays, ind_rays)

        return new_rays * f_weight * ind_rays.weight

    def generate_map(self, rays):
        """
        Generates a haversine balltree to make query for connecting rays more efficient.

        Parameters
        ----------
        rays : Rays
            Rays for which haversine balltree is created

        Returns
        -------
        sklearn.neighbors.BallTree
            Haversine balltree containing all alt/az coordinates for rays.
        """
        az, alt = rays.coords.az.rad, rays.coords.alt.rad
        return BallTree(np.vstack([alt, az]).T, metric="haversine")

    def query_map(self, balltree, rays):
        """
        Query all postions in a balltree, creating a list of indices for each ray that queries.

        Parameters
        ----------
        balltree : sklearn.neighbors.BallTree
            Haversine Balltree containing position of rays
        rays : Rays
            Rays that query Haversine Balltree

        Returns
        -------
        Tuple
            Tuple containing a list of the amount of returned positions for each ray and a concatenated
            array of all queried indices.
        """
        az, alt = rays.coords.az.rad, rays.coords.alt.rad
        ind = balltree.query_radius(np.vstack([alt, az]).T, r=self.radius)
        lengths = [len(x) for x in ind]
        return lengths, np.concatenate(ind)


class Layer(metaclass=ABCMeta):
    def __init__(self, config, N=1, mode=None):
        """
        Main Class for all layers in a model. Each Layers gets passed a config that
        describes informs its build process

        Parameters
        ----------
        config : Dictionary
            Dictionary defining the build process for the Layer
        N : int, optional
            Oversampling of rays passing through the layer, by default 1
        mode : str, optional
            describes if layer is forward or backward propagating, by default None
        """
        self.config = config
        self.parents = []
        self.N = N
        self.mode = mode

    def __call__(self, parents):
        # Connect to parents and get their mode
        self.parents = parents
        parent_modes = [x.mode for x in self.parents]

        # Change own mode depending on the parents mode
        if len(set(parent_modes)) == 1:
            self.mode = parent_modes.pop()
        elif len(set(parent_modes)) == 2:
            self.mode = "bidirectional"
        return self

    def compile(self):
        return None

    @utils.multi_rays
    def call_forward(self, frame, rays):
        return self.forward(frame, rays)

    @utils.multi_rays
    def call_backward(self, frame, rays):
        return self.backward(frame, rays)

    @abstractmethod
    def forward(self, frame, rays):
        return NotImplementedError

    @abstractmethod
    def backward(self, frame, rays):
        return NotImplementedError


class Transmission(Layer):
    def forward(self, frame, f_rays):
        t_args = self.t_args(frame, f_rays)
        return f_rays * self.transmission(*t_args)

    def backward(self, frame, b_rays):
        t_args = self.t_args(frame, b_rays)
        return b_rays * self.transmission(*t_args)

    @abstractmethod
    def transmission(self, *t_args):
        return NotImplementedError

    @abstractmethod
    def t_args(self, frame, rays):
        return NotImplementedError


class Scattering(Layer):
    def map(self, radius):
        """
        Creates a photon map for this scattering layer

        Parameters
        ----------
        radius : astropy.coordinates.Angle
            PhotonMap radius

        Returns
        -------
        PhotonMap
            PhotonMap for this layer
        """
        return PhotonMap(self, radius)

    def evaluate(self, frame, f_rays, b_rays):
        """
        Evaluates the scattering function based on the relation between forward and backward ray

        Parameters
        ----------
        frame : Frame
            Observation Frame
        f_rays : Rays
            Forward Rays object of length N
        b_rays : Rays
            Backward Rays object of length N

        Returns
        -------
        numpy.array
            Resulting weight for each scatter/transmission value
        """
        s_args = self.s_args(frame, f_rays, b_rays)
        t_args = self.t_args(frame, f_rays, b_rays)
        return self.scatter(*s_args)[:, np.newaxis] * self.transmission(*t_args)

    def forward(self, frame, f_rays):
        """
        Propagates Rays forwards through the scattering layer

        Parameters
        ----------
        frame : Frame
            Observation fra,e
        f_rays : Rays
            Forward rays

        Returns
        -------
        Rays
            Weighted Rays after transmitting through layer and scattering
        """
        if self.N > 1:
            s_args = self.s_args(frame, f_rays, None)
            pos, rho = utils.hist_sample(self.f_hist, s_args, self.N)
            b_rays = f_rays.directional_offset_by(pos, rho)

            t_args = self.t_args(frame, f_rays[b_rays.parent], b_rays)
            return b_rays * self.transmission(*t_args)
        else:
            t_args = self.t_args(frame, f_rays, f_rays)
            return f_rays * self.transmission(*t_args)

    def backward(self, frame, b_rays):
        """
        Propagates Rays backwards through scattering layer

        Parameters
        ----------
        frame : Frame
            Observation Frame
        b_rays : Rays
            Backwards rays

        Returns
        -------
        Rays
            Weighted Rays after transmitting through layer and scattering
        """
        if self.N > 1:
            s_args = self.s_args(frame, None, b_rays)
            pos, rho = utils.hist_sample(self.b_hist, s_args, self.N)
            f_rays = b_rays.directional_offset_by(pos, rho)

            t_args = self.t_args(frame, f_rays, b_rays[f_rays.parent])
            return f_rays * self.transmission(*t_args)
        else:
            t_args = self.t_args(frame, b_rays, b_rays)
            return b_rays * self.transmission(*t_args)

    def compile(self):
        """
        Build scattering histograms for forwards and backwards scattering
        """
        self.f_hist, self.b_hist = None, None
        if self.mode == "forward" or self.mode == "bidirectional":
            self.f_hist = self._build_hist("forward", self.config["bins"])
        if self.mode == "backward" or self.mode == "bidirectional":
            self.b_hist = self._build_hist("backward", self.config["bins"])

    @abstractmethod
    def scatter(self):
        return NotImplementedError

    @abstractmethod
    def transmission(self, *t_args):
        return NotImplementedError

    @abstractmethod
    def _build_hist(self, direction, bins):
        return NotImplementedError

    @abstractmethod
    def s_args(self, frame, f_rays, b_rays):
        return NotImplementedError

    @abstractmethod
    def t_args(self, frame, f_rays, b_rays):
        return NotImplementedError
