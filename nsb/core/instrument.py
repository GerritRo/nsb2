import pickle

from nsb.core.logic import Layer
from nsb.core.ray import Ray

import numpy as np
import numpy.lib.recfunctions as recfc
import astropy.units as u
from astropy.coordinates import SkyCoord
from sklearn.neighbors import BallTree
from scipy.interpolate import UnivariateSpline


class Instrument(Layer):
    def compile(self):
        """
        Calculates the emission coordinates in telescopic frame for this combination of camera and bandpass.
        """
        self.camera = self.config["camera"]
        self.bandpass = self.config["bandpass"]

        self.emit_coord = self.calc_emit_coord(self.N)

    def forward(self, frame, rays):
        return self.ray_to_res(frame, rays)

    def backward(self, frame, rays):
        return self.res_to_ray(frame)

    def ray_to_res(self, frame, rays):
        """
        Assigns a pixel to each ray and weights with bandpass and effective area of pixel for the ray location

        Parameters
        ----------
        frame : Frame
            Observation frame
        rays : Ray
            Forward rays to be evaluated

        Returns
        -------
        Ray
            Weighted ray with assigned pixels
        """
        res = self.camera.assign_response(frame, rays)
        return res * self.bandpass(frame.obswl.to(u.nm).value)

    def res_to_ray(self, frame):
        """
        Emits ray from telescope

        Parameters
        ----------
        frame : Frame
            Observation frame

        Returns
        -------
        Ray
            Emitted rays from telescope weighted with bandpass and pixel effective area.
        """
        x_a, y_a, w_a, p_a = self.emit_coord
        return Ray(
            SkyCoord(x_a, y_a, frame=frame.telframe).transform_to(frame.AltAz),
            np.vstack([w_a] * len(frame.obswl)).T
            * self.bandpass(frame.obswl.to(u.nm).value),
            p_a,
            direction="backward",
        )

    def calc_emit_coord(self, N):
        """
        Depending on N, this calculates for each pixel the coordinates of each ray to be emitted

        Parameters
        ----------
        N : int
            Amount of rays to be emitted for each pixel

        Returns
        -------
        tuple
            tuple with (longitude, latitude, weight, pixel id)
        """
        lon_a, lat_a, w_a, p_a = (
            np.asarray([]) * u.rad,
            np.asarray([]) * u.rad,
            np.asarray([]),
            np.asarray([]),
        )
        for i, pix in enumerate(self.camera.pixels):
            lon, lat, v = self.emit_from_hist(pix.response, N)
            lon_a = np.append(lon_a, lon)
            lat_a = np.append(lat_a, lat)
            w_a = np.append(w_a, v)
            p_a = np.append(p_a, np.array([i]).repeat(len(lon)))

        return lon_a, lat_a, w_a, p_a.astype(int)

    def emit_from_hist(self, h, N):
        """
        For a given histogram, this subsamples based on N by fusing bins

        Parameters
        ----------
        h : histlite.Histogram
            Effective area histogram of pixel
        N : int
            Determines emitted rays based on N**2, rounding up to the next instance of 2**i

        Returns
        -------
        tuple
            (longitude, latitude, weight)
        """
        N = int(len(h.values) / (2 ** np.ceil(np.log2(np.sqrt(N)))))
        h_reb = h.rebin(0, h.bins[0][::N]).rebin(1, h.bins[1][::N]) / N**2
        h_reb = h_reb * h_reb.volumes
        mgrid = np.meshgrid(h_reb.centers[0], h_reb.centers[1], indexing="ij")
        return (
            mgrid[0].flatten() * u.rad,
            mgrid[1].flatten() * u.rad,
            h_reb.values.flatten(),
        )


class Camera:
    """
    A camera is defined as a collection of pixels

    Parameters
    ----------
    pixels : list of Pixel objects
        Pixels constituting a camera
    """

    def __init__(self, pixels):
        self.pixels = pixels

        for pixel in pixels:
            pixel.spline_response = pixel.response.spline_fit(log=True)

        self.pix_pos = np.asarray([pix.position for pix in pixels])
        self.pix_rad = np.asarray([pix.radius for pix in pixels])

    def pix_assign(self, rays):
        """
        Adds together all rays belonging to the same pixel

        Parameters
        ----------
        rays : Ray
            Ray object

        Returns
        -------
        numpy.array
            array of length(pixels) with the ray weights added for each pixel
        """
        pix_id, weight = (rays.pixels[rays.pixels >= 0], rays.weight[rays.pixels >= 0])
        return np.bincount(pix_id, weight, minlength=len(self.pix_pos))

    def assign_response(self, frame, rays):
        """
        For a specific observation frame, this transforms the given rays
        into the camera frame and calculates the pixel response for each ray and assigns them a pixel.

        Parameters
        ----------
        frame : Frame
            Observation frame
        rays : Ray
            Incoming rays

        Returns
        -------
        Ray
            ray with pixel assigned and weighted by effective area of each pixel at hit point of ray.
        """
        prays = rays.transform_to(frame.telframe)
        lon, lat = prays.coords.lon.rad, prays.coords.lat.rad
        ray_coor = np.vstack([lat, lon]).T

        tree = BallTree(ray_coor, metric="haversine")
        ray_ind = tree.query_radius(self.pix_pos, r=self.pix_rad)

        # Assign all indices within range to pixel
        inds, weight = [], []
        for i, ind in enumerate(ray_ind):
            dxdy = ray_coor[ind]
            val = np.nan_to_num(self.pixels[i].spline_response(dxdy[:, 1], dxdy[:, 0]))
            inds.extend(ind)
            weight.extend(val)

        res = rays[inds] * np.asarray(weight)[:, np.newaxis]
        res.pixels = np.repeat(np.arange(len(self.pixels)), [len(x) for x in ray_ind])

        return res

    @classmethod
    def from_response(cls, file):
        with open(file, "rb") as pixels:
            return cls(pickle.load(pixels))


class Pixel:
    """
    A pixel with a position in the camera plane and a radius describing the maximum extent of its effective area

    Parameters
    ----------
    position : tuple
        (x,y) tuple giving the position in the camera plane
    radius : astropy.coordinates.Angle
        Angle given the maximum extent of the effective area as a radius
    """

    def __init__(self, position, radius):
        self.position = position
        self.radius = radius

    @property
    def response(self):
        return self._response

    @response.setter
    def response(self, hist):
        self._response = hist


class Bandpass:
    def __init__(self, lam, trx):
        """
        Bandpass that can be initialised with wavelengths and relative transmission. It the gets interpolated
        with a univariate spline. Calling the bandpass with a wavelength then returns the spline values

        Parameters
        ----------
        lam : numpy.array
            Numpy array of wavelengths
        trx : numpy.array
            Numpy array of transmission values
        """
        self.lam = lam
        self.trx = trx

        self.min = np.min(self.lam)
        self.max = np.max(self.lam)

        self.spline = UnivariateSpline(
            self.lam, np.array(self.trx.tolist()).prod(axis=1), s=0, ext=1
        )

    def __call__(self, lam):
        return self.spline(lam)

    @classmethod
    def from_csv(cls, file):
        """
        Generate a bandpass from a csv file. The wavelength column should be marked 'lam', all others
        are assumed to be components of the bandpass and multiplied.

        Parameters
        ----------
        file : csv like file
            CSV file describing the bandpass

        Returns
        -------
        Bandpass
            Bandpass objects gained from the csv file.
        """
        arr = np.genfromtxt(file, delimiter=",", names=True)
        return cls(arr["lam"], recfc.drop_fields(arr, "lam", usemask=False))
