import numpy as np
import functools

import astropy.units as u
from astropy.coordinates import SkyCoord, AltAz, representation
from abc import ABCMeta, abstractmethod
from ctapipe.coordinates import CameraFrame
from sklearn.neighbors import BallTree
import scipy.integrate as si

import graphlib
import nsb.core.utils as utils
from collections import defaultdict

class Frame:
    def __init__(self, location, obstime, target, fov, **kwargs):
        self.AltAz  = AltAz(obstime=obstime, location=location)
        self.location = location
        self.target = target.transform_to(self.AltAz)
        self.time   = obstime
        self.fov    = fov
        self.conf   = kwargs
    
class Model(metaclass=ABCMeta):
    def __init__(self, layers):
        self.layers = layers
        
    def compile(self, method='integrated'):
        # Change setting dependent if method is spectral or integrated
        self.method = method
        # Build computational and physical graph
        self.c_graph = utils.create_computational_graph(self.layers)
        self.p_graph = utils.create_physical_graph(self.layers)
        # Compile all layers comprising the model
        ts = graphlib.TopologicalSorter(self.p_graph).static_order()
        for layer in ts:
            layer.compile()
            
    def summary(self):
        return self.p_graph
    
    def predict(self, frame):
        ts = graphlib.TopologicalSorter(utils.reverse_graph(self.c_graph))
        data_dict  = defaultdict(list)

        results = []
        for node in tuple(ts.static_order()):
            res = [node(frame, data_dict.pop(node, None))]
            if len(self.c_graph[node]) == 0:
                results.extend(res)
            else:
                for j in self.c_graph[node]:
                    data_dict[j].extend(res)
        
        comb = functools.reduce(lambda a,b: a+b, results)
        if self.method == 'integrated':
            comb.weight = si.simpson(comb.weight, x=frame.obswl.to(u.nm).value, axis=1)
        
        return comb
    

class PhotonMap:
    def __init__(self, layer, radius):
        self.mode   = 'backward'
        self.layer   = layer
        self.parents = layer.parents
        self.radius  = radius

        self.forward  = self.photonmap
        self.backward = self.photonmap
        
    def compile(self):
        return None
    
    def photonmap(self, frame, rays):
        forward  = [x for x in rays if x.direction=='forward']
        backward = [x for x in rays if x.direction=='backward']
        
        forward  = functools.reduce(lambda a,b:a+b, forward)
        backward = functools.reduce(lambda a,b:a+b, backward)

        balltree = self.generate_map(forward)
        lengths, ind = self.query_map(balltree, backward)
        new_rays = backward.repeat(lengths)
        ind_rays = forward[ind]
        new_rays.source = ind_rays.source
        f_weight = self.layer.evaluate(frame, new_rays, ind_rays)

            
        return new_rays*f_weight*ind_rays.weight
    
    def generate_map(self, rays):
        az, alt = rays.coords.az.rad, rays.coords.alt.rad
        return BallTree(np.vstack([alt, az]).T, metric='haversine')

    def query_map(self, balltree, rays):
        az, alt = rays.coords.az.rad, rays.coords.alt.rad
        ind = balltree.query_radius(np.vstack([alt, az]).T, r=self.radius)
        lengths = [len(x) for x in ind]
        return lengths, np.concatenate(ind)
    
    
class Layer(metaclass=ABCMeta):
    def __init__(self, config, N=1, mode=None):
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
            self.mode = 'bidirectional'
        return self
    
    def compile(self):
        return None
        
    @abstractmethod
    def forward(self, frame, rays):
        return NotImplementedError
    
    @abstractmethod
    def backward(self, frame, rays):
        return NotImplementedError
        
class Transmission(Layer):
    @utils.reduce_rays
    def forward(self, frame, f_rays):
        t_args = self.t_args(frame, f_rays)
        return f_rays * self.transmission(*t_args)
    
    @utils.reduce_rays
    def backward(self, frame, b_rays):
        t_args = self.t_args(frame, b_rays)
        return b_rays * self.transmission(*t_args)
    
    @abstractmethod
    def transmission(self, frame, *t_args):
        return NotImplementedError

    @abstractmethod
    def t_args(self, frame, rays):
        return NotImplementedError

class Scattering(Layer):      
    def map(self, radius):
        '''
        Create a PhotonMap for this scattering layer
        '''
        return PhotonMap(self, radius)
    
    def evaluate(self, frame, f_rays, b_rays):
        '''
        Evaluate the scatter & transmission function for rays
        '''
        s_args = self.s_args(frame, f_rays, b_rays)
        t_args = self.t_args(frame, f_rays, b_rays)
        return (self.scatter(*s_args)[:,np.newaxis]
                * self.transmission(*t_args))
    
    @utils.reduce_rays
    def forward(self, frame, f_rays):
        if self.N >1:
            s_args = self.s_args(frame, f_rays, None)
            pos, rho = utils.hist_sample(self.f_hist, s_args, self.N)
            b_rays = f_rays.directional_offset_by(pos, rho)
            
            t_args = self.t_args(frame, f_rays[b_rays.parent], b_rays)
            return (b_rays * 
                    self.transmission(*t_args))
        else:
            t_args = self.t_args(frame, f_rays, f_rays)
            return (f_rays * 
                    self.transmission(*t_args))
    
    @utils.reduce_rays
    def backward(self, frame, b_rays):
        if self.N >1:
            s_args = self.s_args(frame, None, b_rays)
            pos, rho = utils.hist_sample(self.b_hist, s_args, self.N)
            f_rays = b_rays.directional_offset_by(pos, rho)

            t_args = self.t_args(frame, f_rays, b_rays[f_rays.parent])
            return (f_rays * 
                    self.transmission(*t_args))
        else:
            t_args = self.t_args(frame, b_rays, b_rays)
            return (b_rays * 
                    self.transmission(*t_args))
        
    def compile(self):
        self.f_hist, self.b_hist = None, None
        if self.mode == 'forward' or self.mode == 'bidirectional':
            self.f_hist = self._build_hist('forward', self.config['bins'])
        if self.mode == 'backward' or self.mode == 'bidirectional':
            self.b_hist = self._build_hist('backward', self.config['bins'])
    
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