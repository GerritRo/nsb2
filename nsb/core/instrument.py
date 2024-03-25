import numpy as np
import histlite as hl
import matplotlib.pyplot as plt
from abc import abstractmethod
import astropy.units as u
from scipy.spatial import KDTree

from nsb.core.logic import Layer, Scattering
from nsb.core.ray import Ray
from nsb.core.utils import reduce_rays, haversine, hist_sample, sq_solid_angle
from nsb.utils.quad import QuadScheme

from astropy.coordinates import SkyCoord
from ctapipe.coordinates import CameraFrame
from sklearn.neighbors import BallTree
from ctapipe.instrument import CameraGeometry
from ctapipe.visualization import CameraDisplay

class Instrument(Layer):
    def compile(self):
        self.camera = self.config['camera']
        self.mirror = self.config['mirror']

        if 'psf_hist' in self.config:
            self.compute_response(self.config['psf_hist'], self.config['compute_parameters'])

        self.emit_coord = self.calc_emit_coord()

    @reduce_rays
    def forward(self, frame, rays):
        return self.ray_to_res(self.calc_cam_frame(frame), rays)

    def backward(self, frame, rays):
        return self.res_to_ray(self.calc_cam_frame(frame))

    def calc_cam_frame(self, frame):
        '''
        Calculate camera frame
        '''
        c_frame = CameraFrame(
            telescope_pointing=frame.target,
            focal_length=self.mirror.focal,
            obstime=frame.time,
            location=frame.location,
            rotation=self.camera.rotate
        )
        c_frame.AltAz = frame.AltAz
        c_frame.obswl = frame.obswl
        return c_frame

    def ray_to_res(self, frame, rays):
        prays = rays.transform_to(frame)
        ray_coor = np.vstack([prays.coords.x.value, prays.coords.y.value]).T
    
        tree = BallTree(ray_coor)
        ray_ind = tree.query_radius(self.camera.pix_pos, r=self.config['compute_parameters'][1]*self.camera.pix_rad)
        
        # Assign all indices within range to pixel
        inds, weight = [], []
        for i, ind in enumerate(ray_ind):
            dxdy = ray_coor[ind]-self.camera.pix_pos[i]
            val = self.camera.pixels[i].response(dxdy[:,0], dxdy[:,1])
            inds.extend(ind)
            weight.extend(val)
        
        res = rays[inds] * np.asarray(weight)[:, np.newaxis]
        res.pixels = np.repeat(np.arange(len(self.camera.pixels)), [len(x) for x in ray_ind])
        return res * self.mirror.bandpass(frame.obswl.to(u.nm).value) * self.mirror.area.value

    def res_to_ray(self, frame):
        x_a, y_a, w_a, p_a = self.emit_coord
        coord = SkyCoord(x_a, y_a, frame=frame).transform_to(frame.AltAz)
        return Ray(SkyCoord(alt = coord.alt, az=coord.az, frame=frame.AltAz),
                   np.vstack([w_a]*len(frame.obswl)).T * self.mirror.bandpass(frame.obswl.to(u.nm).value),
                   p_a,
                   direction='backward')
        
    def calc_emit_coord(self):
        x_a, y_a, w_a, p_a = np.asarray([])*u.m, np.asarray([])*u.m, np.asarray([]), np.asarray([])
        for i, pix in enumerate(self.camera.pixels):
            x, y, v = self.emit_from_hist(pix.response, self.N)
            x_a = np.append(x_a, x+pix.position[0]*u.m)
            y_a = np.append(y_a, y+pix.position[1]*u.m)
            w_a = np.append(w_a, v * self.mirror.area.value)
            p_a = np.append(p_a, np.array([i]).repeat(len(x)))

        return x_a, y_a, w_a, p_a.astype(int)
    
    def emit_from_hist(self, h, N):
        N = int(len(h.values)/(2**np.ceil(np.log2(np.sqrt(N)))))
        h_reb = h.rebin(0, h.bins[0][::N]).rebin(1, h.bins[1][::N])/N**2
        h_reb = h_reb * sq_solid_angle(h_reb.volumes, self.mirror.focal.value)
        mgrid = np.meshgrid(h_reb.centers[0], h_reb.centers[1], indexing='ij')
        return mgrid[0].flatten()*u.m, mgrid[1].flatten()*u.m, h_reb.values.flatten()
        
    def compute_response(self, psf_hist, parameters):
        dd_grid  = parameters[0]
        max_frad = parameters[1]
        N = parameters[2]

        def calc_vals(x, y, f):
            return np.arctan(np.sqrt(x**2+y**2) / f), np.arctan2(y,x)
        
        for pix in self.camera.pixels:
            if not hasattr(pix, 'response'):
                x, y, edges = pix.pixel_grid(max_frad*pix.radius, dd_grid)
            
                off, tht = calc_vals(x + pix.position[0], y + pix.position[1], self.mirror.focal.value)
                pos, rho = hist_sample(psf_hist, off, N)
            
                x_n = x[:,np.newaxis] + np.tan(rho)*self.mirror.focal.value*np.cos(pos)
                y_n = y[:,np.newaxis] + np.tan(rho)*self.mirror.focal.value*np.sin(pos)
            
                pix.response = hl.Hist([edges, edges], np.sum(pix.is_inside(x_n, y_n), axis=1).reshape((dd_grid,dd_grid)).T / N)

class Camera():
    def __init__(self, pixels, rotation):
        self.pixels = pixels
        self.rotate = rotation

        self.pix_pos = np.asarray([pix.position for pix in pixels])
        self.pix_rad = np.asarray([pix.radius for pix in pixels])

    def xy_to_pixel(self, x, y):
        tree = BallTree(np.vstack([x, y]).T)
        ind = tree.query_radius(self.pix_pos, r=self.pix_rad)
        for i, ii in enumerate(ind):
            rays.pixels[ii[pix[i].is_inside(x[ii], y[ii])]] = i
        return rays

    def pix_assign(self, rays):
        pix_id, weight = (rays.pixels[rays.pixels>=0], rays.weight[rays.pixels>=0])
        return np.bincount(pix_id, weight, minlength=len(self.pix_pos))

class Pixel():
    def __init__(self, position, radius, shape):
        self.position = position
        self.radius = radius
        self.shape  = shape

    def is_inside(self, x, y):
        x_a, y_a = np.abs(x), np.abs(y)
        if self.shape == 'hexagon':
            h, v = self.radius*np.cos(np.pi/12), self.radius/2
            return ((x_a < h) & (y_a < 2*v)) & (2*v*h - v*x_a - h*y_a >= 0)
        if self.shape == 'square':
            return (x_a < self.radius/np.sqrt(2)) & (y_a < self.radius/np.sqrt(2))
        if self.shape == 'circle':
            return np.sqrt(x**2 + y**2) < self.radius

    def pixel_grid(self, r, N):
        arr_edge = np.linspace(-r, r, N)
        d = (arr_edge[1]-arr_edge[0])/2
        test_grid = np.meshgrid(arr_edge, arr_edge)
        return test_grid[0].flatten(), test_grid[1].flatten(), np.append(arr_edge[0]-d, arr_edge+d)

    @property
    def response(self):
        return self._response

    @response.setter
    def response(self, hist):
        self._response = hist

class Mirror():
    def __init__(self, focal_length, area):
        self.focal = focal_length
        self.area  = area

    @property
    def bandpass(self):
        return self._spline

    @bandpass.setter
    def bandpass(self, spline):
        self._spline = spline