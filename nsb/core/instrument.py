import pickle

from nsb.core.logic import Layer
from nsb.core.ray import Ray
from nsb.core.utils import reduce_rays, haversine, hist_sample, sq_solid_angle

import numpy as np
import numpy.lib.recfunctions as recfc
import histlite as hl
import astropy.units as u
from astropy.coordinates import angular_separation, position_angle
from astropy.coordinates import SkyCoord, offset_by
from sklearn.neighbors import BallTree
from scipy.interpolate import UnivariateSpline

class Instrument(Layer):
    def compile(self):
        self.camera = self.config['camera']
        self.bandpass = self.config['bandpass']

        self.emit_coord = self.calc_emit_coord(self.N)

    def forward(self, frame, rays):
        return self.ray_to_res(frame, rays)

    def backward(self, frame, rays):
        return self.res_to_ray(frame)

    def ray_to_res(self, frame, rays):
        inds, res = self.camera.assign_response(frame, rays)
        return res * self.bandpass(frame.obswl.to(u.nm).value)

    def res_to_ray(self, frame):
        x_a, y_a, w_a, p_a = self.emit_coord
        return Ray(SkyCoord(x_a, y_a, frame=frame.telframe).transform_to(frame.AltAz),
                   np.vstack([w_a]*len(frame.obswl)).T * self.bandpass(frame.obswl.to(u.nm).value),
                   p_a,
                   direction='backward')
        
    def calc_emit_coord(self, N):
        lon_a, lat_a, w_a, p_a = np.asarray([])*u.rad, np.asarray([])*u.rad, np.asarray([]), np.asarray([])
        for i, pix in enumerate(self.camera.pixels):
            lon, lat, v = self.emit_from_hist(pix.response, N)
            lon_a = np.append(lon_a, lon)
            lat_a = np.append(lat_a, lat)
            w_a = np.append(w_a, v)
            p_a = np.append(p_a, np.array([i]).repeat(len(lon)))

        return lon_a, lat_a, w_a, p_a.astype(int)
    
    def emit_from_hist(self, h, N):
        N = int(len(h.values)/(2**np.ceil(np.log2(np.sqrt(N)))))
        h_reb = h.rebin(0, h.bins[0][::N]).rebin(1, h.bins[1][::N])/N**2
        h_reb = h_reb * h_reb.volumes
        mgrid = np.meshgrid(h_reb.centers[0], h_reb.centers[1], indexing='ij')
        return mgrid[0].flatten()*u.rad, mgrid[1].flatten()*u.rad, h_reb.values.flatten()

class Camera():
    def __init__(self, pixels):
        self.pixels = pixels

        for pixel in pixels:
            pixel.spline_response = pixel.response.spline_fit(log=True)

        self.pix_pos = np.asarray([pix.position for pix in pixels])
        self.pix_rad = np.asarray([pix.radius for pix in pixels])

    def pix_assign(self, rays):
        pix_id, weight = (rays.pixels[rays.pixels>=0], rays.weight[rays.pixels>=0])
        return np.bincount(pix_id, weight, minlength=len(self.pix_pos))

    def assign_response(self, frame, rays):
        prays = rays.transform_to(frame.telframe)
        lon, lat = prays.coords.lon.rad, prays.coords.lat.rad
        ray_coor = np.vstack([lat, lon]).T
        
        tree = BallTree(ray_coor, metric='haversine')
        ray_ind = tree.query_radius(self.pix_pos, r=self.pix_rad)
        
        # Assign all indices within range to pixel
        inds, weight = [], []
        for i, ind in enumerate(ray_ind):
            dxdy = ray_coor[ind]
            val = np.nan_to_num(self.pixels[i].spline_response(dxdy[:,1], dxdy[:,0]))
            inds.extend(ind)
            weight.extend(val)

        res = rays[inds] * np.asarray(weight)[:, np.newaxis]
        res.pixels = np.repeat(np.arange(len(self.pixels)), [len(x) for x in ray_ind])
        
        return inds, res
    
    @classmethod
    def from_response(cls, file):
        with open(file, 'rb') as pixels:
            return cls(pickle.load(pixels))
        
    @classmethod
    def from_ctapipe(cls, camera_geometry, psf_hist, mirror_area, focal_length, d_grid=32):
        # Setting some sampling values that should be dynamically calculated:
        N = 500
        s_rad = 1.5*psf_hist.bins[-1][-1]
        
        # Ctapipe implicitly assumes gnonomic (or small angle) approximation
        def gnonomic(x, y, inverse=False):
            if inverse == False:
                return np.tan(x), np.tan(y)/np.cos(x)
            else:
                return np.arctan(x), np.arctan(y/np.sqrt(1+x**2))

        def create_grid(r, N):
            arr_edge = np.linspace(-r, r, N)
            d = (arr_edge[1]-arr_edge[0])/2
            test_grid = np.meshgrid(arr_edge, arr_edge)
            return test_grid[0].flatten(), test_grid[1].flatten(), np.append(arr_edge[0]-d, arr_edge+d)

        def is_inside(x, y, r, shape):
            x_a, y_a = np.abs(x), np.abs(y)
            if shape == 'hexagon':
                h, v = r*np.cos(np.pi/6), r/2
                return ((x_a < h) & (y_a < 2*v)) & (2*v*h - v*x_a - h*y_a >= 0)
            if shape == 'square':
                return (x_a < r/np.sqrt(2)) & (y_a < r/np.sqrt(2))
            if shape == 'circle':
                return np.sqrt(x**2 + y**2) < r
        
        pixels = []
        for i in camera_geometry.pix_id:
            pix_x, pix_y = camera_geometry.pix_x[i].value, camera_geometry.pix_y[i].value
            pix_lon, pix_lat = gnonomic(pix_y/focal_length, pix_x/focal_length, inverse=True)
            # Create as grid of points to sample from:
            lon, lat, edges = create_grid(s_rad, d_grid)
            # Sample psf:
            pos, rho = hist_sample(psf_hist, haversine(pix_lon+lon, pix_lat+lat, 0), N)
            # Offset coordinates:
            pos_samp = position_angle(pix_lon+lon, pix_lat+lat, 0, 0)
            s_lon, s_lat = offset_by(pix_lon+lon[:,np.newaxis], pix_lat+lat[:,np.newaxis], pos_samp[:,np.newaxis].rad-pos+np.pi, rho)
            # Translate to gnonomic and check percentage inside pixel
            x, y = gnonomic(s_lon, s_lat)
            circum_rad = camera_geometry._pixel_circumradius[i].value
            shape = camera_geometry.pix_type.value
            inside = np.sum(is_inside(y*focal_length-pix_x, x*focal_length-pix_y, circum_rad, shape), axis=1)
            
            # Create pixel and assign radius to be region where 99% of rays are in
            rad99 = np.max(np.sqrt(lon**2+lat**2)[inside > 0.01*N])
            pix = Pixel([pix_lat, pix_lon], min(rad99, s_rad))
            pix.response = hl.Hist([pix_lon + edges, pix_lat+edges], inside.reshape((d_grid, d_grid)).T / N)*mirror_area
            pixels.append(pix)

        return cls(pixels)

class Pixel():
    def __init__(self, position, radius):
        self.position = position
        self.radius = radius

    @property
    def response(self):
        return self._response

    @response.setter
    def response(self, hist):
        self._response = hist

class Bandpass():
    def __init__(self, lam, trx):
        self.lam = lam
        self.trx = trx

        self.min = np.min(self.lam)
        self.max = np.max(self.lam)

        self.spline = UnivariateSpline(self.lam, np.array(self.trx.tolist()).prod(axis=1), s=0, ext=1)

    def __call__(self, lam):
        return self.spline(lam)

    @classmethod
    def from_csv(cls, file):
        arr = np.genfromtxt(file, delimiter=",", names=True)
        return cls(arr['lam'], recfc.drop_fields(arr, "lam", usemask=False))