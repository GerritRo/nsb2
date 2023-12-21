import numpy as np
import histlite as hl
import matplotlib.pyplot as plt
from abc import abstractmethod
import astropy.units as u

from nsb.core.logic import Layer, Scattering
from nsb.core.ray import Ray
from nsb.core.utils import reduce_rays, haversine
from nsb.utils.quad import QuadScheme

from astropy.coordinates import SkyCoord
from ctapipe.coordinates import CameraFrame
from ctapipe.instrument import CameraGeometry
from ctapipe.visualization import CameraDisplay

class Camera(Layer):      
    def calc_cam_frame(self, frame):
        '''
        Calculate camera frame
        '''
        c_frame = CameraFrame(
            telescope_pointing=frame.target,
            focal_length=self.config['focal length'],
            obstime=frame.time,
            location=frame.location,
            rotation=self.config['rotation']
        )
        c_frame.AltAz = frame.AltAz
        c_frame.obswl = frame.obswl
        return c_frame
    
    def calc_solid_angle(self):
        return 6.5432e-6

    @reduce_rays
    def forward(self, frame, f_rays):
        return self.ray_to_pixel(self.calc_cam_frame(frame), f_rays)
    
    def backward(self, frame, b_rays):
        return self.pixel_to_ray(self.calc_cam_frame(frame))
    
    def compile(self):
        # Getting a quadrature scheme based on pixelshape
        self.scheme = QuadScheme(self.N, self.config['cam'])
        # Get x and y position of pixels and oversample
        x, y = self.config['cam'].pix_x.value, self.config['cam'].pix_y.value
        self.coo = self.scheme.oversample(x, y)
    
    def ray_to_pixel(self, frame, rays):
        prays = rays.transform_to(frame)
        x, y = prays.coords.x, prays.coords.y
        unit = x.unit
        coor = np.dstack([x.to_value(unit), y.to_value(unit)])
        circum_rad = self.config['cam']._pixel_circumradius[0].to_value(unit)
        inner_rad  = self.config['cam'].pixel_width[0].to_value(unit)/2
        dist, pix_indices = self.config['cam']._kdtree.query(coor, distance_upper_bound=circum_rad)
        # 1. Mark all points outside pixel circumference as lying outside camera
        pix_indices[pix_indices == self.config['cam'].n_pixels] = -1
        # 2. Get all pixels that are assigned to border pixel and not within inner bounding circle
        border_mask = self.config['cam'].get_border_pixel_mask()
        m = np.isin(pix_indices, np.where(border_mask)[0]) & (dist>inner_rad)
        i = np.nonzero(m)
        # 3. Shift to non-border pixel:
        insidepix_index = np.where(~border_mask)[0][0]
        subtr = np.asarray([self.config['cam'].pix_x[pix_indices[m]].value, self.config['cam'].pix_y[pix_indices[m]].value]).T
        shift = np.asarray([self.config['cam'].pix_x[insidepix_index].value, self.config['cam'].pix_y[insidepix_index].value])
        coor_prime = (coor[m]-subtr+shift)
        # 4. Check with points shifted towards inside pixel if inside camera:
        dist_check, index_check = self.config['cam']._kdtree.query(coor_prime, distance_upper_bound=circum_rad)
        pix_indices[i[0][index_check != insidepix_index], i[1][index_check != insidepix_index]] = -1
        rays.pixels = pix_indices.flatten()
        return rays

    def pixel_to_ray(self, frame):
        '''
        This function uses quadpy to emit rays from each pixel shape, oversampling using quadrature
        '''
        # Translate into SkyCoords
        coords = SkyCoord(self.coo[:,0,:]*u.m, self.coo[:,1,:]*u.m, frame=frame).transform_to(frame.AltAz)
        weight = np.tile(self.scheme.weights, coords.shape[0])*self.calc_solid_angle()
        pixels = np.arange(coords.shape[0]).repeat(coords.shape[-1])
        return Ray(coords.flatten(), 
                   np.vstack([weight.flatten()]*len(frame.obswl)).T, 
                   pixels.flatten(), 
                   direction='backward')

    def pix_assign(self, rays):
        '''
        Sums weights over all rays that fall into a pixel
        '''
        tot_pix  = self.config['cam'].pix_x.shape[0]
        pix_mask = (rays.pixels>=0)

        res = np.bincount(rays.pixels[pix_mask],
                          weights=rays.weight[pix_mask],
                          minlength=tot_pix)
        return res

    def display(self, rays, ax, label='a.u.', **kwargs):
        '''
        Displays camera response to a group of rays
        '''
        display = CameraDisplay(self.config['cam'], ax=ax, **kwargs)
        display.image = self.pix_assign(rays)
        display.add_colorbar(label=label)
        return display

    
class Optics(Scattering):
    def scatter(self, off_in, off_out, pos, rho):
        return self.psf(off_in, off_out, pos, rho)

    @abstractmethod
    def psf(self, off_in, off_out, pos, rho):
        return NotImplementedError
    
    @abstractmethod
    def transmission(self, frame, f_rays, r_rays):
        return NotImplementedError
    
    def calc_offset(self, frame, rays):
        sep_optical = rays.coords.separation(frame.target)
        return sep_optical
    
    def s_args(self, frame, f_rays, b_rays):
        if f_rays == None:
            return self.calc_offset(frame, b_rays).rad
        elif b_rays == None:
            return self.calc_offset(frame, f_rays).rad
        else:
            return (self.calc_offset(frame, f_rays).rad,
                    self.calc_offset(frame, b_rays).rad,
                    f_rays.position_angle(b_rays).rad,
                    f_rays.separation(b_rays).rad)
            
    def t_args(self, frame, f_rays, b_rays):
        lam = frame.obswl.to(u.nm).value
        return lam,
    
    def _build_hist(self, direction, bins):
        if direction == 'forward':
            def f(off_in, pos, rho):
                off_out = np.pi/2-haversine(pos, np.pi/2 - rho, off_in)
                return np.sin(rho)*self.psf(off_in, off_out, pos, rho)
        elif direction == 'backward':
            def f(off_out, pos, rho):
                off_in = np.pi/2-haversine(pos, np.pi/2 - rho, off_out)
                return np.sin(rho)*self.psf(off_in, off_out, pos, rho)

        return hl.hist_from_eval(f, bins=bins)
