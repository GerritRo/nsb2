import numpy as np
import histlite as hl

from nsb.core.logic import Layer
import nsb.core.utils as utils

class RadialScattering(Scattering):
    def scatter(self, rho):
        return self.indicatrix(rho)
    
    def transmission(self, *args):
        return self.gradation(*args)
    
    @abstractmethod
    def indicatrix(self, rho):
        return NotImplementedError

    @abstractmethod
    def gradation(self, *args):
        return NotImplementedError

    def _build_hist(self, direction, bins):
        def f(rho):
            return np.sin(rho)*self.indicatrix(rho)
        return hl.hist_from_eval(f, bins=bins).normalize()
    
    def s_args(self, frame, f_rays, b_rays):
        if f_rays == None:
            return (b_rays.N)
        elif b_rays == None:
            return (f_rays.N)
        else:
            return (f_rays.separation(b_rays).rad,)
        
    @abstractmethod
    def t_args(self, frame, f_rays, b_rays):
        return NotImplementedError

class OffAxisScattering(Scattering):
    def scatter(self, off, pos, rho):
        return self.psf(off, pos, rho)

    @abstractmethod
    def psf(self, off, pos, rho):
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
                    f_rays.position_angle(b_rays).rad,
                    f_rays.separation(b_rays).rad)
    
    def _build_hist(self, direction, bins):
        if direction == 'forward':
            def f(off_in, pos, rho):
                return np.sin(rho)*self.psf(off_in, pos, rho)
        elif direction == 'backward':
            def f(off_out, pos, rho):
                off_in = np.pi/2-haversine(pos, np.pi/2 - rho, off_out)
                return np.sin(rho)*self.psf(off_in, pos, rho)

        return hl.hist_from_eval(f, bins=bins)

    @abstractmethod
    def t_args(self, frame, f_rays, b_rays):
        return NotImplementedError