import numpy as np
import histlite as hl
from abc import abstractmethod

from nsb.core import Scattering
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