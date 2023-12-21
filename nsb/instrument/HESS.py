import numpy as np
import histlite as hl
import astropy.units as u

import nsb.utils.bandpass as bandpass

from nsb.core.instrument import Camera, Optics
from ctapipe.instrument import CameraGeometry

# HESS Optic Descriptions from papers
class Cornils2003(Optics):   
    def psf(self, off_in, off_out, pos, rho):
        p = self.config['parameters']
        off = off_in/np.deg2rad(2.5)
        
        sigma = np.sqrt(p[0]**2 + p[1]*off**2)*1e-3
        
        return 1/(2*np.pi*sigma**2) * np.exp(-rho**2/(2*sigma**2))
    
    def transmission(self, lam):
        bp = bandpass.HESS1U()
        mirror = 108
        return bp(lam) * mirror * self.config['degradation']
    
# HESS Optic Descriptions from papers
class Cornils2003_asym(Optics):   
    def psf(self, off_in, off_out, pos, rho):
        p = self.config['parameters']
        sigma = p[0]*1e-3
        alpha = p[1]*1e-3
        
        off   = off_in/np.deg2rad(2.5)

        corr_t = 1-1/(1+np.exp(5*(np.abs(pos - np.pi)/np.pi - 0.75)))
        sigma = np.sqrt(sigma**2 + (alpha*off)**2*corr_t)
    
        return rho/(sigma**2) * np.exp(-rho**2/(2*sigma**2))
    
    def transmission(self, lam):
        bp = bandpass.HESS1U()
        mirror = 108
        return bp(lam) * mirror * self.config['degradation']

class HESS2012(Optics):
    def f_psf(self, rho, sigma):
        return 1/(2*np.pi*sigma**2) * np.exp(-rho**2/(2*sigma**2))

    def transmission(self, frame, f_rays, r_rays):
        mirror  = 0.8 * 614
        winston = 0.75
        qe      = 0.3
        return np.ones(f_rays.N)*mirror*winston*qe

    def psf(self, off_in, off_out, pos, rho):
        sigma = (self.p[0] + off_in/self.p[1])*1e-3
        return self.f_psf(rho, sigma)
    
class CT1(Camera):
    def __init__(self, N):
        ct1_conf = {'cam':CameraGeometry.from_name('HESS-I'),
                    'focal length':15.28*u.m,
                    'rotation': -0.25*u.deg}
        super().__init__(ct1_conf, N)