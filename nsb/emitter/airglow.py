import numpy as np
from abc import ABCMeta, abstractmethod
import astropy.units as u

from nsb.core import Ray
from nsb.core.emitter import Emitter, DiffuseEmitter
import nsb.utils.spectra as spectra

from astropy.coordinates import SkyCoord, EarthLocation, AltAz
   
class KS1991(DiffuseEmitter):
    def X(self, alt):
        return (1 - 0.96 * (np.sin(np.pi/2-alt))**2)**(-0.5)
    
    def evaluate(self, frame, rays):
        """
        Krisciunas et. al Sky-only-Model
        Brightness of the Sky at certain position without moon
        """
        X_zen = self.X(rays.coords.alt.rad)
        p = self.config['parameters']
        k = frame.conf['k']
        
        rays.source = type(self)
        weight =  p[0] * X_zen * 10**(-0.4 * k * (X_zen - 1))
        return rays*weight
    
class Noll2012(DiffuseEmitter):
    def compile(self):
        self.ag_spectra = spectra.eso_airglow()
    
    def SPF(self, lam):
        return self.ag_spectra(lam)
        
    def vanrhjin(self, z):
        r_rh = (6738/(6738+self.config['H']))
        return 1/(1-r_rh**2*np.sin(z)**2)**0.5
    
    def evaluate(self, frame, rays):
        """
        Brightness model based on the van Rhjin function for airglow
        """
        rays.source = type(self)
        weight =  self.vanrhjin(np.pi/2 - rays.coords.alt.rad)[:,np.newaxis] * self.SPF(frame.obswl.to(u.nm).value)
        
        sf_scale = (0.2 + 0.00614*frame.conf['sfu'])
        return rays*weight*sf_scale
