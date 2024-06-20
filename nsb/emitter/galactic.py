import numpy as np
import pandas as pd
import healpy as hp

from nsb.core import Ray
from nsb.core.emitter import Diffuse
from nsb.utils.formulas import blackbody, ballesteros
from nsb.utils import bandpass
from scipy.interpolate import UnivariateSpline
import scipy.integrate as integrate
import astropy
import astropy.constants as c
import astropy.units as u

class Kawara2017(Diffuse):
    def compile(self):
        file = '/home/gerritr/ECAP/nsb_simulation/nsb2/nsb/utils/assets/IRIS_nohole_4_2048_v2.fits'
        self.hp_map =  np.clip(hp.read_map(file, None)-0.8, 0, self.config['i100_max'])
        self.NSIDE = hp.npix2nside(len(self.hp_map))
        
        lam = np.asarray([0.23, 0.27, 0.32, 0.37, 0.42, 0.47, 0.55, 0.65]) 
        w   = np.asarray([0.04, 0.05, 0.05, 0.05, 0.05, 0.06, 0.1, 0.1])*1e-6
        b_i = np.asarray([3.0, 3.9, 6.1, 8.5, 13.6, 17.5, 20.1, 21.0])
        c_i = np.asarray([0.1, 0.3, 0.7, 1.1, 2.4, 3.3, 4.4, 4.5])*1e-5
        
        self.b_i = UnivariateSpline(lam, b_i/w, k=1, s=0)
        self.c_i = UnivariateSpline(lam, c_i/w, k=1, s=0)
        
    def SPF(self, lam, dust):
        E_p = c.h.value*c.c.value / (lam*1e-6)
        
        f1 = self.b_i(lam) * 1e-9 / E_p
        f2 = self.c_i(lam) * 3000/lam * 1e-9  / E_p
        return np.clip(dust[:, np.newaxis] * f1 - (dust**2)[:, np.newaxis] * f2, 0, None)
        
    def evaluate(self, frame, rays):
        r_g = rays.transform_to('galactic')
        pix = hp.ang2pix(self.NSIDE, r_g.coords.l.deg, r_g.coords.b.deg, lonlat=True)
        dust = self.hp_map[pix]
        
        rays.source = type(self)
        weight = 1e-9*self.SPF(frame.obswl.to(u.micron).value, dust)
        return rays*weight
    
class GaiaDR3Mag15(Diffuse):
    def compile(self):
        spath = '/home/gerritr/ECAP/nsb_simulation/nsb2/nsb/utils/assets/'
        self.m_b = -2.5*np.log10(np.load(spath + 'gaia_m_b_15plus.npy')+1e-31)
        self.m_g = -2.5*np.log10(np.load(spath + 'gaia_m_g_15plus.npy')+1e-31)
        self.NSIDE = hp.npix2nside(len(self.m_g))
        
        self.T = ballesteros(np.clip(self.m_b - self.m_g, -0.45, None),
                                     4261, 0.77, 12.0, 0.445)
        
        self.norm = self._norm_mag_spectrum()
        
    def SPF(self, lam, T):
        E_p = c.h.value*c.c.value / lam
        return blackbody(lam, T[:,np.newaxis]) / self.norm(T[:,np.newaxis]) / E_p
    
    def calc_flux(self, ind, lam):
        M = self.m_g[ind]
        T = self.T[ind]
        
        return 10**(-0.4*M[:,np.newaxis])*self.config['Mag_0'] * self.SPF(lam, T)
    
    def _norm_mag_spectrum(self):
        x = np.linspace(500, 45000, 50)
        y = []
        s = bandpass.GaiaDR3()        
        
        for T_eff in x:
            y.append(integrate.quad(lambda lam: s(lam)*blackbody(1e-9*lam, T_eff), 250, 1100, limit=150)[0])
            
        return UnivariateSpline(x, y, s=0, ext=1)
        
    def evaluate(self, frame, rays):
        r_g = rays.transform_to('icrs')
        pix = hp.ang2pix(self.NSIDE, r_g.coords.ra.deg, r_g.coords.dec.deg, lonlat=True, nest=True)
        corr = hp.nside2pixarea(self.NSIDE)
        
        weight = self.calc_flux(pix, frame.obswl.to(u.m).value)/corr
        
        rays.source = type(self)
        return rays*weight