from .. import ASSETS_PATH
from nsb.core.emitter import Diffuse

import numpy as np
import healpy as hp
from scipy.interpolate import UnivariateSpline
import astropy.constants as c
import astropy.units as u

class Kawara2017(Diffuse):
    def compile(self):
        file = ASSETS_PATH+'IRIS_nohole_4_2048_v2.fits'
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