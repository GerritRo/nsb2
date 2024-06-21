import numpy as np
import astropy.units as u

from nsb.core import Ray
from nsb.core.emitter import Diffuse
from nsb.utils.formulas import blackbody

import astropy
import astropy.constants as c
import astropy.units as u
from scipy.interpolate import RegularGridInterpolator

class Masana2021(Diffuse):
    def compile(self):
        zod = np.genfromtxt('/home/gerritr/ECAP/nsb_simulation/nsb2/nsb/utils/assets/leinert_zodiac.csv', delimiter=",")
        fit_points = [zod[1:,0], zod[0,1:]]
        self.A = RegularGridInterpolator(points=fit_points, values=zod[1:,1:])
    
    def SPF(self, lam):
        E_p = c.h.value*c.c.value / lam
        return blackbody(lam, 5778) / blackbody(0.5e-6, 5778) / E_p
        
    def color_corr(self, lam, elon):
        elon_f = -0.3* (np.clip(elon, 30, 90)-30)/60 
        return 1 + (1.2+ elon_f[:, np.newaxis]) * np.log(lam/500)
        
    def evaluate(self, frame, rays):
        r_e = rays.transform_to('geocentrictrueecliptic')
        
        sun = astropy.coordinates.get_body("sun", frame.time)
        s_e = sun.transform_to('geocentrictrueecliptic')
        
        alpha = np.abs((r_e.coords.lon - s_e.lon).wrap_at(180*u.deg).deg)
        beta  = np.abs(r_e.coords.lat.deg)
        
        corr = self.color_corr(frame.obswl.to(u.nm).value, alpha)
        
        rays.source = type(self)
        weight = 1e-11*self.A(np.asarray([alpha, beta]).T)[:,np.newaxis] * corr * self.SPF(frame.obswl.to(u.m).value)
        return rays*weight