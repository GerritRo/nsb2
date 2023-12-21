from abc import abstractmethod
import numpy as np
import astropy.units as u
import histlite as hl

from nsb.core import Ray
from nsb.core.emitter import Emitter
from nsb.utils.formulas import blackbody

from scipy.interpolate import UnivariateSpline

import astropy
import astropy.units as u
import astropy.constants as c

class KS1991(Emitter):
    def emit(self, frame):
        sun   = astropy.coordinates.get_sun(frame.time)
        moon  = astropy.coordinates.get_body("moon", frame.time)

        alpha = astropy.coordinates.Angle('180°') - moon.separation(sun)
        Vmag  = -12.73 + 0.026 * np.abs(alpha.deg) + 4*10**(-9) * alpha.deg**4
        moon_alt = moon.transform_to(frame.AltAz).alt
        z_switch = 1 if moon_alt > 0*u.deg else 0
        
        moon = astropy.coordinates.get_body("moon",
                                            frame.time).transform_to(frame.AltAz)
        
        return Ray(moon.reshape(1), 
                   weight=np.asarray([10**(-0.4*(Vmag)) * 8.79 * 1e9 * z_switch]), 
                   source=type(self), 
                   direction='forward')

    
class Jones2013(Emitter):
    def compile(self):
        self.rol = np.genfromtxt('/home/gerritr/ECAP/nsb_simulation/nsb_refactor/nsb/utils/assets/rolo.csv', delimiter=",")
    
    def SPF(self, lam):
        '''
        Returns the photon flux for an emitter based on absolute
        flux normalization and wavelength
        '''
        E_p = c.h.value*c.c.value / lam
        return blackbody(lam, np.asarray([5778])[:,np.newaxis]) / E_p
    
    def lnA(self, p, g, s_sel):
        p_1, p_2, p_3, p_4 = 4.06054, 12.8802, np.deg2rad(-30.5858), np.deg2rad(16.7498)

        sum_a = p[0] + p[1]*g + p[2]*g**2 + p[3]*g**3
        sum_b = p[4]*s_sel + p[5]*s_sel**3 + p[6]*s_sel**5
        sum_c = p[7]*np.exp(-g/p_1) + p[8]*np.exp(-g/p_2) + p[9]*np.cos((g-p_3)/p_4)

        return sum_a + sum_b + sum_c
        
    def calc_norm(self, lam, moon_dist, g, s_sel):
        norm_sol = 1/13600
        omega_moon = 6.4177*1e-5
        
        res = []
        for j in range(22):
            res.append(np.exp(self.lnA(self.rol[j][1:], g, s_sel)))
            
        s = UnivariateSpline(self.rol[:22,0], np.asarray(res), k=2)
        
        return norm_sol * omega_moon/np.pi * s(lam) * (384400/moon_dist)**2
    
    def emit(self, frame):
        sun   = astropy.coordinates.get_sun(frame.time)
        moon  = astropy.coordinates.get_body("moon", frame.time)       
        sun_angle = moon.separation(sun)
        alpha = astropy.coordinates.Angle('180°') - sun_angle
        
        norm = self.calc_norm(frame.obswl.to(u.nm).value,
                              moon.distance.to(u.km).value,
                              alpha.rad,
                              sun_angle.rad)
        
        z_switch = 1 if moon.transform_to(frame.AltAz).alt > 0*u.deg else 0
        
        return Ray(moon.transform_to(frame.AltAz).reshape(1), 
                   weight=norm*self.SPF(frame.obswl.to(u.m).value)*z_switch, 
                   source=type(self), 
                   direction='forward')