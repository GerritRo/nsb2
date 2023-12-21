import numpy as np
import histlite as hl
import scipy
import scipy.integrate as si
import astropy.units as u

from nsb.core.atmosphere import RadialScattering
from nsb.utils.formulas import henyey_greenstein, bucholtz, k_rayleigh, k_mie

class KS1991(RadialScattering):
    def X(self, alt):
        return (1 - 0.96 * (np.sin(np.pi/2-alt))**2)**(-0.5)

    def indicatrix(self, rho):
        A, B = self.config['parameters']
        D = 180/np.pi/40
        norm = 1/(10**B*0.434294*(1-10**(np.pi*(-D)))/D + 4.90088*10**A)
        return norm*(10**A * (1.06 + (np.cos(rho))**2) + 10**(B - np.rad2deg(rho)/40))

    def gradation(self, frame, f_rays, r_rays):
        k = frame.conf['k']
        return 10 ** (-0.4 * k * self.X(f_rays.coords.alt.rad)) * (1-10**(-0.4*k*self.X(r_rays.coords.alt.rad)))
        
class Rayleigh(RadialScattering):
    def X(self, z):
        return (1 - 0.96 * (np.sin(z))**2)**(-0.5)
    
    def gradation(self, tau, scale, z, Z):
        sec_Z = self.X(Z)[:,np.newaxis]
        sec_z = self.X(z)[:,np.newaxis]
        return sec_Z * scale * (np.exp(-sec_Z*tau) - np.exp(-sec_z*tau)) / (sec_z - sec_Z)
    
    def indicatrix(self, rho):
        p = self.config['parameters']
        return bucholtz(p[0], rho)

    def t_args(self, frame, f_rays, b_rays):
        tau   = (k_rayleigh(frame) + k_mie(frame)) / frame.conf['albedo']
        scale = k_rayleigh(frame) / tau * (1 + 2.2*tau)
        
        z = np.pi/2-f_rays.coords.alt.rad
        Z = np.pi/2-b_rays.coords.alt.rad
        
        return tau, scale, z, Z
    
class Mie(RadialScattering):
    def X(self, z):
        return (1 - 0.96 * (np.sin(z))**2)**(-0.5)
    
    def gradation(self, tau, scale, z, Z):
        sec_Z = self.X(Z)[:,np.newaxis]
        sec_z = self.X(z)[:,np.newaxis]
        return sec_Z * scale * (np.exp(-sec_Z*tau) - np.exp(-sec_z*tau)) / (sec_z - sec_Z)
    
    def indicatrix(self, rho):
        p = self.config['parameters']
        return henyey_greenstein(p[0], rho)
    
    def t_args(self, frame, f_rays, b_rays):
        tau   = (k_rayleigh(frame) + k_mie(frame)) / frame.conf['albedo']
        scale = k_mie(frame) / tau * (1 + 2.2*tau)
        
        z = np.pi/2-f_rays.coords.alt.rad
        Z = np.pi/2-b_rays.coords.alt.rad
        
        return tau, scale, z, Z