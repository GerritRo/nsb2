import numpy as np
import astropy.units as u
import astropy.constants as c

def henyey_greenstein(g, theta):
    gsq = g**2
    return 1/(4*np.pi) * (1-gsq) / (1+gsq-2*g*np.cos(theta))**1.5

def bucholtz(chi, theta):
    return 3/(16*np.pi) * (1-chi)/(1+2*chi) * ((1+3*chi)/(1-chi) + np.cos(theta)**2)

def ballesteros(diff, T, a, b, c):
    return T * (1/(a*diff+b) + 1/(a*diff + c))

def blackbody(lam, T):
    f1 = 1e-9*2*c.h.value*c.c.value**2
    f2 = c.h.value*c.c.value/c.k_B.value
    return (f1/lam**5 * 1/(np.exp(f2/(T*lam))-1))

def k_rayleigh(frame):
    lam, height = frame.obswl, frame.location.height
    return 0.00879*(lam.to(u.micron).value)**-4.09 * np.exp(-height.to(u.km).value/8)
                    
def k_mie(frame):
    lam, height, aero = frame.obswl, frame.location.height, frame.conf['aero']
    return aero[0]*(lam.to(u.nm).value/380)**(-aero[1]) * np.exp(-height.to(u.km).value/1.54)

            
    