import numpy as np
import astropy.units as u
import astropy.constants as c

def henyey_greenstein(g, theta):
    """
    Henyey & Greenstein (1941) formula for mie scattering based on parameter g and scattering angle theta

    Parameters
    ----------
    g : floatlike
        _description_
    theta : floatlike
        Angle in radians

    Returns
    -------
    _type_
        _description_
    """
    gsq = g**2
    return 1/(4*np.pi) * (1-gsq) / (1+gsq-2*g*np.cos(theta))**1.5

def bucholtz(chi, theta):
    """
    Formula for rayleigh scattering based on Bucholtz (1995)

    Parameters
    ----------
    chi : _type_
        _description_
    theta : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """
    return 3/(16*np.pi) * (1-chi)/(1+2*chi) * ((1+3*chi)/(1-chi) + np.cos(theta)**2)

def k_rayleigh(frame):
    """
    Rayleight extinction coefficient based on height of observatory and observation wavelength

    Parameters
    ----------
    frame : Frame
        Observation Frame

    Returns
    -------
    numpy.array
        Array of extinction values for each wavelength in frame.obswl
    """
    lam, height = frame.obswl, frame.location.height
    return 0.00879*(lam.to(u.micron).value)**-4.09 * np.exp(-height.to(u.km).value/8)
                    
def k_mie(frame):
    """
    Mie extinction coefficient based on height of observatory, observation wavelength and
    aeronet values for AOD at 380 nm and angstrom coefficient.

    Parameters
    ----------
    frame : Frame
        Observation Frame

    Returns
    -------
    numpy.array
        Array of extinction values for each wavelength in frame.obswl
    """
    lam, height, aero = frame.obswl, frame.location.height, frame.conf['aero']
    return aero[0]*(lam.to(u.nm).value/380)**(-aero[1]) * np.exp(-height.to(u.km).value/1.54)

            
    