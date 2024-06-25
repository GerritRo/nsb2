import numpy as np
import astropy
from astropy.coordinates import SkyCoord, EarthLocation, AltAz
import astropy.units as u
import histlite as hl

from nsb.core import Frame, Model

from nsb.core.instrument import Camera, Instrument
from nsb.atmosphere import scattering, extinction

from nsb.emitter import moon, stars, airglow, galactic, zodiacal

from ctapipe.instrument import CameraGeometry
import nsb.utils.bandpass as bandpass

def CT1_Standard():
    cachepath = '/home/gerritr/ECAP/nsb_simulation/data/'

    # Moonlight:
    conf_jons = {}

    # Diffuse Emitters:
    conf_glow = {"H": 87}
    conf_gbl  = {"Mag_0": 236.8*2.3504*1e-11}
    conf_zodi = {}

    # Starlight
    conf_gaia = {"catalog_file": '/lfs/l7/hess/users/gerritr/GaiaDR3Tycho.pkl',
                 "magmin" : -2,
                 "magmax": 15}

    # Atmospheric
    conf_mie = {"parameters": [0.8],
                "bins": [np.linspace(0, np.pi, 1000)]}      
    conf_ray = {"parameters": [0.0148],
                "bins": [np.linspace(0, np.pi, 1000)]}

    c_bins = [np.linspace(0, np.deg2rad(3.5), 50), np.linspace(0, 2*np.pi, 20), np.linspace(0, 2*1e-3, 100)]

    def psf(off, pos, rho):
        p = [0.12, 0.5]
        off = off/np.deg2rad(2.5)
        
        sigma = np.sqrt(p[0]**2 + p[1]*off**2)*1e-3
        
        return np.sin(rho)/(2*np.pi*sigma**2) * np.exp(-rho**2/(2*sigma**2))
    
    h_psf = hl.hist_from_eval(psf, bins=c_bins)
    cam = CameraGeometry.from_name('HESS-I')
    hess1u = Camera.from_ctapipe(cam, h_psf, 94.4, 15.28)
    
    # Load all emitters
    # Sources:
    glow = airglow.Noll2012(conf_glow)
    zodi = zodiacal.Masana2021(conf_zodi)

    gaia = stars.StarCatalog(conf_gaia)
    gbl  = galactic.GaiaDR3Mag15(conf_gbl)

    jons = moon.Jones2013(conf_jons)

    # Atmospheric Extinction:
    atm_airglow = extinction.Noll2012({'scale':1., 'offset':0.4})([glow])
    atm_diffuse = extinction.Masana2021({'gamma':0.5})([zodi])
    atm_stellar = extinction.Masana2021({'gamma':1})([gaia,gbl])

    # Atmospheric Scattering:
    atm_ray = scattering.Rayleigh(conf_ray)([jons]).map(np.deg2rad(180))
    atm_mie = scattering.Mie(conf_mie)([jons]).map(np.deg2rad(180))

    # Camera:
    CT1 = Instrument({'camera':hess1u, 'bandpass':bandpass.HESS1U()}, 32)([atm_airglow, atm_diffuse, atm_stellar, atm_ray, atm_mie])
    
    return Model(CT1)