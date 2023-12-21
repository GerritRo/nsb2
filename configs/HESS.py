import numpy as np
import astropy
from astropy.coordinates import SkyCoord, EarthLocation, AltAz
import astropy.units as u

from nsb.core import Frame, Model

from nsb.instrument import HESS
from nsb.atmosphere import scattering, extinction

from nsb.emitter import moon, stars, airglow, galactic, zodiacal

def CT1_Standard():
    cachepath = '/home/gerritr/ECAP/nsb_simulation/data/'

    # Moonlight:
    conf_jons = {}

    # Diffuse Emitters:
    conf_glow = {"H": 87}
    conf_gbl  = {"Mag_0": 562.53*2.3504*1e-11}
    conf_zodi = {}

    # Starlight
    conf_hipp = {"cache_path": cachepath,
                 "magmin": -1.5,
                 "magmax": 8,
                 "Mag_0": 143.18*2.3504*1e-11,
                 "cache": True}

    conf_gaia = {"cache_path": cachepath,
                 "magmin" : 8,
                 "magmax": 15,
                 "Mag_0": 562.53*2.3504*1e-11,
                 "cache": True}

    # Atmospheric
    conf_mie = {"parameters": [0.8],
                "bins": [np.linspace(0, np.pi, 1000)]}      
    conf_ray = {"parameters": [0.0148],
                "bins": [np.linspace(0, np.pi, 1000)]}
    conf_masana = {'gamma':1}
    conf_noll = {'scale':1., 'offset':0.4}

    # Instrument
    c_bins = [np.linspace(0, np.deg2rad(3), 50), np.linspace(0, 2*np.pi, 20), np.linspace(0, np.deg2rad(0.5), 100)]
    conf_cornils = {"parameters": [0.35, 3], 'bins': c_bins, 'degradation':0.75}
    
    # Load all emitters
    # Sources:
    glow = airglow.Noll2012(conf_glow)
    zodi = zodiacal.Masana2021(conf_zodi)

    hipp = stars.Hipparcos(conf_hipp)
    gaia = stars.GaiaDR3(conf_gaia)
    gbl  = galactic.GaiaDR3Mag15(conf_gbl)

    jons = moon.Jones2013(conf_jons)

    # Atmospheric Extinction:
    atm_airglow = extinction.Masana2021(conf_masana)([glow])
    atm_diffuse = extinction.Masana2021(conf_masana)([zodi, gbl])
    atm_stellar = extinction.Masana2021(conf_masana)([gaia, hipp])

    # Atmospheric Scattering:
    atm_ray = scattering.Rayleigh(conf_ray)([jons]).map(np.deg2rad(180))
    atm_mie = scattering.Mie(conf_mie)([jons]).map(np.deg2rad(180))

    # Instrument:
    cornils_df = HESS.Cornils2003(conf_cornils, 1)([atm_ray, atm_mie, atm_diffuse, atm_airglow])
    cornils_st = HESS.Cornils2003(conf_cornils, 50)([atm_stellar])

    # Camera:
    CT1 = HESS.CT1(3)([cornils_df, cornils_st])
    
    return Model(CT1)