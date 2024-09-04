import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import astropy
import astropy.units as u
from astropy.coordinates import SkyCoord, EarthLocation, AltAz

from nsb import ASSETS_PATH
from nsb.core import Frame, Model
from nsb.instrument import HESS
from nsb.emitter import moon, airglow, galactic, zodiacal, stars
from nsb.atmosphere import scattering, extinction

def CT1_Standard():
    DATA_PATH = '/lfs/l7/hess/users/gerritr/'
    # Sources:
    glow = airglow.Noll2012({"H": 87})
    zodi = zodiacal.Masana2021({})
    jons = moon.Jones2013({})
    scat = stars.GaiaDR3({'gaia_file':DATA_PATH+'gaiadr3.npy', 
                          'supp_file':DATA_PATH+"hipp_gaia_suppl.npy", 
                          'method':'synthetic'})
    smap = stars.GaiaDR3Mag15({'magnitude_maps':DATA_PATH+'gaia_mag15plus.npy'})
    gbl = galactic.Kawara2017({})
    
    # Atmospheric Extinction:
    atm_airglow = extinction.Noll2012({'scale':1.66, 'offset':-0.16})([glow])
    atm_diffuse = extinction.Masana2021({'gamma':0.75})([zodi])
    atm_stellar = extinction.Masana2021({'gamma':1})([scat,smap,gbl])
    
    # Atmospheric Scattering:
    conf_mie = {"parameters": [0.8],
                "bins": [np.linspace(0, np.pi, 1000)]}      
    conf_ray = {"parameters": [0.0148],
                "bins": [np.linspace(0, np.pi, 1000)]}
    atm_ray = scattering.Rayleigh(conf_ray)([jons]).map(np.deg2rad(180))
    atm_mie = scattering.Mie(conf_mie)([jons]).map(np.deg2rad(180))
    
    # Camera:
    CT1 = HESS.CT1(8)([atm_stellar, atm_airglow, atm_diffuse])

    return Model(CT1)