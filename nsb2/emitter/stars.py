import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.utils.data import download_file

from nsb2.core.photometry import PicklesTRDSAtlas1998
from nsb2.core.sources import CatalogSource, HEALPixSource
from nsb2.core.spectral import Bandpass

from .. import ASSETS_PATH


def from_gaia_dr3_catalog():
    gaia_down = download_file('https://zenodo.org/records/15396676/files/gaiadr3.npy', cache=True)
    gaia = np.load(gaia_down)

    G  = Bandpass.from_SVO('GAIA/GAIA3.G')
    BP = Bandpass.from_SVO('GAIA/GAIA3.Gbp')
    RP = Bandpass.from_SVO('GAIA/GAIA3.Grp')

    coords = SkyCoord(gaia['ra']*u.deg, gaia['dec']*u.deg, frame='icrs')
    rp = gaia['phot_rp_mean_mag']
    rp[~np.isfinite(rp)] = np.nanmin(rp)
    bp = gaia['phot_bp_mean_mag']
    bp[~np.isfinite(bp)] = np.nanmin(bp)
    rp_bp = rp - bp
    rp_bp = np.clip(rp_bp, None, 0.3)

    spec_lib = PicklesTRDSAtlas1998()

    return CatalogSource.from_photometric_catalog(coords, [G, gaia['phot_g_mean_mag']], [[RP, BP], rp_bp], spec_lib, name='GaiaDR3_G<15')

def from_gaia_dr3_map():
    gaia_down = download_file('https://zenodo.org/records/15396676/files/gaia_mag15plus.npy', cache=True)
    mag_map = np.load(gaia_down)

    G  = Bandpass.from_SVO('GAIA/GAIA3.G')
    BP = Bandpass.from_SVO('GAIA/GAIA3.Gbp')
    RP = Bandpass.from_SVO('GAIA/GAIA3.Grp')

    rp = mag_map[2]
    rp[~np.isfinite(rp)] = np.nanmin(rp)
    bp = mag_map[1]
    bp[~np.isfinite(bp)] = np.nanmin(bp)
    rp_bp = rp - bp
    rp_bp = np.clip(rp_bp, None, 0.3)

    spec_lib = PicklesTRDSAtlas1998()

    return HEALPixSource.from_photometric_map('icrs', [G, mag_map[0]], [[RP, BP], rp_bp],  spec_lib, name='GaiaDR3_G>15')

def from_gaia_suppl_catalog():
    xhip = np.genfromtxt(ASSETS_PATH / 'anderson2012_xhip_suppl.dat', skip_header=3, delimiter=',', names=True)

    V = Bandpass.from_SVO('OSN/Johnson.V')
    B = Bandpass.from_SVO('OSN/Johnson.B')

    coords = SkyCoord(xhip['RAJ2000']*u.deg, xhip['DEJ2000']*u.deg, frame='icrs')
    v_b = xhip['Vmag'] - xhip['Bmag']
    spec_lib = PicklesTRDSAtlas1998()

    return CatalogSource.from_photometric_catalog(coords, [V, xhip['Vmag']], [[V, B], v_b], spec_lib, name='XHIP_Gaia_Suppl')
