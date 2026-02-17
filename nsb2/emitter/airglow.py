import astropy.units as u
import numpy as np

from nsb2.core.sources import LonLatSource
from nsb2.core.spectral import SpectralGrid

from .. import ASSETS_PATH


def van_rhijn(height_km, zenith_angle):
    r_rh = 6738 / (6738 + height_km)
    return 1 / (1 - r_rh**2 * np.sin(zenith_angle) ** 2) ** 0.5

def from_eso_skycalc(height, sfu):
    ag_array = np.genfromtxt(ASSETS_PATH / 'eso_skycalc_airglow_130sfu.dat')
    spectral = SpectralGrid([], ag_array[:,0]*u.nm, np.atleast_2d(ag_array[:,1]).T/u.s/u.m**2/u.micron/u.arcsec**2)

    def weight_function(lon, lat):
        return van_rhijn(height.to(u.km).value, np.pi/2-lat) * (0.2 + 0.00614 * sfu) * u.dimensionless_unscaled

    def data_function(lon, lat):
        return np.empty((len(lat), 0))

    return LonLatSource(None, weight_function, data_function, spectral, name='airglow_eso_skycalc')
