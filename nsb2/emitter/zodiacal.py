import astropy.units as u
import numpy as np
from astropy.constants import c, h
from scipy.interpolate import RegularGridInterpolator

from nsb2.core.coordinates import SunRelativeEclipticFrame
from nsb2.core.photometry import SolarSpectrumRieke2008
from nsb2.core.sources import LonLatSource
from nsb2.core.spectral import SpectralGrid

from .. import ASSETS_PATH


def from_leinert1998():
    zod = np.genfromtxt(ASSETS_PATH / "leinert1998_zodiacal_light.dat", delimiter=",")
    A = RegularGridInterpolator(points=[np.deg2rad(zod[1:, 0]), np.deg2rad(zod[0, 1:])], values=zod[1:, 1:])

    wvl, spectrum = SolarSpectrumRieke2008()

    def color_corr(lam, elon):
        elon_low = np.where(lam < 500*u.nm, 1.2, 0.8)
        elon_high = np.where(lam < 500*u.nm, 0.9, 0.6)
        return 1 + np.vstack([elon_low, elon_high]) * np.log(lam / (500*u.nm))

    value_500nm = np.interp(0.5*u.micron, wvl, spectrum)
    spectra = spectrum*color_corr(wvl, np.linspace(30,90))/value_500nm/(h*c/wvl)

    spectral = SpectralGrid([np.deg2rad([30,90])], wvl, np.expand_dims(spectra, axis=2))

    def weight_function(lon, lat):
        return A(np.abs(np.asarray([(lon + np.pi) % (2 * np.pi) - np.pi, lat]).T)) * 1e-8  * u.W/u.m**2/u.sr/u.micron

    def data_function(lon, lat):
        return np.atleast_2d(np.clip(lon, np.pi/6, np.pi/2)).T

    return LonLatSource(SunRelativeEclipticFrame, weight_function, data_function, spectral, name='Zodiacal_Leinert1998')
