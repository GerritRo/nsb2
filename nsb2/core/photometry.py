import astropy.units as u
import numpy as np
import scipy.integrate as si
from astropy.constants import c, h
from astropy.io import fits
from astropy.utils.data import download_file
from dust_extinction.parameter_averages import G23

from nsb2.core.spectral import SpectralGrid

from .. import ASSETS_PATH


def create_color_grid(magnitude, color, color_range, spec_library,
                      EBVs=None, extmod=None):
    """Build a SpectralGrid mapping photometric color to spectra."""
    if EBVs is None:
        EBVs = np.linspace(0, 10, 20)
    if extmod is None:
        extmod = G23(Rv=3.1)

    def calculate_magnitude(wvl, flx, bandpass):
        z = flx * bandpass(wvl) * wvl
        return -2.5 * np.log10(si.simpson(y=z, x=wvl) * z.unit / bandpass.vegazero)

    def redden_by_dust_extinction(EBVs):
        wvl = spec_library.wvl
        flx = spec_library.flx.T
        flx = flx[:, np.newaxis, :] * extmod.extinguish(wvl, Ebv=EBVs[..., np.newaxis])
        mag_corr = calculate_magnitude(wvl, flx, magnitude)
        return wvl, 10 ** (0.4 * mag_corr[..., np.newaxis]) * flx

    wvl, flx = redden_by_dust_extinction(EBVs)
    synth_color = calculate_magnitude(wvl, flx, color[0]) - calculate_magnitude(wvl, flx, color[1])

    color_space = np.linspace(*color_range)
    ebv_interp = np.zeros((len(synth_color), len(color_space)))
    for i, color_arr in enumerate(synth_color):
        c_sort = np.argsort(color_arr)
        ebv_interp[i] = np.interp(color_space, color_arr[c_sort], EBVs[c_sort],
                                   left=np.nan, right=np.nan)

    wvl, flx = redden_by_dust_extinction(ebv_interp)
    flx = flx.T / (h * c / wvl[:, np.newaxis, np.newaxis])
    return SpectralGrid([color_space], wvl, np.transpose(flx, [1, 0, 2]))


def PicklesTRDSAtlas1998():
    """Load the Pickles 1998 TRDS stellar spectral atlas."""
    file = np.genfromtxt(ASSETS_PATH / 'pickles1998_trds_atlas.dat')
    return SpectralGrid([], file[0] * u.angstrom,
                        file[1:].T * u.erg / u.angstrom / u.s / u.cm**2)


def SolarSpectrumRieke2008():
    """Load the Rieke 2008 solar reference spectrum."""
    f_down = download_file(
        'https://archive.stsci.edu/hlsps/reference-atlases/cdbs/grid/solsys/solar_spec.fits',
        cache=True)
    hdul = fits.open(f_down)
    return (hdul[1].data['WAVELENGTH'] * u.angstrom,
            hdul[1].data['FLUX'] * u.erg / u.s / u.cm**2 / u.angstrom)
