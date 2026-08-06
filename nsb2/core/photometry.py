"""Photometric calibration and stellar spectral libraries.

Broadband catalogues give a magnitude and a colour index per star rather than
a spectrum.  This module turns that pair into a spectrum: a template library
is reddened over a range of dust column densities, synthetic colours are
computed for each reddened template, and the relation is inverted to obtain
the spectrum belonging to an observed colour.
"""

import logging

import astropy.units as u
import numpy as np
from astropy.constants import c, h
from astropy.io import fits
from astropy.utils.data import download_file
from dust_extinction.parameter_averages import G23

from nsb2.core.spectral import Bandpass, SpectralGrid, integrate_wavelength

from .. import ASSETS_PATH

__all__ = [
    "PicklesTRDSAtlas1998",
    "SolarSpectrumRieke2008",
    "create_color_grid",
    "synthetic_magnitude",
]


logger = logging.getLogger(__name__)

SOLAR_SPECTRUM_URL = (
    "https://archive.stsci.edu/hlsps/reference-atlases/cdbs/grid/solsys/solar_spec.fits"
)

#: Default grid of colour excesses E(B-V) used to redden template spectra.
DEFAULT_EBVS = np.linspace(0, 10, 20)

#: Default total-to-selective extinction ratio of the reddening law.
DEFAULT_RV = 3.1


def synthetic_magnitude(
    wvl: u.Quantity, flx: u.Quantity, bandpass: Bandpass
) -> np.ndarray:
    """Compute the synthetic Vega magnitude of a spectrum in a passband.

    Parameters
    ----------
    wvl : astropy.units.Quantity
        Wavelength grid, shape ``(W,)``.
    flx : astropy.units.Quantity
        Spectral flux density, with the wavelength axis last.
    bandpass : nsb2.core.spectral.Bandpass
        Passband to integrate over.

    Returns
    -------
    numpy.ndarray
        Magnitudes on the Vega system, with the wavelength axis removed.

    Notes
    -----
    The integrand is photon-weighted (an extra factor of wavelength), matching
    the convention of the zeropoint returned by
    :attr:`nsb2.core.spectral.Bandpass.vegazero`.  Because both integrals
    carry the wavelength unit, the result does not depend on whether ``wvl``
    is given in nanometres or angstrom.
    """
    integrand = flx * bandpass(wvl) * wvl
    ratio = integrate_wavelength(integrand, wvl) / bandpass.vegazero
    return -2.5 * np.log10(ratio.to_value(u.dimensionless_unscaled))


def create_color_grid(
    magnitude: Bandpass,
    color: list[Bandpass],
    color_range: list[float],
    spec_library: SpectralGrid,
    EBVs: np.ndarray | None = None,
    extmod=None,
) -> SpectralGrid:
    """Build a spectral grid indexed by observed photometric colour.

    Each template of ``spec_library`` is reddened over a range of colour
    excesses using the [Gordon2023]_ average extinction law, and renormalised
    to unit brightness in the ``magnitude`` band.  The resulting relation
    between colour excess and synthetic colour is inverted, so that the
    returned grid can be evaluated directly at an observed colour index.

    Parameters
    ----------
    magnitude : nsb2.core.spectral.Bandpass
        Passband the catalogue magnitudes are measured in.  The returned
        spectra are normalised to magnitude zero in this band.
    color : list of nsb2.core.spectral.Bandpass
        The two passbands whose difference defines the colour index, as
        ``[blue_band, red_band]``.
    color_range : list of float
        Lower and upper bound of the colour index to tabulate.
    spec_library : nsb2.core.spectral.SpectralGrid
        Template spectra, e.g. from :func:`PicklesTRDSAtlas1998`.
    EBVs : numpy.ndarray, optional
        Colour excesses E(B-V) to redden the templates by.  Default is
        :data:`DEFAULT_EBVS`.
    extmod : optional
        Extinction model exposing an ``extinguish(wvl, Ebv=...)`` method.
        Default is :class:`dust_extinction.parameter_averages.G23` with
        ``Rv`` = :data:`DEFAULT_RV`.

    Returns
    -------
    nsb2.core.spectral.SpectralGrid
        Grid with a single parameter axis holding the colour index, and
        spectra in photon flux density per unit magnitude-zeropoint flux.

    Notes
    -----
    Colours outside the range spanned by a given template produce ``nan``,
    which propagates to ``nan`` spectra for stars that no template can
    reproduce.  Callers are expected to handle those, e.g. via
    :func:`numpy.nansum`.
    """
    if EBVs is None:
        EBVs = DEFAULT_EBVS
    if extmod is None:
        extmod = G23(Rv=DEFAULT_RV)

    def redden(ebvs):
        """Redden every template by ``ebvs`` and renormalise its magnitude."""
        wvl = spec_library.wvl
        flx = spec_library.flx.T
        flx = flx[:, np.newaxis, :] * extmod.extinguish(wvl, Ebv=ebvs[..., np.newaxis])
        mag_corr = synthetic_magnitude(wvl, flx, magnitude)
        return wvl, 10 ** (0.4 * mag_corr[..., np.newaxis]) * flx

    logger.debug("reddening %d template spectra", spec_library.flx.shape[-1])
    wvl, flx = redden(EBVs)
    synth_color = synthetic_magnitude(wvl, flx, color[0]) - synthetic_magnitude(
        wvl, flx, color[1]
    )

    color_space = np.linspace(color_range[0], color_range[1])
    ebv_interp = np.zeros((len(synth_color), len(color_space)))
    for i, color_arr in enumerate(synth_color):
        c_sort = np.argsort(color_arr)
        ebv_interp[i] = np.interp(
            color_space, color_arr[c_sort], EBVs[c_sort], left=np.nan, right=np.nan
        )

    wvl, flx = redden(ebv_interp)
    flx = flx.T / (h * c / wvl[:, np.newaxis, np.newaxis])
    return SpectralGrid([color_space], wvl, np.transpose(flx, [1, 0, 2]))


def PicklesTRDSAtlas1998() -> SpectralGrid:
    """Load the Pickles stellar spectral atlas [Pickles1998]_.

    The atlas holds 131 flux-calibrated template spectra covering all
    spectral types and luminosity classes, and ships with ``nsb2``.

    Returns
    -------
    nsb2.core.spectral.SpectralGrid
        Grid without parameter axes; the templates occupy the component axis.
    """
    table = np.genfromtxt(ASSETS_PATH / "pickles1998_trds_atlas.dat")
    return SpectralGrid(
        [],
        table[0] * u.angstrom,
        table[1:].T * u.erg / u.angstrom / u.s / u.cm**2,
    )


def SolarSpectrumRieke2008() -> tuple[u.Quantity, u.Quantity]:
    """Load the solar reference spectrum of [Rieke2008]_.

    Used as the illuminating spectrum for reflecting bodies -- the Moon and
    the interplanetary dust responsible for zodiacal light.

    Returns
    -------
    wvl : astropy.units.Quantity
        Wavelength grid.
    flx : astropy.units.Quantity
        Solar spectral flux density at 1 au.

    Notes
    -----
    Downloaded from the STScI reference atlases on first call and cached by
    :func:`astropy.utils.data.download_file`; requires network access once.
    """
    logger.debug("downloading Rieke 2008 solar reference spectrum")
    path = download_file(SOLAR_SPECTRUM_URL, cache=True)
    with fits.open(path) as hdul:
        return (
            hdul[1].data["WAVELENGTH"] * u.angstrom,
            hdul[1].data["FLUX"] * u.erg / u.s / u.cm**2 / u.angstrom,
        )
