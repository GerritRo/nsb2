"""common pytest fixtures."""

import astropy.units as u
import numpy as np
import pytest
from astropy.coordinates import AltAz, EarthLocation, SkyCoord
from astropy.time import Time

from nsb2.atmosphere import SingleScatteringAtmosphere
from nsb2.core.instrument import EffectiveApertureInstrument
from nsb2.core.sources import LonLatSource
from nsb2.core.spectral import Bandpass, SpectralGrid, integrate_wavelength


def stub_download(monkeypatch, module, payload):
    """Redirect a module's ``download_file`` to a local path.

    ``download_file`` is imported into each consuming module by name, so the
    patch has to target the consumer rather than ``astropy.utils.data``.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        The active monkeypatch fixture.
    module : str
        Dotted path of the module whose ``download_file`` to replace, e.g.
        ``"nsb2.emitter.stars"``.
    payload : callable or str or pathlib.Path
        A local path to return for every URL, or a callable taking the URL
        and returning one.  Raising from the callable simulates a failed
        download.
    """
    if callable(payload):
        replacement = payload
    else:

        def replacement(url, *args, **kwargs):
            return str(payload)

    monkeypatch.setattr(f"{module}.download_file", replacement)


def calibrated_bandpass(lam_min=300, lam_max=1100, n=100):
    """A top-hat bandpass with its Vega zeropoint pinned.

    ``vegazero`` is a cached property backed by a download; assigning to it
    pins a known value so photometry can be exercised offline.
    """
    lam = np.linspace(lam_min, lam_max, n) * u.nm
    bandpass = Bandpass(lam, np.ones(n))
    reference = np.ones(n) * u.erg / u.s / u.cm**2 / u.nm
    bandpass.vegazero = integrate_wavelength(lam * bandpass(lam) * reference, lam)
    return bandpass


def make_observation(alt=70, az=180):
    """Build a sky offset frame of the kind nsb2 treats as an observation.

    The returned frame carries the pointing direction on its ``origin``
    attribute, and pixel positions are offsets from that origin.

    Parameters
    ----------
    alt, az : float, optional
        Pointing altitude and azimuth in degrees.

    Returns
    -------
    astropy.coordinates.SkyOffsetFrame
        The observation frame.
    """
    location = EarthLocation(lat=-23.27 * u.deg, lon=16.5 * u.deg, height=1800 * u.m)
    altaz = AltAz(obstime=Time("2024-06-15T22:00:00"), location=location)
    return SkyCoord(alt=alt * u.deg, az=az * u.deg, frame=altaz).skyoffset_frame()


def make_spectral_grid(n_wvl=20, n_comp=3):
    """Build a flat spectral grid with no parameter axes."""
    wvl = np.linspace(300, 700, n_wvl) * u.nm
    flx = np.ones((n_wvl, n_comp)) * u.erg / u.s / u.cm**2 / u.nm
    return SpectralGrid([], wvl, flx)


def make_bandpass(n=50, lam_min=300, lam_max=700):
    """Build a top-hat bandpass with unit transmission."""
    lam = np.linspace(lam_min, lam_max, n) * u.nm
    return Bandpass(lam, np.ones(n))


def make_response(n_pix=4, grid_size=5):
    """Build an effective aperture map for a row of ``n_pix`` square pixels."""
    x_arr, y_arr, v_arr = [], [], []
    for i in range(n_pix):
        cx = np.deg2rad(i * 0.5 - 0.75)
        x_arr.append(np.linspace(cx - 0.005, cx + 0.005, grid_size))
        y_arr.append(np.linspace(-0.005, 0.005, grid_size))
        v_arr.append(np.ones((grid_size, grid_size)) * 10.0)
    return {
        "x": np.array(x_arr),
        "y": np.array(y_arr),
        "values": np.array(v_arr),
    }


@pytest.fixture
def observation():
    """A sky offset frame pointing 70 degrees above the southern horizon."""
    return make_observation()


@pytest.fixture
def bandpass():
    """A top-hat bandpass spanning 300 to 700 nm."""
    return make_bandpass()


@pytest.fixture
def spectral_grid():
    """A flat spectral grid with three components and no parameter axes."""
    return make_spectral_grid()


@pytest.fixture
def instrument(bandpass):
    """A four-pixel instrument with a flat effective aperture."""
    return EffectiveApertureInstrument(make_response(), bandpass)


@pytest.fixture
def atmosphere():
    """A single-scattering atmosphere with plain secant airmass."""
    return SingleScatteringAtmosphere(
        airmass_func=lambda z: 1 / np.cos(np.clip(z, 0, np.deg2rad(85))),
        tau_rayleigh=lambda wvl: 0.1 * (400 * u.nm / wvl) ** 4,
        tau_mie=lambda wvl: 0.05 * np.ones_like(wvl.value),
        tau_absorption=lambda wvl: 0.01 * np.ones_like(wvl.value),
        g=0.65,
    )


@pytest.fixture
def diffuse_source():
    """A uniform diffuse source with a flat spectrum."""
    wvl = np.linspace(300, 700, 20) * u.nm
    flx = np.ones((20, 1)) * 1e-12 * u.erg / u.s / u.cm**2 / u.nm

    def weight_function(lon, lat):
        return np.ones(len(lon)) * u.dimensionless_unscaled

    def data_function(lon, lat):
        return np.empty((len(lon), 0))

    return LonLatSource(
        None, weight_function, data_function, SpectralGrid([], wvl, flx)
    )
