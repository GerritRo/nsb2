import astropy.units as u
import numpy as np
from astropy.coordinates import AltAz, EarthLocation, SkyCoord
from astropy.time import Time

from nsb2.core.spectral import Bandpass, SpectralGrid


def make_observation(alt=70, az=180):
    """Create a SkyOffsetFrame observation (the standard 'observation' type in nsb2).

    The returned frame has an `origin` attribute (pointing SkyCoord in AltAz)
    and pixel positions are offsets from that origin.
    """
    loc = EarthLocation(lat=-23.27 * u.deg, lon=16.5 * u.deg, height=1800 * u.m)
    t = Time('2024-06-15T22:00:00')
    altaz = AltAz(obstime=t, location=loc)
    pointing = SkyCoord(alt=alt * u.deg, az=az * u.deg, frame=altaz)
    return pointing.skyoffset_frame()


def make_spectral_grid(n_wvl=20, n_comp=3):
    """Minimal spectral grid with no parameter dimensions."""
    wvl = np.linspace(300, 700, n_wvl) * u.nm
    flx = np.ones((n_wvl, n_comp)) * u.erg / u.s / u.cm**2 / u.nm
    return SpectralGrid([], wvl, flx)


def make_bandpass(n=50, lam_min=300, lam_max=700):
    """Create a simple top-hat bandpass for testing."""
    lam = np.linspace(lam_min, lam_max, n) * u.nm
    trx = np.ones(n)
    return Bandpass(lam, trx)
