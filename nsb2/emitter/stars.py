"""Starlight emission models."""

import logging

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.utils.data import download_file

from nsb2.core.photometry import PicklesTRDSAtlas1998
from nsb2.core.sources import CatalogSource, HEALPixSource
from nsb2.core.spectral import Bandpass

from .. import ASSETS_PATH

__all__ = [
    "from_gaia_dr3_catalog",
    "from_gaia_dr3_map",
    "from_gaia_suppl_catalog",
]


logger = logging.getLogger(__name__)

GAIA_CATALOG_URL = "https://zenodo.org/records/15396676/files/gaiadr3.npy"
GAIA_MAP_URL = "https://zenodo.org/records/15396676/files/gaia_mag15plus.npy"

#: Magnitude splitting the resolved catalogue from the integrated map.
GAIA_SPLIT_MAGNITUDE = 15

#: Upper bound on the RP-BP colour index, beyond which no template matches.
MAX_RP_BP = 0.3


def _gaia_bandpasses() -> tuple[Bandpass, Bandpass, Bandpass]:
    """Fetch the three Gaia DR3 passbands from the SVO Filter Profile Service."""
    return (
        Bandpass.from_SVO("GAIA/GAIA3.G"),
        Bandpass.from_SVO("GAIA/GAIA3.Gbp"),
        Bandpass.from_SVO("GAIA/GAIA3.Grp"),
    )


def _rp_bp_color(rp: np.ndarray, bp: np.ndarray) -> np.ndarray:
    """Compute the RP-BP colour index, filling gaps and clipping the red end.

    Sources without a valid magnitude in one band are assigned the brightest
    magnitude present, and the index is clipped at :data:`MAX_RP_BP` because
    the template library holds no spectrum redder than that.

    Parameters
    ----------
    rp, bp : numpy.ndarray
        Red and blue photometer magnitudes.  Not modified.

    Returns
    -------
    numpy.ndarray
        The colour index.
    """
    rp = np.where(np.isfinite(rp), rp, np.nanmin(rp))
    bp = np.where(np.isfinite(bp), bp, np.nanmin(bp))
    return np.clip(rp - bp, None, MAX_RP_BP)


def from_gaia_dr3_catalog() -> CatalogSource:
    """Build a point source catalogue from the bright half of Gaia DR3.

    Stars brighter than G = :data:`GAIA_SPLIT_MAGNITUDE` are resolved
    individually, since a single bright star in the field of view raises the
    background of the pixels it falls into far above their neighbours.  Their
    spectra are inferred from the RP-BP colour index against the [Pickles1998]_
    template library.

    Returns
    -------
    nsb2.core.sources.CatalogSource
        The resolved stars.  Call
        :meth:`~nsb2.core.sources.CatalogSource.build_balltree` before
        querying it.

    See Also
    --------
    from_gaia_dr3_map : The faint half of the same catalogue.

    Notes
    -----
    Requires network access on first use to fetch the catalogue from Zenodo
    [Roellinghoff2025]_ and the passbands from the SVO Filter Profile
    Service.
    """
    logger.debug("downloading Gaia DR3 bright star catalogue")
    gaia = np.load(download_file(GAIA_CATALOG_URL, cache=True))

    g_band, bp_band, rp_band = _gaia_bandpasses()
    coords = SkyCoord(gaia["ra"] * u.deg, gaia["dec"] * u.deg, frame="icrs")
    color = _rp_bp_color(gaia["phot_rp_mean_mag"], gaia["phot_bp_mean_mag"])

    return CatalogSource.from_photometric_catalog(
        coords,
        [g_band, gaia["phot_g_mean_mag"]],
        [[rp_band, bp_band], color],
        PicklesTRDSAtlas1998(),
        name=f"GaiaDR3_G<{GAIA_SPLIT_MAGNITUDE}",
    )


def from_gaia_dr3_map() -> HEALPixSource:
    """Build a diffuse map from the faint half of Gaia DR3.

    Stars fainter than G = :data:`GAIA_SPLIT_MAGNITUDE` are too numerous to
    trace individually and too dim to resolve, so they are pre-binned into a
    HEALPix map of integrated magnitude and mean colour and treated as
    diffuse emission.

    Returns
    -------
    nsb2.core.sources.HEALPixSource
        The unresolved stellar background.

    See Also
    --------
    from_gaia_dr3_catalog : The bright half of the same catalogue.

    Notes
    -----
    Requires network access on first use to fetch the map from Zenodo
    [Roellinghoff2025]_ and the passbands from the SVO Filter Profile
    Service.
    """
    logger.debug("downloading Gaia DR3 faint star map")
    mag_map = np.load(download_file(GAIA_MAP_URL, cache=True))

    g_band, bp_band, rp_band = _gaia_bandpasses()
    color = _rp_bp_color(mag_map[2], mag_map[1])

    return HEALPixSource.from_photometric_map(
        "icrs",
        [g_band, mag_map[0]],
        [[rp_band, bp_band], color],
        PicklesTRDSAtlas1998(),
        name=f"GaiaDR3_G>{GAIA_SPLIT_MAGNITUDE}",
    )


def from_gaia_suppl_catalog() -> CatalogSource:
    """Build a catalogue of the stars Gaia is too bright to measure.

    The very brightest stars saturate Gaia's detectors and are missing or
    unreliable in DR3, yet they are exactly the ones that matter most for the
    night sky background.  They are supplied instead from the extended
    Hipparcos compilation [Anderson2012]_, which ships with ``nsb2``, using
    Johnson V-B colours.

    Returns
    -------
    nsb2.core.sources.CatalogSource
        The supplementary bright stars.  Call
        :meth:`~nsb2.core.sources.CatalogSource.build_balltree` before
        querying it.

    Notes
    -----
    Requires network access on first use to fetch the passbands from the SVO
    Filter Profile Service.
    """
    xhip = np.genfromtxt(
        ASSETS_PATH / "anderson2012_xhip_suppl.dat",
        skip_header=3,
        delimiter=",",
        names=True,
    )

    v_band = Bandpass.from_SVO("OSN/Johnson.V")
    b_band = Bandpass.from_SVO("OSN/Johnson.B")

    coords = SkyCoord(xhip["RAJ2000"] * u.deg, xhip["DEJ2000"] * u.deg, frame="icrs")
    color = xhip["Vmag"] - xhip["Bmag"]

    return CatalogSource.from_photometric_catalog(
        coords,
        [v_band, xhip["Vmag"]],
        [[v_band, b_band], color],
        PicklesTRDSAtlas1998(),
        name="XHIP_Gaia_Suppl",
    )
