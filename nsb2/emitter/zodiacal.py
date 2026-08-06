"""Zodiacal light emission model."""

import logging

import astropy.units as u
import numpy as np
from astropy.constants import c, h
from scipy.interpolate import RegularGridInterpolator

from nsb2.core.coordinates import SunRelativeEclipticFrame
from nsb2.core.photometry import SolarSpectrumRieke2008
from nsb2.core.sources import LonLatSource
from nsb2.core.spectral import SpectralGrid

from .. import ASSETS_PATH

__all__ = [
    "color_correction",
    "from_leinert1998",
    "helioecliptic_longitude",
    "solar_elongation",
]

logger = logging.getLogger(__name__)

#: Unit of the tabulated Leinert brightness, in W / (m2 sr um).
LEINERT_SCALE = 1e-8

#: Wavelength the colour correction is normalised at.
REFERENCE_WAVELENGTH = 500 * u.nm

#: Elongation range, in degrees, over which the colour correction is tabulated.
ELONGATION_RANGE = (30, 90)

#: Reddening slope at the near end of the elongation range, below and above
#: :data:`REFERENCE_WAVELENGTH`.
SLOPE_NEAR = (1.2, 0.8)

#: Reddening slope at the far end of the elongation range, below and above
#: :data:`REFERENCE_WAVELENGTH`.
SLOPE_FAR = (0.9, 0.6)


def helioecliptic_longitude(lon):
    """Fold an ecliptic longitude onto the Sun-facing half of the sky.

    Longitudes come out of
    :class:`~nsb2.core.coordinates.SunRelativeEclipticFrame` normalised to
    ``[0, 2 pi)``, so they are wrapped to ``[-pi, pi)`` before the absolute
    value is taken.  Without the wrap a direction just west of the Sun would
    be read as lying almost a full turn away from it.

    Parameters
    ----------
    lon : array_like
        Ecliptic longitude relative to the Sun, in radians.

    Returns
    -------
    numpy.ndarray
        Absolute helioecliptic longitude in ``[0, pi]``.

    Examples
    --------
    >>> import numpy as np
    >>> round(float(np.degrees(helioecliptic_longitude(np.radians(350.0)))), 6)
    10.0
    """
    return np.abs((lon + np.pi) % (2 * np.pi) - np.pi)


def solar_elongation(lon, lat):
    """Angular distance from the Sun, in radians.

    This is the great-circle distance, not the difference in longitude: a
    direction on the same meridian as the Sun but sixty degrees above the
    ecliptic is sixty degrees from the Sun, not zero.

    Parameters
    ----------
    lon : array_like
        Ecliptic longitude relative to the Sun, in radians.
    lat : array_like
        Ecliptic latitude, in radians.

    Returns
    -------
    numpy.ndarray
        Solar elongation in ``[0, pi]``.

    Examples
    --------
    >>> import numpy as np
    >>> round(float(np.degrees(solar_elongation(0.0, np.radians(60.0)))), 6)
    60.0
    """
    cos_eps = np.cos(helioecliptic_longitude(lon)) * np.cos(lat)
    return np.arccos(np.clip(cos_eps, -1.0, 1.0))


@u.quantity_input(wvl=u.nm)
def color_correction(wvl: u.Quantity) -> np.ndarray:
    """Reddening of zodiacal light relative to the solar spectrum.

    Zodiacal light is sunlight scattered off interplanetary dust, which
    reddens it, and the more so the closer to the Sun one looks.
    [Leinert1998]_ describes this as a correction factor that is unity at
    :data:`REFERENCE_WAVELENGTH` and varies logarithmically with wavelength,
    with a slope that differs either side of that wavelength and between the
    two ends of :data:`ELONGATION_RANGE`.

    Parameters
    ----------
    wvl : astropy.units.Quantity
        Wavelength grid, shape ``(W,)``.

    Returns
    -------
    numpy.ndarray
        Correction factor, shape ``(2, W)``: one curve at each end of
        :data:`ELONGATION_RANGE`, to be interpolated between.

    Notes
    -----
    The logarithm is base 10.  With these slopes a natural logarithm would
    drive the correction negative below about 220 nm, which is unphysical,
    and would overstate the reddening by a factor of ``ln(10)`` throughout.

    Examples
    --------
    >>> import astropy.units as u
    >>> near, far = color_correction([300, 500, 700] * u.nm)
    >>> [round(float(x), 3) for x in near]
    [0.734, 1.0, 1.117]
    """
    wvl = u.Quantity(wvl)
    is_blue = wvl < REFERENCE_WAVELENGTH
    slope = np.vstack(
        [
            np.where(is_blue, SLOPE_NEAR[0], SLOPE_NEAR[1]),
            np.where(is_blue, SLOPE_FAR[0], SLOPE_FAR[1]),
        ]
    )
    ratio = (wvl / REFERENCE_WAVELENGTH).to_value(u.dimensionless_unscaled)
    return 1 + slope * np.log10(ratio)


def from_leinert1998() -> LonLatSource:
    """Build the zodiacal light source of [Leinert1998]_.

    Zodiacal light is fixed relative to the Sun rather than to the stars, so
    its brightness is tabulated in
    :class:`~nsb2.core.coordinates.SunRelativeEclipticFrame`.  Its spectrum
    is the solar spectrum of [Rieke2008]_ reddened by
    :func:`color_correction`, interpolated by solar elongation.

    Returns
    -------
    nsb2.core.sources.LonLatSource
        Zodiacal light.

    Notes
    -----
    Requires network access on first use to fetch the solar reference
    spectrum; see :func:`nsb2.core.photometry.SolarSpectrumRieke2008`.
    """
    zod = np.genfromtxt(ASSETS_PATH / "leinert1998_zodiacal_light.dat", delimiter=",")
    brightness = RegularGridInterpolator(
        points=[np.deg2rad(zod[1:, 0]), np.deg2rad(zod[0, 1:])], values=zod[1:, 1:]
    )

    wvl, spectrum = SolarSpectrumRieke2008()

    logger.debug("building zodiacal light spectral grid")
    reference_flux = np.interp(REFERENCE_WAVELENGTH, wvl, spectrum)
    spectra = spectrum * color_correction(wvl) / reference_flux / (h * c / wvl)

    spectral = SpectralGrid(
        [np.deg2rad(ELONGATION_RANGE)], wvl, np.expand_dims(spectra, axis=2)
    )

    def weight_function(lon, lat):
        """Interpolate the tabulated brightness at the given sky position.

        The Leinert table is indexed by helioecliptic longitude and ecliptic
        latitude, and is symmetric in both, so the folded absolute values are
        used directly rather than the elongation.
        """
        coords = np.asarray([helioecliptic_longitude(lon), np.abs(lat)]).T
        return brightness(coords) * LEINERT_SCALE * u.W / u.m**2 / u.sr / u.micron

    def data_function(lon, lat):
        """Return the solar elongation, clipped to the tabulated range."""
        clipped = np.clip(solar_elongation(lon, lat), *np.deg2rad(ELONGATION_RANGE))
        return np.atleast_2d(clipped).T

    return LonLatSource(
        SunRelativeEclipticFrame,
        weight_function,
        data_function,
        spectral,
        name="Zodiacal_Leinert1998",
    )
