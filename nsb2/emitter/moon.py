"""Moonlight emission model.

The Moon shines by reflected sunlight, so its spectrum is the solar spectrum
scaled by the lunar albedo.  The albedo model is separated from the source
factory below so that it can be evaluated -- and tested -- without the
network access the solar reference spectrum needs.
"""

import logging

import astropy.units as u
import numpy as np
from astropy.coordinates import get_body, get_body_barycentric
from scipy.interpolate import UnivariateSpline

from nsb2.core.photometry import SolarSpectrumRieke2008
from nsb2.core.sources import EphemerisSource
from nsb2.core.spectral import SpectralGrid

from .. import ASSETS_PATH

__all__ = [
    "from_noll2013",
    "load_rolo_table",
    "lunar_distance_scaling",
    "lunar_phase_angle",
    "rolo_albedo",
    "rolo_log_albedo",
    "tabulate_lunar_spectra",
]


logger = logging.getLogger(__name__)

#: Mean solid angle of the lunar disc, in steradian, at the reference distance.
LUNAR_SOLID_ANGLE = 6.4236e-5

#: Earth-Moon distance the reference solid angle applies to, in kilometres.
LUNAR_REFERENCE_DISTANCE = 384400

#: Albedo reduction applied to the ROLO fit, as recommended by Noll et al.
ROLO_ALBEDO_SCALE = 0.87

#: Libration spread bracketing the albedo uncertainty, in degrees.
LIBRATION_SPREAD = 8

#: Number of phase angle samples tabulated between new and full Moon.
N_PHASE_SAMPLES = 50

#: The four fixed exponential and cosine parameters of the ROLO fit.
#: ``A1`` and ``A2`` are e-folding phase angles in radians; ``A3`` and ``A4``
#: set the phase and period of the cosine term.
ROLO_PHASE_PARAMETERS = (
    4.06054,
    12.8802,
    np.deg2rad(-30.5858),
    np.deg2rad(16.7498),
)


def load_rolo_table() -> np.ndarray:
    """Load the bundled ROLO albedo coefficient table.

    Returns
    -------
    numpy.ndarray
        Shape ``(25, 11)``.  Column 0 holds the band wavelength in
        nanometres; columns 1 to 10 hold that band's ROLO coefficients.

    Notes
    -----
    The table ships with ``nsb2``, so this needs no network access.
    """
    return np.genfromtxt(ASSETS_PATH / "noll2013_lunar_rolo.dat", delimiter=",")


def rolo_log_albedo(coefficients, phase_angle, sun_selenographic_lon):
    """Evaluate the ROLO log-albedo of one wavelength band.

    The empirical fit of [Kieffer2005]_ is a sum of three terms: a cubic
    polynomial in phase angle, an odd polynomial in the Sun's selenographic
    longitude that carries the libration dependence, and two exponentials
    plus a cosine that capture the opposition surge near full Moon.

    Parameters
    ----------
    coefficients : array_like
        The ten ROLO coefficients of the band, i.e. columns 1 to 10 of one
        row of :func:`load_rolo_table`.
    phase_angle : array_like
        Absolute lunar phase angle in radians, zero at full Moon.
    sun_selenographic_lon : array_like
        Selenographic longitude of the Sun in radians, near zero at full
        Moon.  The fit was derived over roughly ``[-pi/2, pi/2]``; beyond
        that the fifth-power term diverges rapidly and the result stops
        being physical.

    Returns
    -------
    numpy.ndarray
        Natural logarithm of the disc-equivalent albedo.

    See Also
    --------
    rolo_albedo : Interpolates this across bands onto a wavelength grid.
    """
    p = np.asarray(coefficients)
    g = phase_angle
    s_sel = sun_selenographic_lon
    p_1, p_2, p_3, p_4 = ROLO_PHASE_PARAMETERS

    sum_a = p[0] + p[1] * g + p[2] * g**2 + p[3] * g**3
    sum_b = p[4] * s_sel + p[5] * s_sel**3 + p[6] * s_sel**5
    sum_c = (
        p[7] * np.exp(-g / p_1)
        + p[8] * np.exp(-g / p_2)
        + p[9] * np.cos((g - p_3) / p_4)
    )

    return sum_a + sum_b + sum_c


def rolo_albedo(rolo_table, wvl: u.Quantity, phase_angle, sun_selenographic_lon):
    """Interpolate the ROLO albedo onto a wavelength grid.

    The model is tabulated in 25 discrete bands, so each band is evaluated at
    the requested geometry and the result linearly interpolated in
    wavelength.  The 13 per cent reduction of [Noll2012]_ is applied.

    Parameters
    ----------
    rolo_table : numpy.ndarray
        Coefficient table from :func:`load_rolo_table`.
    wvl : astropy.units.Quantity
        Wavelengths to evaluate at.  Must lie within the tabulated range;
        outside it the linear spline extrapolates.
    phase_angle : float
        Absolute lunar phase angle in radians.
    sun_selenographic_lon : float
        Selenographic longitude of the Sun in radians; see
        :func:`rolo_log_albedo` for its valid range.

    Returns
    -------
    numpy.ndarray
        Dimensionless disc-equivalent albedo at each wavelength.  At full
        Moon this runs from about 0.09 at 350 nm to about 0.21 at 1060 nm.
    """
    bands = [
        np.exp(rolo_log_albedo(row[1:], phase_angle, sun_selenographic_lon))
        * ROLO_ALBEDO_SCALE
        for row in rolo_table
    ]
    spline = UnivariateSpline(rolo_table[:, 0], np.asarray(bands), k=1, s=0)
    return spline(wvl.to_value(u.nm))


def lunar_distance_scaling(obstime) -> u.Quantity:
    """Scale the lunar brightness for the Sun-Moon and Moon-Earth distances.

    Both the sunlight falling on the Moon and the solid angle the Moon
    subtends vary over the orbit, so the tabulated albedo is corrected by the
    inverse square of each distance relative to its reference value.

    Parameters
    ----------
    obstime : astropy.time.Time
        Time of observation.

    Returns
    -------
    astropy.units.Quantity
        Dimensionless brightness weight, shape ``(1,)`` for a scalar time.
    """
    moon = get_body("moon", obstime)
    sun = get_body("sun", obstime)

    sun_factor = (1.0 * u.AU / sun.distance.to(u.AU)) ** 2
    obs_factor = (LUNAR_REFERENCE_DISTANCE * u.km / moon.distance.to(u.km)) ** 2

    return (
        np.atleast_1d(1 / np.pi * LUNAR_SOLID_ANGLE * obs_factor * sun_factor)
        * u.dimensionless_unscaled
    )


def lunar_phase_angle(obstime) -> np.ndarray:
    """Compute the lunar phase angle seen from Earth.

    The phase angle is the Sun-Moon-Earth angle: zero at full Moon and pi at
    new Moon.  It is computed from barycentric positions rather than from
    ecliptic longitudes so that it stays correct out of the ecliptic plane.

    Parameters
    ----------
    obstime : astropy.time.Time
        Time of observation.

    Returns
    -------
    numpy.ndarray
        Phase angle in radians, in ``[0, pi]``, shape ``(1,)`` for a scalar
        time.
    """
    pos = {
        b: get_body_barycentric(b, obstime).xyz.value for b in ("sun", "moon", "earth")
    }

    v_sun = pos["sun"] - pos["moon"]
    v_earth = pos["earth"] - pos["moon"]

    cos_g = np.dot(v_sun, v_earth) / (np.linalg.norm(v_sun) * np.linalg.norm(v_earth))

    return np.atleast_1d(np.arccos(np.clip(cos_g, -1, 1)))


def tabulate_lunar_spectra(
    rolo_table, wvl: u.Quantity, solar_spectrum: u.Quantity
) -> tuple[np.ndarray, u.Quantity]:
    """Tabulate reflected solar spectra over phase angle and libration.

    Parameters
    ----------
    rolo_table : numpy.ndarray
        Coefficient table from :func:`load_rolo_table`.
    wvl : astropy.units.Quantity
        Wavelength grid, shape ``(W,)``.
    solar_spectrum : astropy.units.Quantity
        Solar spectral flux density on that grid, shape ``(W,)``.

    Returns
    -------
    phase_angles : numpy.ndarray
        The tabulated phase angles in radians, shape
        ``(N_PHASE_SAMPLES,)``.
    spectra : astropy.units.Quantity
        Reflected photon spectra, shape ``(N_PHASE_SAMPLES, W, 3)``.  The
        trailing axis holds the three libration variants, ordered so that
        they bracket the albedo uncertainty.

    Notes
    -----
    The Sun's selenographic longitude tracks the phase angle: the sub-solar
    point is near longitude zero at full Moon and swings towards the limb as
    the Moon wanes.  The libration offsets are added on top, bracketing the
    uncertainty left by the observer-libration terms of [Kieffer2005]_ that
    this implementation omits.

    Only the absolute phase angle is available from
    :func:`lunar_phase_angle`, so the same sign of the selenographic
    longitude is used for a waxing and a waning Moon.  Because the fit is odd
    in that argument, this is an approximation worth up to about 15 per cent
    at quarter phase.
    """
    logger.debug("tabulating lunar albedo over %d phase angles", N_PHASE_SAMPLES)
    phase_angles = np.deg2rad(np.linspace(0, 180, N_PHASE_SAMPLES))
    libration = np.deg2rad([LIBRATION_SPREAD, 0, -LIBRATION_SPREAD])

    spec = np.zeros((len(phase_angles), len(wvl), len(libration)))
    for i, g in enumerate(phase_angles):
        for j, offset in enumerate(libration):
            albedo = rolo_albedo(rolo_table, wvl, g, g - offset)
            spec[i, :, j] = albedo * solar_spectrum

    spec = spec * solar_spectrum.unit
    photon_spec = (
        spec.to(
            u.photon / (u.nm * u.s * u.cm**2),
            equivalencies=u.spectral_density(wvl[np.newaxis, :, np.newaxis]),
        )
        / u.photon
    )
    return phase_angles, photon_spec


def from_noll2013() -> EphemerisSource:
    """Build the lunar source of [Noll2012]_ from the ROLO albedo model.

    The Moon shines by reflected sunlight, so its spectrum is the solar
    spectrum of [Rieke2008]_ multiplied by the lunar albedo.  The albedo is
    taken from the ROLO photometric model of [Kieffer2005]_, which
    parametrises it as a function of wavelength, phase angle and libration,
    with the 13 per cent reduction recommended by [Noll2012]_.

    Because libration is not modelled explicitly, three albedo variants
    spanning +/- :data:`LIBRATION_SPREAD` degrees of selenographic longitude
    are tabulated side by side.  They propagate through the simulation as the
    minimum, median and maximum of a
    :class:`~nsb2.core.dtypes.Prediction`, bracketing this uncertainty.

    Returns
    -------
    nsb2.core.sources.EphemerisSource
        The Moon, with brightness and phase angle evaluated from the
        ephemeris at prediction time.

    Notes
    -----
    Requires network access on first use to fetch the solar reference
    spectrum; see :func:`nsb2.core.photometry.SolarSpectrumRieke2008`.  The
    albedo model itself needs no network -- see :func:`rolo_albedo`.
    """
    rolo = load_rolo_table()
    wvl, spectrum = SolarSpectrumRieke2008()

    mask = (wvl >= rolo[0, 0] * u.nm) & (wvl <= rolo[-1, 0] * u.nm)
    wvl = wvl[mask]
    spectrum = spectrum[mask]

    phase_angles, photon_spec = tabulate_lunar_spectra(rolo, wvl, spectrum)

    return EphemerisSource(
        "moon",
        lunar_distance_scaling,
        lunar_phase_angle,
        SpectralGrid([phase_angles], wvl, photon_spec),
    )
