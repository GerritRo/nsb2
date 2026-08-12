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

LUNAR_SOLID_ANGLE = 6.4236e-5

LUNAR_REFERENCE_DISTANCE = 384400

ROLO_ALBEDO_SCALE = 0.87

# Bound on the Moon's libration in longitude, in degrees. This is also the
# range the ROLO libration coefficients were fitted over, so it doubles as the
# validity limit of that term -- see rolo_log_albedo.
LIBRATION_SPREAD = 8

# The phase axis is sampled per branch -- waning and waxing -- so the total is
# odd and one sample lands exactly on full Moon.
N_PHASE_SAMPLES_PER_BRANCH = 50
N_PHASE_SAMPLES = 2 * N_PHASE_SAMPLES_PER_BRANCH - 1

ECLIPTIC_OBLIQUITY = np.deg2rad(23.439281)
ECLIPTIC_POLE_ICRS = np.array(
    [0.0, -np.sin(ECLIPTIC_OBLIQUITY), np.cos(ECLIPTIC_OBLIQUITY)]
)

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
        Shape ``(25, 11)``. Column 0 holds the band wavelength in
        nanometres; columns 1 to 10 hold that band's ROLO coefficients.
    """
    return np.genfromtxt(ASSETS_PATH / "noll2013_lunar_rolo.dat", delimiter=",")


def rolo_log_albedo(coefficients, phase_angle, libration_lon):
    """Evaluate the ROLO log-albedo of one wavelength band.

    The empirical fit of [Kieffer2005]_ is a sum of three terms: a cubic
    polynomial in phase angle, an odd polynomial in the observer's
    selenographic longitude that carries the libration dependence, and two
    exponentials plus a cosine that capture the opposition surge near full
    Moon.

    The phase and libration arguments are independent geometry.  All of the
    phase dependence sits in the first and third terms; the libration term
    contributes well under a per cent across its physical range and must not
    be driven by the phase angle.

    Parameters
    ----------
    coefficients : array_like
        The ten ROLO coefficients of the band.
    phase_angle : array_like
        Absolute lunar phase angle in radians.
    libration_lon : array_like
        Selenographic longitude of the observer in radians, i.e. the Moon's
        libration in longitude.  Physically bounded by
        :data:`LIBRATION_SPREAD`, which is also the range the fit was derived
        over.  The fifth-power term turns over near 90 degrees and diverges
        beyond it, so extrapolating this argument is not meaningful.

    Returns
    -------
    numpy.ndarray
        Natural logarithm of the disc-equivalent albedo.

    Notes
    -----
    The full model of [Kieffer2005]_ carries two further terms in the
    selenographic latitude and longitude of the Sun.  Those are omitted here,
    so the fit is even in phase angle and cannot reproduce the difference
    between a waxing and a waning Moon.
    """
    p = np.asarray(coefficients)
    g = phase_angle
    phi = libration_lon
    p_1, p_2, p_3, p_4 = ROLO_PHASE_PARAMETERS

    sum_a = p[0] + p[1] * g + p[2] * g**2 + p[3] * g**3
    sum_b = p[4] * phi + p[5] * phi**3 + p[6] * phi**5
    sum_c = (
        p[7] * np.exp(-g / p_1)
        + p[8] * np.exp(-g / p_2)
        + p[9] * np.cos((g - p_3) / p_4)
    )

    return sum_a + sum_b + sum_c


def rolo_albedo(rolo_table, wvl: u.Quantity, phase_angle, libration_lon):
    """Interpolate the ROLO albedo onto a wavelength grid.

    The model is tabulated in 25 discrete bands, so each band is evaluated at
    the requested geometry and the result linearly interpolated in
    wavelength. The 13 per cent reduction of [Noll2012]_ is applied.

    Parameters
    ----------
    rolo_table : numpy.ndarray
        Coefficient table from :func:`load_rolo_table`.
    wvl : astropy.units.Quantity
        Wavelengths to evaluate at.
    phase_angle : float
        Absolute lunar phase angle in radians.
    libration_lon : float
        Libration in longitude in radians; see :func:`rolo_log_albedo` for
        its valid range.

    Returns
    -------
    numpy.ndarray
        Dimensionless disc-equivalent albedo at each wavelength.
    """
    bands = [
        np.exp(rolo_log_albedo(row[1:], phase_angle, libration_lon)) * ROLO_ALBEDO_SCALE
        for row in rolo_table
    ]
    spline = UnivariateSpline(rolo_table[:, 0], np.asarray(bands), k=1, s=0)
    return spline(wvl.to_value(u.nm))


def lunar_distance_scaling(obstime) -> u.Quantity:
    """
    Scale the lunar brightness for the Sun-Moon and Moon-Earth distances.

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
    """
    Compute the signed lunar phase angle seen from Earth.

    Parameters
    ----------
    obstime : astropy.time.Time
        Time of observation.

    Returns
    -------
    numpy.ndarray
        Phase angle in radians, in ``[-pi, pi]``, shape ``(N,)`` for ``N``
        times and ``(1,)`` for a scalar one. Positive while the Moon is
        waxing, negative when waning.
    """
    # Reshaping to (3, N) lets scalar and array times share one code path.
    pos = {
        b: get_body_barycentric(b, obstime).xyz.value.reshape(3, -1)
        for b in ("sun", "moon", "earth")
    }

    v_sun = pos["sun"] - pos["moon"]
    v_earth = pos["earth"] - pos["moon"]

    cos_g = np.einsum("ij,ij->j", v_sun, v_earth) / (
        np.linalg.norm(v_sun, axis=0) * np.linalg.norm(v_earth, axis=0)
    )
    g = np.arccos(np.clip(cos_g, -1, 1))

    # The Moon leads the Sun in ecliptic longitude by 0 to 180 deg while waxing.
    r_moon = pos["moon"] - pos["earth"]
    r_sun = pos["sun"] - pos["earth"]
    handedness = np.einsum(
        "ij,i->j", np.cross(r_sun, r_moon, axis=0), ECLIPTIC_POLE_ICRS
    )

    # Never zero: an exactly coplanar Moon is counted as waxing rather than
    # having its phase angle collapse to zero.
    return np.where(handedness < 0, -1.0, 1.0) * g


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
        The tabulated signed phase angles in radians, shape
        ``(N_PHASE_SAMPLES,)``, spanning ``[-pi, pi]``.
    spectra : astropy.units.Quantity
        Reflected photon spectra, shape ``(N_PHASE_SAMPLES, W, 3)``. The
        trailing axis holds the three libration variants, ordered so that
        they bracket the albedo uncertainty.

    Notes
    -----
    The libration variants are offsets in the observer's selenographic
    longitude alone.  They are deliberately independent of the phase angle:
    the two are separate arguments to the ROLO fit, and coupling them drives
    the libration polynomial far outside the range it was derived over, where
    its fifth-power term diverges.

    Because the fit omits the Sun's selenographic coordinates it is even in
    phase angle, so the tabulated axis is symmetric about full Moon.  It is
    still tabulated signed, over ``[-pi, pi]``, to match the convention of
    :func:`lunar_phase_angle` and to leave room for a waxing/waning
    asymmetry once those terms are added.
    """
    logger.debug("tabulating lunar albedo over %d phase angles", N_PHASE_SAMPLES)
    phase_angles = np.deg2rad(np.linspace(-180, 180, N_PHASE_SAMPLES))
    # Ascending, so the trailing axis comes out as the minimum, median and
    # maximum a Prediction expects: the fit rises with libration longitude.
    libration = np.deg2rad([-LIBRATION_SPREAD, 0, LIBRATION_SPREAD])

    spec = np.zeros((len(phase_angles), len(wvl), len(libration)))
    for i, g in enumerate(phase_angles):
        for j, offset in enumerate(libration):
            albedo = rolo_albedo(rolo_table, wvl, abs(g), offset)
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
    The selenographic terms of [Kieffer2005]_ for the Sun are not
    implemented, so the albedo is symmetric about full Moon: a waxing and a
    waning Moon at the same phase angle are predicted to be equally bright.

    Requires network access on first use to fetch the solar reference
    spectrum; see :func:`nsb2.core.photometry.SolarSpectrumRieke2008`.
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
