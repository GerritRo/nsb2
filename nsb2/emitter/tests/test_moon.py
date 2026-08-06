"""Tests for the offline parts of :mod:`nsb2.emitter.moon`.

Everything except :func:`~nsb2.emitter.moon.from_noll2013` itself runs
without network access: the ROLO coefficient table ships with the package,
and the ephemeris comes from astropy's built-in one.  Only the solar
reference spectrum is downloaded, so only that is exercised in
``test_remote_emitters.py``.
"""

import astropy.units as u
import numpy as np
import pytest
from astropy.time import Time

from nsb2.emitter.moon import (
    LIBRATION_SPREAD,
    LUNAR_REFERENCE_DISTANCE,
    N_PHASE_SAMPLES,
    ROLO_ALBEDO_SCALE,
    load_rolo_table,
    lunar_distance_scaling,
    lunar_phase_angle,
    rolo_albedo,
    rolo_log_albedo,
    tabulate_lunar_spectra,
)

FULL_MOON = Time("2024-01-25T17:54:00")
NEW_MOON = Time("2024-01-11T11:57:00")


@pytest.fixture(scope="module")
def rolo_table():
    return load_rolo_table()


class TestLoadRoloTable:
    def test_shape(self, rolo_table):
        """25 bands, one wavelength column plus ten coefficients."""
        assert rolo_table.shape == (25, 11)

    def test_wavelengths_are_monotonic_and_optical(self, rolo_table):
        wavelengths = rolo_table[:, 0]
        assert np.all(np.diff(wavelengths) > 0)
        assert wavelengths[0] == pytest.approx(350.0)
        assert wavelengths[-1] == pytest.approx(1059.5)

    def test_is_finite(self, rolo_table):
        assert np.all(np.isfinite(rolo_table))


class TestRoloLogAlbedo:
    def test_returns_a_scalar_for_scalar_geometry(self, rolo_table):
        value = rolo_log_albedo(rolo_table[0][1:], 0.5, np.pi - 0.5)
        assert np.isscalar(value) or value.shape == ()

    def test_albedo_is_physical(self, rolo_table):
        """exp(log_albedo) must land in a plausible albedo range."""
        for row in rolo_table:
            albedo = np.exp(rolo_log_albedo(row[1:], 0.0, np.pi))
            assert 0 < albedo < 1

    def test_brightest_at_full_moon(self, rolo_table):
        """Phase angle zero is full Moon; the albedo must fall away from it."""
        full = rolo_log_albedo(rolo_table[10][1:], 0.0, np.pi)
        quarter = rolo_log_albedo(rolo_table[10][1:], np.pi / 2, np.pi / 2)
        assert full > quarter

    def test_decreases_with_phase_angle_beyond_the_opposition_surge(self, rolo_table):
        """Past the surge near full Moon, brightness falls monotonically.

        Inside roughly the first ten degrees the cosine term of the fit makes
        the curve non-monotonic, which is the opposition surge itself.
        """
        phases = np.linspace(np.deg2rad(20), np.pi, 20)
        values = np.array([rolo_log_albedo(rolo_table[10][1:], g, 0.0) for g in phases])
        assert np.all(np.diff(values) < 0)

    def test_broadcasts_over_phase_angle(self, rolo_table):
        phases = np.linspace(0, 1, 7)
        values = rolo_log_albedo(rolo_table[0][1:], phases, np.pi - phases)
        assert values.shape == (7,)

    def test_libration_changes_the_result(self, rolo_table):
        """The selenographic-longitude terms must actually contribute."""
        centred = rolo_log_albedo(rolo_table[5][1:], 0.5, np.pi - 0.5)
        librated = rolo_log_albedo(
            rolo_table[5][1:], 0.5, np.pi - 0.5 + np.deg2rad(LIBRATION_SPREAD)
        )
        assert centred != pytest.approx(librated)

    def test_accepts_a_plain_list_of_coefficients(self, rolo_table):
        from_array = rolo_log_albedo(rolo_table[0][1:], 0.3, 2.0)
        from_list = rolo_log_albedo(list(rolo_table[0][1:]), 0.3, 2.0)
        assert from_list == pytest.approx(from_array)


class TestRoloAlbedo:
    def test_shape_follows_the_wavelength_grid(self, rolo_table):
        wvl = np.linspace(400, 1000, 13) * u.nm
        assert rolo_albedo(rolo_table, wvl, 0.0, np.pi).shape == (13,)

    def test_values_are_physical(self, rolo_table):
        wvl = np.linspace(400, 1000, 50) * u.nm
        albedo = rolo_albedo(rolo_table, wvl, 0.0, np.pi)
        assert np.all((albedo > 0) & (albedo < 1))

    def test_reproduces_a_tabulated_band_at_its_own_wavelength(self, rolo_table):
        """A linear spline through the bands is exact at the band centres."""
        band = rolo_table[7]
        expected = (
            np.exp(rolo_log_albedo(band[1:], 0.4, np.pi - 0.4)) * ROLO_ALBEDO_SCALE
        )
        interpolated = rolo_albedo(rolo_table, [band[0]] * u.nm, 0.4, np.pi - 0.4)
        assert float(interpolated[0]) == pytest.approx(expected)

    def test_applies_the_noll_reduction(self, rolo_table):
        """Noll et al. recommend a 13 per cent reduction of the ROLO fit."""
        band = rolo_table[7]
        raw = np.exp(rolo_log_albedo(band[1:], 0.0, np.pi))
        reduced = rolo_albedo(rolo_table, [band[0]] * u.nm, 0.0, np.pi)
        assert float(reduced[0]) / raw == pytest.approx(ROLO_ALBEDO_SCALE)

    def test_moon_is_red_within_the_physical_libration_range(self, rolo_table):
        """Lunar regolith reflects far more strongly in the red than the blue.

        Evaluated with the libration term in its real range (a few degrees),
        the fit rises smoothly from about 0.09 at 350 nm to about 0.21 at
        1060 nm, which is the published disc-equivalent lunar albedo.
        """
        wvl = np.linspace(350, 1059.5, 20) * u.nm
        for libration in np.deg2rad([-LIBRATION_SPREAD, 0, LIBRATION_SPREAD]):
            albedo = rolo_albedo(rolo_table, wvl, 0.0, libration)
            # The band-to-band fit wobbles slightly, so compare the trend
            # rather than every consecutive step.
            assert albedo[-1] > 2 * albedo[0], "albedo must roughly double"
            assert albedo[-5:].mean() > albedo[:5].mean()
            assert float(albedo[0]) == pytest.approx(0.086, abs=0.01)
            assert float(albedo[-1]) == pytest.approx(0.212, abs=0.01)

    def test_out_of_range_longitude_is_unphysical(self, rolo_table):
        """Guard on the argument range documented for the fit.

        The fit is odd in the selenographic longitude with a fifth-power
        term, so feeding it a value near pi -- as an earlier revision did by
        passing ``pi - phase_angle`` -- makes that term dominate and destroys
        the wavelength dependence.  Kept so the mistake cannot come back
        unnoticed.
        """
        wvl = np.linspace(350, 1059.5, 20) * u.nm
        sensible = rolo_albedo(rolo_table, wvl, 0.0, 0.0)
        out_of_range = rolo_albedo(rolo_table, wvl, 0.0, np.pi)
        assert sensible[-1] > 2 * sensible[0]
        assert out_of_range[-1] < out_of_range[0]

    def test_accepts_any_wavelength_unit(self, rolo_table):
        in_nm = rolo_albedo(rolo_table, [500] * u.nm, 0.2, 2.0)
        in_angstrom = rolo_albedo(rolo_table, [5000] * u.angstrom, 0.2, 2.0)
        assert float(in_angstrom[0]) == pytest.approx(float(in_nm[0]))


class TestLunarPhaseAngle:
    def test_shape_and_range(self):
        angle = lunar_phase_angle(FULL_MOON)
        assert angle.shape == (1,)
        assert 0 <= angle[0] <= np.pi

    def test_small_near_full_moon(self):
        assert np.rad2deg(lunar_phase_angle(FULL_MOON)[0]) < 15

    def test_large_near_new_moon(self):
        assert np.rad2deg(lunar_phase_angle(NEW_MOON)[0]) > 165

    def test_full_moon_is_closer_to_zero_than_new_moon(self):
        assert lunar_phase_angle(FULL_MOON)[0] < lunar_phase_angle(NEW_MOON)[0]


class TestLunarDistanceScaling:
    def test_shape_and_unit(self):
        weight = lunar_distance_scaling(FULL_MOON)
        assert weight.shape == (1,)
        assert weight.unit == u.dimensionless_unscaled

    def test_is_close_to_the_reference_solid_angle(self):
        """At the mean distance the weight is the disc solid angle over pi."""
        weight = float(lunar_distance_scaling(FULL_MOON)[0])
        assert weight == pytest.approx(6.4236e-5 / np.pi, rel=0.15)

    def test_brighter_at_perigee_than_apogee(self):
        """The inverse-square correction must track the Earth-Moon distance."""
        from astropy.coordinates import get_body

        times = Time("2024-01-01") + np.arange(0, 28, 0.5) * u.day
        distances = get_body("moon", times).distance.to_value(u.km)
        perigee = times[np.argmin(distances)]
        apogee = times[np.argmax(distances)]
        assert lunar_distance_scaling(perigee)[0] > lunar_distance_scaling(apogee)[0]

    def test_reference_distance_is_the_mean_orbit(self):
        assert 350000 < LUNAR_REFERENCE_DISTANCE < 410000


@pytest.fixture(scope="module")
def tabulated(rolo_table):
    """Reflected spectra for a flat unit solar spectrum."""
    wvl = np.linspace(400, 1000, 25) * u.nm
    solar = np.ones(25) * u.erg / u.s / u.cm**2 / u.nm
    return tabulate_lunar_spectra(rolo_table, wvl, solar)


class TestTabulateLunarSpectra:
    def test_shapes(self, tabulated):
        phase_angles, spectra = tabulated
        assert phase_angles.shape == (N_PHASE_SAMPLES,)
        assert spectra.shape == (N_PHASE_SAMPLES, 25, 3)

    def test_phase_angles_span_new_to_full(self, tabulated):
        phase_angles, _ = tabulated
        assert phase_angles[0] == pytest.approx(0.0)
        assert phase_angles[-1] == pytest.approx(np.pi)

    def test_spectra_are_photon_flux_density(self, tabulated):
        _, spectra = tabulated
        assert spectra.unit.is_equivalent(1 / (u.nm * u.s * u.cm**2))

    def test_spectra_are_positive_and_finite(self, tabulated):
        _, spectra = tabulated
        assert np.all(spectra.value > 0)
        assert np.all(np.isfinite(spectra.value))

    def test_brightest_at_full_moon(self, tabulated):
        """Phase angle zero is full Moon, the first tabulated entry."""
        _, spectra = tabulated
        assert np.median(spectra[0].value) > np.median(spectra[-1].value)

    def test_three_libration_variants_differ(self, tabulated):
        _, spectra = tabulated
        assert not np.allclose(spectra[10, :, 0].value, spectra[10, :, 2].value)

    def test_libration_variants_bracket_the_median(self, tabulated):
        """The zero-libration variant must sit between the two offsets."""
        _, spectra = tabulated
        low, mid, high = (spectra[10, :, i].value for i in range(3))
        assert np.all((mid >= np.minimum(low, high)) & (mid <= np.maximum(low, high)))

    def test_libration_spread_is_a_small_correction(self, tabulated):
        """The bracket represents an uncertainty, not a leading-order term."""
        _, spectra = tabulated
        variants = spectra[10].value
        spread = (variants.max(axis=-1) - variants.min(axis=-1)) / variants[:, 1]
        assert np.all(spread < 0.1)

    def test_full_moon_spectrum_is_red(self, tabulated):
        """Reflected moonlight inherits the lunar albedo's rise to the red.

        The tabulated spectrum is a photon flux density against a flat
        energy-flux solar spectrum, so it rises for two reasons: the albedo
        roughly doubles across this range, and the photon energy falls.
        """
        _, spectra = tabulated
        full_moon = spectra[0, :, 1].value
        assert full_moon[-1] > full_moon[0]

    def test_brightness_falls_away_from_full_moon(self, tabulated):
        """Every phase step past the opposition surge must be dimmer."""
        _, spectra = tabulated
        median_over_wavelength = np.median(spectra[:, :, 1].value, axis=1)
        assert np.all(np.diff(median_over_wavelength[2:]) < 0)

    def test_new_moon_is_orders_of_magnitude_fainter(self, tabulated):
        _, spectra = tabulated
        assert np.median(spectra[-1, :, 1].value) < 1e-3 * np.median(
            spectra[0, :, 1].value
        )
