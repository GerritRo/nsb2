import astropy.units as u
import numpy as np
import pytest
from astropy.coordinates import get_body
from astropy.time import Time

from nsb2.emitter.moon import (
    LIBRATION_SPREAD,
    LUNAR_REFERENCE_DISTANCE,
    N_PHASE_SAMPLES,
    N_PHASE_SAMPLES_PER_BRANCH,
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


@pytest.fixture(scope="module")
def tabulated(rolo_table):
    """Reflected spectra for a flat unit solar spectrum."""
    wvl = np.linspace(400, 1000, 25) * u.nm
    solar = np.ones(25) * u.erg / u.s / u.cm**2 / u.nm
    return tabulate_lunar_spectra(rolo_table, wvl, solar)


class TestLoadRoloTable:
    def test_holds_25_optical_bands_of_finite_coefficients(self, rolo_table):
        """One wavelength column plus ten coefficients, in ascending order."""
        assert rolo_table.shape == (25, 11)
        assert np.all(np.isfinite(rolo_table))
        wavelengths = rolo_table[:, 0]
        assert np.all(np.diff(wavelengths) > 0)
        assert wavelengths[0] == pytest.approx(350.0)
        assert wavelengths[-1] == pytest.approx(1059.5)


class TestRoloLogAlbedo:
    def test_is_physical_and_peaks_at_full_moon(self, rolo_table):
        """Phase angle zero is full Moon; brightness falls away from it."""
        for row in rolo_table:
            assert 0 < np.exp(rolo_log_albedo(row[1:], 0.0, 0.0)) < 1

        band = rolo_table[10][1:]
        assert rolo_log_albedo(band, 0.0, 0.0) > rolo_log_albedo(band, np.pi / 2, 0.0)
        past_surge = np.linspace(np.deg2rad(20), np.pi, 20)
        values = np.array([rolo_log_albedo(band, g, 0.0) for g in past_surge])
        assert np.all(np.diff(values) < 0)

        centred = rolo_log_albedo(rolo_table[5][1:], 0.5, 0.0)
        librated = rolo_log_albedo(rolo_table[5][1:], 0.5, np.deg2rad(LIBRATION_SPREAD))
        assert centred != pytest.approx(librated)

    def test_phase_and_libration_are_independent_arguments(self, rolo_table):
        """Libration is worth well under a per cent; phase carries the model."""
        band = rolo_table[10][1:]
        limit = np.deg2rad(LIBRATION_SPREAD)
        at_phase = [rolo_log_albedo(band, g, 0.0) for g in (0.0, np.pi / 2)]
        at_libration = [rolo_log_albedo(band, 0.0, p) for p in (-limit, limit)]
        assert abs(np.diff(at_phase)[0]) > 2
        assert abs(np.diff(at_libration)[0]) < 0.02

    def test_accepts_scalars_arrays_and_plain_lists(self, rolo_table):
        coefficients = rolo_table[0][1:]
        scalar = rolo_log_albedo(coefficients, 0.5, 0.1)
        assert np.isscalar(scalar) or scalar.shape == ()

        phases = np.linspace(0, 1, 7)
        librations = np.linspace(-0.1, 0.1, 7)
        assert rolo_log_albedo(coefficients, phases, librations).shape == (7,)

        from_list = rolo_log_albedo(list(coefficients), 0.3, 0.1)
        assert from_list == pytest.approx(rolo_log_albedo(coefficients, 0.3, 0.1))


class TestRoloAlbedo:
    def test_interpolates_the_bands_onto_a_wavelength_grid(self, rolo_table):
        wvl = np.linspace(400, 1000, 13) * u.nm
        albedo = rolo_albedo(rolo_table, wvl, 0.0, 0.1)
        assert albedo.shape == (13,)
        assert np.all((albedo > 0) & (albedo < 1))

        band = rolo_table[7]
        raw = np.exp(rolo_log_albedo(band[1:], 0.4, 0.1))
        interpolated = rolo_albedo(rolo_table, [band[0]] * u.nm, 0.4, 0.1)[0]
        assert interpolated == pytest.approx(raw * ROLO_ALBEDO_SCALE)

        in_angstrom = rolo_albedo(rolo_table, [5000] * u.angstrom, 0.2, 0.1)
        in_nm = rolo_albedo(rolo_table, [500] * u.nm, 0.2, 0.1)
        assert float(in_angstrom[0]) == pytest.approx(float(in_nm[0]))

    def test_moon_is_red_within_the_physical_libration_range(self, rolo_table):
        wvl = np.linspace(350, 1059.5, 20) * u.nm
        for libration in np.deg2rad([-LIBRATION_SPREAD, 0, LIBRATION_SPREAD]):
            albedo = rolo_albedo(rolo_table, wvl, 0.0, libration)
            assert albedo[-1] > 2 * albedo[0], "albedo must roughly double"
            assert albedo[-5:].mean() > albedo[:5].mean()
            assert float(albedo[0]) == pytest.approx(0.086, abs=0.01)
            assert float(albedo[-1]) == pytest.approx(0.212, abs=0.01)

    def test_out_of_range_libration_is_unphysical(self, rolo_table):
        """Why the libration argument must never be driven by the phase angle.

        The fifth-power term turns over near 90 degrees; extrapolated to pi it
        dominates and turns the Moon blue.
        """
        wvl = np.linspace(350, 1059.5, 20) * u.nm
        sensible = rolo_albedo(rolo_table, wvl, 0.0, 0.0)
        out_of_range = rolo_albedo(rolo_table, wvl, 0.0, np.pi)
        assert sensible[-1] > 2 * sensible[0]
        assert out_of_range[-1] < out_of_range[0]

    def test_phase_curve_matches_the_observed_lunar_phase_law(self, rolo_table):
        """A(0)/A(g) against the Moon's measured visual magnitudes.

        This is the external check the libration coupling failed: it inflated
        the curve by 7 per cent at quarter phase and 61 per cent at new Moon.
        """
        wvl = [550] * u.nm
        full = rolo_albedo(rolo_table, wvl, 0.0, 0.0)[0]
        for degrees, expected in [(30, 2.13), (60, 5.0), (90, 12.4), (120, 45.6)]:
            ratio = full / rolo_albedo(rolo_table, wvl, np.deg2rad(degrees), 0.0)[0]
            assert ratio == pytest.approx(expected, rel=0.15)


class TestLunarEphemeris:
    def test_phase_angle_is_zero_at_full_moon_and_pi_at_new(self):
        full = lunar_phase_angle(FULL_MOON)
        new = lunar_phase_angle(NEW_MOON)
        assert full.shape == new.shape == (1,)
        assert abs(np.rad2deg(full[0])) < 15
        assert abs(np.rad2deg(new[0])) > 165

        assert 350000 < LUNAR_REFERENCE_DISTANCE < 410000

        weight = lunar_distance_scaling(FULL_MOON)
        assert weight.shape == (1,)
        assert weight.unit == u.dimensionless_unscaled
        assert float(weight[0]) == pytest.approx(6.4236e-5 / np.pi, rel=0.15)

        times = Time("2024-01-01") + np.arange(0, 28, 0.5) * u.day
        distances = get_body("moon", times).distance.to_value(u.km)
        perigee = times[np.argmin(distances)]
        apogee = times[np.argmax(distances)]
        assert lunar_distance_scaling(perigee)[0] > lunar_distance_scaling(apogee)[0]

    def test_phase_angle_is_positive_while_waxing(self):
        """The sign carries the waxing/waning branch of the same phase angle."""
        first_quarter = lunar_phase_angle(Time("2024-01-18T03:53:00"))
        last_quarter = lunar_phase_angle(Time("2024-02-02T23:18:00"))
        assert np.rad2deg(first_quarter[0]) == pytest.approx(90, abs=5)
        assert np.rad2deg(last_quarter[0]) == pytest.approx(-90, abs=5)

    def test_phase_angle_accepts_an_array_of_times(self):
        times = Time("2024-01-11T11:57:00") + np.arange(0, 30, 3) * u.day
        angles = lunar_phase_angle(times)
        assert angles.shape == times.shape
        assert np.all(np.abs(angles) <= np.pi)
        one_by_one = [lunar_phase_angle(t)[0] for t in times]
        assert angles == pytest.approx(one_by_one)


#: Index of the full Moon on the signed phase axis, which is its centre.
FULL_MOON_INDEX = N_PHASE_SAMPLES_PER_BRANCH - 1

#: A mid-phase sample, on the waxing half.
MID_PHASE_INDEX = FULL_MOON_INDEX + 10


class TestTabulateLunarSpectra:
    def test_tabulates_photon_spectra_over_phase_and_libration(self, tabulated):
        phase_angles, spectra = tabulated
        assert phase_angles.shape == (N_PHASE_SAMPLES,)
        assert phase_angles[0] == pytest.approx(-np.pi)
        assert phase_angles[FULL_MOON_INDEX] == pytest.approx(0.0)
        assert phase_angles[-1] == pytest.approx(np.pi)
        assert spectra.shape == (N_PHASE_SAMPLES, 25, 3)
        assert spectra.unit.is_equivalent(1 / (u.nm * u.s * u.cm**2))
        assert np.all(spectra.value > 0) and np.all(np.isfinite(spectra.value))

    def test_brightness_peaks_at_full_moon_and_falls_towards_new(self, tabulated):
        _, spectra = tabulated
        median = np.median(spectra[:, :, 1].value, axis=1)
        assert np.argmax(median) == FULL_MOON_INDEX
        assert np.all(np.diff(median[: FULL_MOON_INDEX + 1]) > 0)
        assert np.all(np.diff(median[FULL_MOON_INDEX:]) < 0)
        assert median[0] < 1e-3 * median[FULL_MOON_INDEX]
        assert median[-1] < 1e-3 * median[FULL_MOON_INDEX]

        full_moon = spectra[FULL_MOON_INDEX, :, 1].value
        assert full_moon[-1] > full_moon[0]

    def test_is_symmetric_about_full_moon(self, tabulated):
        """The fit omits the Sun's selenographic terms, so it is even in phase.

        Any asymmetry would mean the phase angle is leaking into the
        libration argument, which is what drove that polynomial out of its
        fitted range.
        """
        phase_angles, spectra = tabulated
        assert phase_angles == pytest.approx(-phase_angles[::-1])
        assert spectra.value == pytest.approx(spectra.value[::-1])

    def test_libration_variants_bracket_the_median(self, tabulated):
        _, spectra = tabulated
        low, mid, high = (spectra[MID_PHASE_INDEX, :, i].value for i in range(3))
        assert np.all(low < mid) and np.all(mid < high)

    def test_libration_spread_stays_small_at_every_phase(self, tabulated):
        """The variants bracket a sub-per-cent uncertainty, not a fan.

        Coupling the libration to the phase angle used to blow this up to
        134 per cent at new Moon, which propagated into the Prediction error
        bar.
        """
        _, spectra = tabulated
        variants = spectra.value
        spread = (variants.max(axis=-1) - variants.min(axis=-1)) / variants[:, :, 1]
        assert np.all(spread < 0.02)
