"""Tests for the offline parts of :mod:`nsb2.emitter.zodiacal`."""

import astropy.units as u
import numpy as np
import pytest

from nsb2.emitter.zodiacal import (
    ELONGATION_RANGE,
    REFERENCE_WAVELENGTH,
    color_correction,
    helioecliptic_longitude,
    solar_elongation,
)


class TestHelioeclipticLongitude:
    def test_sun_is_at_zero(self):
        assert helioecliptic_longitude(0.0) == pytest.approx(0.0)

    def test_folds_the_western_half_onto_the_eastern(self):
        east = helioecliptic_longitude(np.deg2rad(10))
        west = helioecliptic_longitude(np.deg2rad(350))
        assert west == pytest.approx(east)

    def test_result_never_exceeds_pi(self):
        lon = np.linspace(0, 4 * np.pi, 101)
        folded = helioecliptic_longitude(lon)
        assert np.all((folded >= 0) & (folded <= np.pi + 1e-12))

    def test_antisolar_point_maps_to_pi(self):
        assert helioecliptic_longitude(np.pi) == pytest.approx(np.pi)


class TestSolarElongation:
    def test_zero_at_the_sun(self):
        assert solar_elongation(0.0, 0.0) == pytest.approx(0.0)

    def test_equals_longitude_on_the_ecliptic(self):
        lon = np.deg2rad([10, 45, 90, 170])
        np.testing.assert_allclose(solar_elongation(lon, 0.0), lon)

    def test_accounts_for_ecliptic_latitude(self):
        """The great-circle distance, not the longitude offset.

        Directly above the Sun by sixty degrees is sixty degrees away, even
        though the longitude offset is zero.
        """
        assert solar_elongation(0.0, np.deg2rad(60)) == pytest.approx(np.deg2rad(60))

    def test_combines_longitude_and_latitude(self):
        eps = solar_elongation(np.deg2rad(30), np.deg2rad(30))
        assert np.rad2deg(eps) == pytest.approx(41.4, abs=0.1)

    def test_ecliptic_pole_is_ninety_degrees_from_the_sun(self):
        lon = np.deg2rad([0, 45, 120, 300])
        np.testing.assert_allclose(
            solar_elongation(lon, np.pi / 2), np.pi / 2, atol=1e-8
        )

    def test_never_exceeds_pi(self):
        lon, lat = np.meshgrid(
            np.linspace(0, 2 * np.pi, 37), np.linspace(-1.5, 1.5, 21)
        )
        eps = solar_elongation(lon, lat)
        assert np.all((eps >= 0) & (eps <= np.pi + 1e-12))


class TestColorCorrection:
    def test_unity_at_the_reference_wavelength(self):
        near, far = color_correction(u.Quantity([REFERENCE_WAVELENGTH]))
        assert float(near[0]) == pytest.approx(1.0)
        assert float(far[0]) == pytest.approx(1.0)

    def test_returns_one_curve_per_elongation_endpoint(self):
        assert color_correction(np.linspace(300, 700, 9) * u.nm).shape == (2, 9)

    def test_zodiacal_light_is_redder_than_the_sun(self):
        near, _ = color_correction([300, 700] * u.nm)
        assert near[0] < 1.0 < near[1]

    def test_reddening_is_stronger_close_to_the_sun(self):
        """The near-Sun curve must depart from unity further than the far one."""
        near, far = color_correction([300, 700] * u.nm)
        assert abs(near[0] - 1) > abs(far[0] - 1)
        assert abs(near[1] - 1) > abs(far[1] - 1)

    def test_uses_base_ten_logarithm(self):
        """A decade below the reference, the factor is 1 minus the slope.

        Guards against a natural logarithm, which would overstate the
        reddening by ln(10) and is the difference between this and an
        earlier revision of the model.
        """
        near, far = color_correction([50] * u.nm)
        assert float(near[0]) == pytest.approx(1 - 1.2)
        assert float(far[0]) == pytest.approx(1 - 0.9)

    def test_stays_positive_across_the_leinert_range(self):
        """Leinert tabulates down to 0.2 micron; a negative factor is unphysical."""
        corrections = color_correction(np.linspace(200, 2500, 200) * u.nm)
        assert np.all(corrections > 0)

    def test_slope_changes_either_side_of_the_reference(self):
        blue = color_correction([250] * u.nm)[0, 0]
        red = color_correction([1000] * u.nm)[0, 0]
        # Same factor-of-two distance in log space, different slopes.
        assert abs(blue - 1) / np.log10(2) == pytest.approx(1.2, abs=1e-6)
        assert abs(red - 1) / np.log10(2) == pytest.approx(0.8, abs=1e-6)

    def test_elongation_range_is_ordered(self):
        assert ELONGATION_RANGE[0] < ELONGATION_RANGE[1]

    def test_rejects_a_wavelength_that_is_not_a_length(self):
        """@quantity_input turns an ambiguous unit into an immediate error."""
        with pytest.raises(u.UnitsError):
            color_correction([500] * u.s)

    def test_accepts_any_length_unit(self):
        in_nm = color_correction([500, 700] * u.nm)
        in_angstrom = color_correction([5000, 7000] * u.angstrom)
        np.testing.assert_allclose(in_angstrom, in_nm)
