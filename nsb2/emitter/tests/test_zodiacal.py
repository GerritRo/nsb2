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
    def test_folds_the_western_half_onto_the_eastern(self):
        assert helioecliptic_longitude(0.0) == pytest.approx(0.0)
        assert helioecliptic_longitude(np.pi) == pytest.approx(np.pi)
        assert helioecliptic_longitude(np.deg2rad(350)) == pytest.approx(
            helioecliptic_longitude(np.deg2rad(10))
        )
        folded = helioecliptic_longitude(np.linspace(0, 4 * np.pi, 101))
        assert np.all((folded >= 0) & (folded <= np.pi + 1e-12))


class TestSolarElongation:
    def test_is_the_great_circle_distance_from_the_sun(self):
        assert solar_elongation(0.0, 0.0) == pytest.approx(0.0)

        on_ecliptic = np.deg2rad([10, 45, 90, 170])
        np.testing.assert_allclose(solar_elongation(on_ecliptic, 0.0), on_ecliptic)
        assert solar_elongation(0.0, np.deg2rad(60)) == pytest.approx(np.deg2rad(60))
        assert np.rad2deg(
            solar_elongation(np.deg2rad(30), np.deg2rad(30))
        ) == pytest.approx(41.4, abs=0.1)
        np.testing.assert_allclose(
            solar_elongation(np.deg2rad([0, 45, 120, 300]), np.pi / 2),
            np.pi / 2,
            atol=1e-8,
        )

        lon, lat = np.meshgrid(
            np.linspace(0, 2 * np.pi, 37), np.linspace(-1.5, 1.5, 21)
        )
        eps = solar_elongation(lon, lat)
        assert np.all((eps >= 0) & (eps <= np.pi + 1e-12))


class TestColorCorrection:
    def test_reddens_away_from_the_reference_wavelength(self):
        assert ELONGATION_RANGE[0] < ELONGATION_RANGE[1]
        assert color_correction(np.linspace(300, 700, 9) * u.nm).shape == (2, 9)

        at_reference = color_correction(u.Quantity([REFERENCE_WAVELENGTH]))
        np.testing.assert_allclose(at_reference, 1.0)

        near, far = color_correction([300, 700] * u.nm)
        assert near[0] < 1.0 < near[1], "blue is suppressed, red enhanced"
        assert abs(near[0] - 1) > abs(far[0] - 1)
        assert abs(near[1] - 1) > abs(far[1] - 1)

        assert np.all(color_correction(np.linspace(200, 2500, 200) * u.nm) > 0)

    def test_uses_base_ten_logarithm(self):
        """Regression test against previous mistake."""
        near, far = color_correction([50] * u.nm)
        assert float(near[0]) == pytest.approx(1 - 1.2)
        assert float(far[0]) == pytest.approx(1 - 0.9)

        blue = color_correction([250] * u.nm)[0, 0]
        red = color_correction([1000] * u.nm)[0, 0]
        assert abs(blue - 1) / np.log10(2) == pytest.approx(1.2, abs=1e-6)
        assert abs(red - 1) / np.log10(2) == pytest.approx(0.8, abs=1e-6)

    def test_requires_a_wavelength_but_accepts_any_length_unit(self):
        with pytest.raises(u.UnitsError):
            color_correction([500] * u.s)
        np.testing.assert_allclose(
            color_correction([5000, 7000] * u.angstrom),
            color_correction([500, 700] * u.nm),
        )
