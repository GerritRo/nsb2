"""Tests for :mod:`nsb2.core.interpolation`."""

import astropy.units as u
import numpy as np
import pytest

from nsb2.core.interpolation import UnitRegularGridInterpolator


class TestUnitRegularGridInterpolator:
    def test_requires_a_quantity(self):
        with pytest.raises(ValueError, match="Quantity"):
            UnitRegularGridInterpolator([np.arange(3)], np.ones(3))

    def test_preserves_unit(self):
        rgi = UnitRegularGridInterpolator([np.arange(3)], np.ones(3) * u.m**2)
        assert rgi(np.array([[1.0]])).unit == u.m**2

    def test_converts_to_requested_unit(self):
        rgi = UnitRegularGridInterpolator(
            [np.arange(3)], np.ones(3) * u.m**2, unit=u.cm**2
        )
        assert rgi(np.array([[1.0]])).to_value(u.cm**2) == pytest.approx(1e4)

    def test_linear_interpolation(self):
        rgi = UnitRegularGridInterpolator(
            [np.arange(3)], np.array([0.0, 10.0, 20.0]) * u.m**2
        )
        assert rgi(np.array([[0.5]])).to_value(u.m**2)[0] == pytest.approx(5.0)

    def test_narrow_range_stays_linear(self):
        rgi = UnitRegularGridInterpolator(
            [np.arange(3)], np.array([1.0, 2.0, 3.0]) * u.m**2
        )
        assert rgi.log_values is False

    def test_wide_positive_range_switches_to_log_space(self):
        rgi = UnitRegularGridInterpolator(
            [np.arange(3)], np.array([1.0, 1e3, 1e6]) * u.m**2
        )
        assert rgi.log_values is True

    def test_log_space_interpolation_is_geometric(self):
        rgi = UnitRegularGridInterpolator(
            [np.arange(3)], np.array([1.0, 1e3, 1e6]) * u.m**2
        )
        # Halfway between 1 and 1e3 in log space is 1e1.5.
        assert rgi(np.array([[0.5]])).to_value(u.m**2)[0] == pytest.approx(10**1.5)

    def test_values_crossing_zero_stay_linear(self):
        """Log space is only valid for strictly positive values."""
        rgi = UnitRegularGridInterpolator(
            [np.arange(3)], np.array([0.0, 1e3, 1e6]) * u.m**2
        )
        assert rgi.log_values is False

    def test_accepts_scalar_coordinate(self):
        rgi = UnitRegularGridInterpolator([np.arange(3)], np.ones(3) * u.m**2)
        assert rgi(np.array([1.0])).shape == (1,)
