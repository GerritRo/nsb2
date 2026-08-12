import astropy.units as u
import numpy as np
import pytest

from nsb2.core.interpolation import UnitRegularGridInterpolator


class TestUnitRegularGridInterpolator:
    def test_requires_a_quantity_and_returns_one(self):
        with pytest.raises(ValueError, match="Quantity"):
            UnitRegularGridInterpolator([np.arange(3)], np.ones(3))

        rgi = UnitRegularGridInterpolator([np.arange(3)], np.ones(3) * u.m**2)
        assert rgi(np.array([[1.0]])).unit == u.m**2

        converted = UnitRegularGridInterpolator(
            [np.arange(3)], np.ones(3) * u.m**2, unit=u.cm**2
        )
        assert converted(np.array([[1.0]])).to_value(u.cm**2) == pytest.approx(1e4)

    def test_interpolates_linearly_over_a_narrow_range(self):
        rgi = UnitRegularGridInterpolator(
            [np.arange(3)], np.array([0.0, 10.0, 20.0]) * u.m**2
        )
        assert rgi.log_values is False
        assert rgi(np.array([[0.5]])).to_value(u.m**2)[0] == pytest.approx(5.0)
        # A bare coordinate is promoted rather than rejected.
        assert rgi(np.array([1.0])).shape == (1,)

    def test_switches_to_log_space_over_many_decades(self):
        rgi = UnitRegularGridInterpolator(
            [np.arange(3)], np.array([1.0, 1e3, 1e6]) * u.m**2
        )
        assert rgi.log_values is True
        # Halfway between 1 and 1e3 in log space is 1e1.5.
        assert rgi(np.array([[0.5]])).to_value(u.m**2)[0] == pytest.approx(10**1.5)

        # Log space is only valid for strictly positive values.
        with_zero = UnitRegularGridInterpolator(
            [np.arange(3)], np.array([0.0, 1e3, 1e6]) * u.m**2
        )
        assert with_zero.log_values is False
