import astropy.units as u
import numpy as np
import pytest

from nsb2.core.interpolation import UnitRegularGridInterpolator


class TestUnitRegularGridInterpolatorEdgeCases:
    def test_raises_for_non_quantity_values(self):
        points = (np.linspace(0, 1, 5),)
        values = np.ones(5)  # plain array, not a Quantity
        with pytest.raises(ValueError, match="astropy Quantity"):
            UnitRegularGridInterpolator(points, values)

    def test_log_space_interpolation(self):
        """Large dynamic range triggers log-space interpolation."""
        pts = np.linspace(0, 1, 10)
        # Span more than 3 decades → log_values=True
        vals = (10.0 ** np.linspace(0, 4, 10)) * u.ct / u.s
        interp = UnitRegularGridInterpolator((pts,), vals)
        assert interp.log_values is True
        result = interp(np.array([[0.5]]))
        assert result.value > 0
        assert result.unit == u.ct / u.s

    def test_log_space_result_positive(self):
        pts = np.linspace(0, 1, 10)
        vals = (10.0 ** np.linspace(1, 5, 10)) * u.photon / u.s
        interp = UnitRegularGridInterpolator((pts,), vals)
        result = interp(np.array([[0.0], [0.5], [1.0]]))
        assert np.all(result.value > 0)

    def test_linear_space_not_log(self):
        """Small dynamic range does NOT trigger log-space."""
        pts = np.linspace(0, 1, 5)
        vals = np.linspace(1, 2, 5) * u.m**2
        interp = UnitRegularGridInterpolator((pts,), vals)
        assert interp.log_values is False
        result = interp(np.array([[0.5]]))
        assert abs(result.value[0] - 1.5) < 0.01

    def test_custom_unit(self):
        pts = np.linspace(0, 1, 5)
        vals = np.linspace(10, 50, 5) * u.m**2
        interp = UnitRegularGridInterpolator((pts,), vals, unit=u.cm**2)
        assert interp.unit == u.cm**2
