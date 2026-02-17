import numpy as np
from scipy.interpolate import RegularGridInterpolator


class UnitRegularGridInterpolator:
    """RegularGridInterpolator wrapper that handles astropy Quantities and log-space."""

    def __init__(self, points, values, threshold=3, unit=None, **kwargs):
        if not hasattr(values, "unit"):
            raise ValueError("`values` must be an astropy Quantity.")

        self.unit = unit or values.unit
        self.log_values = False
        val_data = values.to_value(self.unit)

        min_val = np.nanmin(val_data)
        max_val = np.nanmax(val_data)
        if min_val > 0 and np.log10(max_val / min_val) > threshold:
            self.log_values = True
            val_data = np.log10(val_data)

        self.interpolator = RegularGridInterpolator(points, val_data, **kwargs)

    def __call__(self, xi):
        xi = np.atleast_2d(xi)
        result = self.interpolator(xi)
        if self.log_values:
            result = 10 ** result
        return result * self.unit
