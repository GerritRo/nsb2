import logging

import numpy as np
from scipy.interpolate import RegularGridInterpolator

__all__ = [
    "UnitRegularGridInterpolator",
]


logger = logging.getLogger(__name__)

DEFAULT_LOG_THRESHOLD = 3


class UnitRegularGridInterpolator:
    """Regular-grid interpolator that preserves units and dynamic range.

    Wraps :class:`scipy.interpolate.RegularGridInterpolator` with two
    additions needed by the lookup tables in :mod:`nsb2.core.solver`.  Units
    are stripped before interpolation and reattached afterwards, and values
    spanning many decades are interpolated in log space.

    Parameters
    ----------
    points : tuple of numpy.ndarray
        Grid coordinates along each axis.
    values : astropy.units.Quantity
        Values on the grid.  Must carry a unit.
    threshold : int, optional
        Interpolate in log space when the values span more than this many
        decades and are strictly positive.  Default is
        :data:`DEFAULT_LOG_THRESHOLD`.
    unit : astropy.units.UnitBase, optional
        Unit to convert the values to and return results in.  Default is the
        unit of ``values``.
    **kwargs
        Passed through to :class:`scipy.interpolate.RegularGridInterpolator`.

    Attributes
    ----------
    unit : astropy.units.UnitBase
        Unit of the interpolated results.
    log_values : bool
        Whether interpolation is performed in log space.

    Raises
    ------
    ValueError
        If ``values`` is not an `~astropy.units.Quantity`.
    """

    def __init__(
        self, points, values, threshold=DEFAULT_LOG_THRESHOLD, unit=None, **kwargs
    ):
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
            logger.debug(
                "interpolating in log space: values span %.1f decades",
                np.log10(max_val / min_val),
            )

        self.interpolator = RegularGridInterpolator(points, val_data, **kwargs)

    def __call__(self, xi):
        """Interpolate at the given coordinates.

        Parameters
        ----------
        xi : array_like
            Coordinates to evaluate at, with the grid axes last.

        Returns
        -------
        astropy.units.Quantity
            Interpolated values, in :attr:`unit`.
        """
        xi = np.atleast_2d(xi)
        result = self.interpolator(xi)
        if self.log_values:
            result = 10**result
        return result * self.unit
