"""Interface for atmospheric extinction and scattering models.

An atmosphere answers two questions.  Along the direct path it says what
fraction of a source's light survives the journey to the telescope.  Along
the scattered path it says how much light arriving from one direction is
redirected into another, which is what turns a bright Moon into a raised
background across the whole field of view.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import astropy.units as u
import numpy as np

__all__ = [
    "Atmosphere",
    "haversine",
]


def haversine(delta_lon, lat1, lat2):
    """Angular distance between two points on a sphere.

    Uses the haversine form, which stays numerically accurate for the small
    separations that dominate scattering close to a bright source, where the
    spherical law of cosines loses precision.

    Parameters
    ----------
    delta_lon : array_like
        Difference in longitude, in radians.
    lat1, lat2 : array_like
        Latitudes of the two points, in radians.  Shapes must be mutually
        broadcastable with ``delta_lon``.

    Returns
    -------
    numpy.ndarray
        Angular separation in radians, in ``[0, pi]``.

    Examples
    --------
    >>> import numpy as np
    >>> round(float(np.degrees(haversine(np.radians(90.0), 0.0, 0.0))), 6)
    90.0
    """
    delta_lat = lat1 - lat2
    sin_delta_lat = np.sin(delta_lat / 2) ** 2
    sin_sum_lat = np.sin((lat1 + lat2) / 2) ** 2
    sin_delta_lon = np.sin(delta_lon / 2) ** 2
    return 2 * np.arcsin(
        np.sqrt(sin_delta_lat + (1 - sin_delta_lat - sin_sum_lat) * sin_delta_lon)
    )


class Atmosphere(ABC):
    """Base class for atmospheric models.

    Subclasses implement :meth:`_compute_extinction` and
    :meth:`_compute_scattering`.  The public :meth:`extinction` and
    :meth:`scattering` methods wrap those with the unit handling, so that
    implementations can work in plain arrays.
    """

    def extinction(self, alt, az, wvl: u.Quantity) -> np.ndarray:
        """Compute the transmission along a line of sight.

        Parameters
        ----------
        alt, az : array_like
            Source altitude and azimuth in radians.  Shapes must be mutually
            broadcastable.
        wvl : astropy.units.Quantity
            Wavelength grid, shape ``(W,)``.

        Returns
        -------
        numpy.ndarray
            Dimensionless transmission in ``[0, 1]``, broadcastable to
            ``(..., W)``.
        """
        return self._compute_extinction(alt, az, wvl)

    def scattering(self, eval_alt, eval_az, alt, az, wvl: u.Quantity) -> u.Quantity:
        """Compute the scattering kernel between two directions.

        Parameters
        ----------
        eval_alt, eval_az : array_like
            Altitude and azimuth of the direction being observed, in radians.
        alt, az : array_like
            Altitude and azimuth of the illuminating source, in radians.
            Shapes must be mutually broadcastable with the evaluation point.
        wvl : astropy.units.Quantity
            Wavelength grid, shape ``(W,)``.

        Returns
        -------
        astropy.units.Quantity
            Scattering kernel in ``1 / sr``, broadcastable to ``(..., W)``.
        """
        return self._compute_scattering(eval_alt, eval_az, alt, az, wvl) / u.radian**2

    @abstractmethod
    def _compute_extinction(self, alt, az, wvl: u.Quantity) -> np.ndarray:
        """Compute the dimensionless transmission factor.

        Implementations must support numpy broadcasting on ``alt`` and ``az``.

        Parameters
        ----------
        alt, az : array_like
            Source altitude and azimuth in radians.
        wvl : astropy.units.Quantity
            Wavelength grid, shape ``(W,)``.

        Returns
        -------
        numpy.ndarray
            Dimensionless transmission in ``[0, 1]``, broadcastable to
            ``(..., W)``.
        """
        ...

    @abstractmethod
    def _compute_scattering(
        self, eval_alt, eval_az, alt, az, wvl: u.Quantity
    ) -> np.ndarray:
        """Compute the scattering kernel before the ``1 / sr`` is attached.

        Implementations must support numpy broadcasting on all positional
        arguments.

        Parameters
        ----------
        eval_alt, eval_az : array_like
            Altitude and azimuth of the direction being observed, in radians.
        alt, az : array_like
            Altitude and azimuth of the illuminating source, in radians.
        wvl : astropy.units.Quantity
            Wavelength grid, shape ``(W,)``.

        Returns
        -------
        numpy.ndarray
            Dimensionless scattering kernel, broadcastable to ``(..., W)``.
        """
        ...
