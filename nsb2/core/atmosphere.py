from __future__ import annotations

from abc import ABC, abstractmethod

import astropy.units as u
import numpy as np


def haversine(delta_lon, lat1, lat2):
    """The haversine angular distance formula."""
    delta_lat = lat1 - lat2
    sin_delta_lat = np.sin(delta_lat / 2) ** 2
    sin_sum_lat = np.sin((lat1 + lat2) / 2) ** 2
    sin_delta_lon = np.sin(delta_lon / 2) ** 2
    return 2 * np.arcsin(
        np.sqrt(sin_delta_lat + (1 - sin_delta_lat - sin_sum_lat) * sin_delta_lon)
    )


class Atmosphere(ABC):
    """Base class for atmospheric models.
    """

    def extinction(self, alt, az, wvl: u.Quantity) -> np.ndarray:
        """Compute extinction weights along line of sight.

        Parameters
        ----------
        alt, az : array-like
            Source altitude and azimuth in radians.  Shapes must be mutually
            broadcastable.
        wvl : Quantity, shape (W,)
            Wavelength grid.

        Returns
        -------
        weights : ndarray, shape broadcastable to (..., W)
            Dimensionless transmission factor in [0, 1].
        """
        return self._compute_extinction(alt, az, wvl)

    def scattering(self, eval_alt, eval_az, alt, az, wvl: u.Quantity) -> u.Quantity:
        """Compute scattering kernel.

        Parameters
        ----------
        eval_alt, eval_az : array-like
            Evaluation-point altitude/azimuth in radians.
        alt, az : array-like
            Source altitude/azimuth in radians.
        wvl : Quantity, shape (W,)
            Wavelength grid.

        Returns
        -------
        kernel : ndarray, shape broadcastable to (..., W), units 1/sr
            Scattering kernel with ``1/sr`` units.
        """
        return self._compute_scattering(eval_alt, eval_az, alt, az, wvl) / u.radian**2

    @abstractmethod
    def _compute_extinction(self, alt, az, wvl: u.Quantity) -> np.ndarray:
        """Compute dimensionless extinction (transmission) factor.

        Must support numpy broadcasting on ``alt`` and ``az``

        Parameters
        ----------
        alt, az : array-like
            Source altitude and azimuth in radians.
        wvl : Quantity, shape (W,)
            Wavelength grid.

        Returns
        -------
        ndarray, shape broadcastable to (..., W)
            Dimensionless transmission in [0, 1].
        """
        ...

    @abstractmethod
    def _compute_scattering(self, eval_alt, eval_az, alt, az, wvl: u.Quantity) -> np.ndarray:
        """Compute dimensionless scattering kernel (before ``/sr``).

        Must support numpy broadcasting on all positional arguments.

        Parameters
        ----------
        eval_alt, eval_az : array-like
            Evaluation-point altitude/azimuth (radians).
        alt, az : array-like
            Source altitude/azimuth (radians).
        wvl : Quantity, shape (W,)
            Wavelength grid.

        Returns
        -------
        ndarray, shape broadcastable to (..., W)
            Dimensionless scattering kernel.
        """
        ...
