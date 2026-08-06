"""Single-scattering atmosphere model."""

from __future__ import annotations

from collections.abc import Callable

import astropy.units as u
import numpy as np

from nsb2.core.atmosphere import Atmosphere, haversine

__all__ = [
    "SingleScatteringAtmosphere",
]


class SingleScatteringAtmosphere(Atmosphere):
    """Atmosphere in which light is scattered at most once.

    Combines Rayleigh scattering off air molecules with Mie scattering off
    aerosols, following the plane-parallel single-scattering treatment of
    [Krisciunas1991]_ and [Noll2012]_.  The scattering kernel factorises into
    an *indicatrix*, which describes the angular dependence, and a
    *gradation*, which describes how much scattering material lies along the
    two lines of sight.  The Rayleigh component uses the classical dipole
    phase function and the Mie component the forward-peaked approximation of
    [HenyeyGreenstein1941]_.  Ignoring multiple scattering underestimates the
    background close to a very bright source such as a full Moon, but is
    accurate to a few per cent elsewhere.

    Parameters
    ----------
    airmass_func : callable
        Airmass ``X(z)`` as a function of zenith angle in radians.
    tau_rayleigh : callable
        Rayleigh scattering optical depth as a function of wavelength.
    tau_mie : callable
        Mie scattering optical depth as a function of wavelength.
    tau_absorption : callable
        Absorption optical depth as a function of wavelength.  Contributes to
        the extinction but not to the scattering.
    g : float
        Henyey-Greenstein asymmetry parameter of the Mie phase function, in
        ``(-1, 1)``.  Positive values favour forward scattering.

    Examples
    --------
    >>> import numpy as np, astropy.units as u
    >>> atmosphere = SingleScatteringAtmosphere(
    ...     airmass_func=lambda z: 1 / np.cos(z),
    ...     tau_rayleigh=lambda w: 0.1 * (400 * u.nm / w) ** 4,
    ...     tau_mie=lambda w: 0.05 * np.ones_like(w.value),
    ...     tau_absorption=lambda w: 0.01 * np.ones_like(w.value),
    ...     g=0.65,
    ... )
    >>> transmission = atmosphere.extinction(
    ...     np.array([np.pi / 2]), np.array([0.0]), np.array([500]) * u.nm)
    >>> bool(0 < transmission[0, 0] <= 1)
    True
    """

    def __init__(
        self,
        airmass_func: Callable,
        tau_rayleigh: Callable,
        tau_mie: Callable,
        tau_absorption: Callable,
        g: float,
    ) -> None:
        self.X = airmass_func
        self.tau_rayleigh = tau_rayleigh
        self.tau_mie = tau_mie
        self.tau_absorption = tau_absorption
        self.g = g

    @staticmethod
    def _rayleigh(theta):
        """Rayleigh phase function, normalised to unit integral over solid angle.

        Parameters
        ----------
        theta : array_like
            Scattering angle in radians.

        Returns
        -------
        array_like
            Phase function value per steradian.
        """
        return 1 / (4 * np.pi) * 3 / 4 * (1 + np.cos(theta) ** 2)

    @staticmethod
    def _henyey_greenstein(g, theta):
        """Henyey-Greenstein phase function [HenyeyGreenstein1941]_.

        Parameters
        ----------
        g : float
            Asymmetry parameter in ``(-1, 1)``.
        theta : array_like
            Scattering angle in radians.

        Returns
        -------
        array_like
            Phase function value per steradian.
        """
        gsq = g**2
        return 1 / (4 * np.pi) * (1 - gsq) / (1 + gsq - 2 * g * np.cos(theta)) ** 1.5

    def _compute_extinction(self, alt, az, wvl: u.Quantity) -> np.ndarray:
        """Compute Beer-Lambert transmission along the line of sight.

        See :meth:`nsb2.core.atmosphere.Atmosphere._compute_extinction` for
        the parameters and returns.  The result is azimuth independent.
        """
        tau = self.tau_rayleigh(wvl) + self.tau_mie(wvl) + self.tau_absorption(wvl)
        return np.exp(-tau[np.newaxis, :] * self.X(np.pi / 2 - alt)[:, np.newaxis])

    def _compute_scattering(
        self, eval_alt, eval_az, alt, az, wvl: u.Quantity
    ) -> np.ndarray:
        """Compute the single-scattering kernel between two directions.

        See :meth:`nsb2.core.atmosphere.Atmosphere._compute_scattering` for
        the parameters and returns.
        """
        tau_r = self.tau_rayleigh(wvl)
        tau_m = self.tau_mie(wvl)
        tau = tau_r + tau_m + self.tau_absorption(wvl)
        theta = haversine(eval_az - az, eval_alt, alt)

        return self._indicatrix(tau_r, tau_m, tau, theta) * self._gradation(
            tau, np.pi / 2 - eval_alt, np.pi / 2 - alt
        )

    def _indicatrix(self, tau_r, tau_m, tau, theta):
        """Combine the Rayleigh and Mie phase functions.

        The two are weighted by their share of the total optical depth, so
        that the mixture follows the wavelength dependence of the scatterers.

        Parameters
        ----------
        tau_r, tau_m, tau : array_like
            Rayleigh, Mie and total optical depth.
        theta : array_like
            Scattering angle in radians.

        Returns
        -------
        array_like
            Combined phase function value per steradian.
        """
        frac_r = tau_r / tau
        frac_m = tau_m / tau
        rho_r = self._rayleigh(theta)
        rho_m = self._henyey_greenstein(self.g, theta)
        return frac_r * rho_r + frac_m * rho_m

    def _gradation(self, tau, Z, z):
        """Integrate the scattering probability along the two lines of sight.

        Parameters
        ----------
        tau : array_like
            Total optical depth.
        Z : array_like
            Zenith angle of the direction being observed, in radians.
        z : array_like
            Zenith angle of the illuminating source, in radians.

        Returns
        -------
        array_like
            Dimensionless gradation factor.  The general expression is
            singular when the two zenith angles coincide, where its limit is
            used instead.
        """
        sec_Z = self.X(Z)
        sec_z = self.X(z)
        with np.errstate(divide="ignore", invalid="ignore"):
            sec_diff = sec_Z / (sec_z - sec_Z)
            exp_diff = np.exp(-tau * sec_Z) - np.exp(-tau * sec_z)
            return np.where(
                Z == z, sec_Z * tau * np.exp(-sec_Z * tau), sec_diff * exp_diff
            )
