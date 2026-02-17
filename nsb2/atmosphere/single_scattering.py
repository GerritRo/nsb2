from __future__ import annotations

from typing import Callable

import astropy.units as u
import numpy as np

from nsb2.core.atmosphere import Atmosphere, haversine


class SingleScatteringAtmosphere(Atmosphere):
    """Single-scattering atmosphere with Rayleigh + Mie components.

    Parameters
    ----------
    airmass_func : callable
        Airmass function X(zenith_angle).
    tau_rayleigh : callable
        Rayleigh scattering optical depth as a function of wavelength.
    tau_mie : callable
        Mie scattering optical depth as a function of wavelength.
    tau_absorption : callable
        Absorption optical depth as a function of wavelength.
    g : float
        Henyey-Greenstein asymmetry parameter for Mie scattering.
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
        """Rayleigh phase function."""
        return 1 / (4 * np.pi) * 3 / 4 * (1 + np.cos(theta) ** 2)

    @staticmethod
    def _henyey_greenstein(g, theta):
        """Henyey-Greenstein phase function."""
        gsq = g ** 2
        return 1 / (4 * np.pi) * (1 - gsq) / (1 + gsq - 2 * g * np.cos(theta)) ** 1.5

    def _compute_extinction(self, alt, az, wvl: u.Quantity) -> np.ndarray:
        tau = self.tau_rayleigh(wvl) + self.tau_mie(wvl) + self.tau_absorption(wvl)
        return np.exp(-tau[np.newaxis, :] * self.X(np.pi / 2 - alt)[:, np.newaxis])

    def _compute_scattering(self, eval_alt, eval_az, alt, az, wvl: u.Quantity) -> np.ndarray:
        tau_r = self.tau_rayleigh(wvl)
        tau_m = self.tau_mie(wvl)
        tau = tau_r + tau_m + self.tau_absorption(wvl)
        theta = haversine(eval_az - az, eval_alt, alt)

        return self._indicatrix(tau_r, tau_m, tau, theta) * self._gradation(tau, np.pi / 2 - eval_alt, np.pi / 2 - alt)

    def _indicatrix(self, tau_r, tau_m, tau, theta):
        frac_r = tau_r / tau
        frac_m = tau_m / tau
        rho_r = self._rayleigh(theta)
        rho_m = self._henyey_greenstein(self.g, theta)
        return frac_r * rho_r + frac_m * rho_m

    def _gradation(self, tau, Z, z):
        sec_Z = self.X(Z)
        sec_z = self.X(z)
        with np.errstate(divide='ignore', invalid='ignore'):
            sec_diff = sec_Z / (sec_z - sec_Z)
            exp_diff = np.exp(-tau * sec_Z) - np.exp(-tau * sec_z)
            return np.where(Z == z, sec_Z * tau * np.exp(-sec_Z * tau), sec_diff * exp_diff)
