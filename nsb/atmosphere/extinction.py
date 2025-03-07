import numpy as np
import histlite as hl
import scipy
import scipy.integrate as si

import astropy.units as u

from nsb.core.logic import Transmission
from nsb.utils.formulas import k_rayleigh, k_mie


class Noll2012(Transmission):
    """
    Implements the transmission function from Noll et. al (2012)
    """

    def X(self, Z):
        return (1 - 0.972 * (np.sin(Z)) ** 2) ** (-0.5)

    def transmission(self, tau, Z):
        M = self.X(Z)
        M_corr = (self.config["scale"] * np.log10(M) + self.config["offset"]) * M
        return np.exp(-M_corr[:, np.newaxis] * tau)

    def t_args(self, frame, rays):
        Z = np.pi / 2 - rays.coords.alt.rad
        tau = k_rayleigh(frame) + k_mie(frame)

        return (tau, Z)


class Masana2021(Transmission):
    """
    Implements the transmission function from Masana et. al (2021)
    """

    def X(self, Z):
        return 1 / (np.cos(Z) + 0.50572 * (96.07995 - np.rad2deg(Z)) ** (-1.6364))

    def transmission(self, tau, Z):
        return np.exp(-self.X(Z)[:, np.newaxis] * self.config["gamma"] * tau)

    def t_args(self, frame, rays):
        Z = np.pi / 2 - rays.coords.alt.rad
        tau = k_rayleigh(frame) + k_mie(frame)

        return (tau, Z)
