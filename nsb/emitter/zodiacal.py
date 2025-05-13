from .. import ASSETS_PATH
from nsb.core.emitter import Diffuse
import nsb.utils.spectra as spectra

import numpy as np
import astropy
import astropy.constants as c
import astropy.units as u
from scipy.interpolate import RegularGridInterpolator


class Masana2021(Diffuse):
    """
    Zodiacal light following the method from Masana et al. (2021), which uses data
    from Leinert et al. (1998), which summarizes a table from Dumont & Sanchez (1976).
    """

    def compile(self):
        zod = np.genfromtxt(ASSETS_PATH + "zodiacal_leinert1998.dat", delimiter=",")
        fit_points = [zod[1:, 0], zod[0, 1:]]
        self.A = RegularGridInterpolator(points=fit_points, values=zod[1:, 1:])
        self.spectrum = spectra.SolarSpectrumRieke2008()

    def SPF(self, lam):
        E_p = c.h.value * c.c.value / lam * 1e9
        return self.spectrum(lam) / self.spectrum(500) / E_p 

    def color_corr(self, lam, elon):
        elon_f = -0.3 * (np.clip(elon, 30, 90) - 30) / 60
        return 1 + (1.2 + elon_f[:, np.newaxis]) * np.log(lam / 500)

    def evaluate(self, frame, rays):
        r_e = rays.transform_to("geocentrictrueecliptic")

        sun = astropy.coordinates.get_body("sun", frame.time)
        s_e = sun.transform_to("geocentrictrueecliptic")

        alpha = np.abs((r_e.coords.lon - s_e.lon).wrap_at(180 * u.deg).deg)
        beta = np.abs(r_e.coords.lat.deg)

        corr = self.color_corr(frame.obswl.to(u.nm).value, alpha)

        rays.source = type(self)
        weight = (
            1e-11
            * self.A(np.asarray([alpha, beta]).T)[:, np.newaxis]
            * corr
            * self.SPF(frame.obswl.to(u.nm).value)
        )
        return rays * weight
