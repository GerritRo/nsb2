import numpy as np
import astropy.units as u

from nsb.core.emitter import Diffuse
import nsb.utils.spectra as spectra


class Noll2012(Diffuse):
    """
    Diffuse Airglow emission modelled from Noll et. al (2012)
    """

    def compile(self):
        """
        Loads the airglow spectrum
        """
        self.ag_spectra = spectra.eso_airglow()

    def SPF(self, lam):
        """
        Description

        Parameters
        ----------
        lam : _type_
            _description_

        Returns
        -------
        _type_
            _description_
        """
        return self.ag_spectra(lam)

    def vanrhjin(self, z):
        """
        Van Rhjin function for increase in brightness based on intersection with airglow layer

        Parameters
        ----------
        z : floatlike
            zenith

        Returns
        -------
        _type_
            _description_
        """
        r_rh = 6738 / (6738 + self.config["H"])
        return 1 / (1 - r_rh**2 * np.sin(z) ** 2) ** 0.5

    def evaluate(self, frame, rays):
        """
        Brightness model based on the van Rhjin function for airglow and the solar flux scaling from Noll et. al (2012)
        """
        rays.source = type(self)
        weight = self.vanrhjin(np.pi / 2 - rays.coords.alt.rad)[
            :, np.newaxis
        ] * self.SPF(frame.obswl.to(u.nm).value)

        sf_scale = 0.2 + 0.00614 * frame.conf["sfu"]
        return rays * weight * sf_scale
