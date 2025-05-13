import numpy as np

from astropy.coordinates import SkyCoord
import astropy.units as u
import astropy.constants as c

import nsb.blacksky.bandpass as bandpass
from nsb.blacksky.catalog import StarCatalog, StarMap

from nsb.core import Ray
from nsb.core.emitter import Emitter, Diffuse


class GenericStarCatalog(Emitter):
    """
    Generic Star Catalog object which emits light based on a catalog query from blacksky.
    """

    def emit(self, frame):
        wvl = frame.obswl.to(u.nm).value
        E_p = c.h.value * c.c.value / (wvl * 1e-9)

        q_coords = np.asarray(
            [frame.pix_coord.icrs.dec.rad, frame.pix_coord.icrs.ra.rad]
        ).T

        ind, coords, spec = self.catalog.query(wvl, q_coords, frame.pix_radii)
        coords = SkyCoord(
            coords[:, 1], coords[:, 0], unit="rad", frame="icrs"
        ).transform_to(frame.AltAz)

        return Ray(
            coords,
            weight=spec / E_p,
            source=type(self),
            parent=ind,
            direction="forward",
        )


class GenericStarMap(Diffuse):
    """
    Generic Star Map object which allows querying a star map (brightness of stars added into one pixel)
    """

    def evaluate(self, frame, rays):
        r_g = rays.transform_to("icrs")
        wvl = frame.obswl.to(u.nm).value
        E_p = c.h.value * c.c.value / (wvl * 1e-9)

        ind, spec = self.catalog.query(
            wvl, np.vstack([r_g.coords.dec.deg, r_g.coords.ra.deg]).T
        )
        rays.source = type(self)

        return rays * (spec / E_p)


class GaiaDR3(GenericStarCatalog):
    """
    GaiaDR3 catalog generated from a gaia catalog file.
    """

    def compile(self):
        # Loading GaiaDR3 main catalog file
        main_catalog = np.load(self.config["gaia_file"])
        mags = [
            main_catalog["phot_g_mean_mag"],
            main_catalog["phot_bp_mean_mag"],
            main_catalog["phot_rp_mean_mag"],
        ]
        bpas = [bandpass.GaiaDR3_G(), bandpass.GaiaDR3_BP(), bandpass.GaiaDR3_RP()]

        gaia = StarCatalog.from_photometry(
            main_catalog["ra"],
            main_catalog["dec"],
            magnitudes=mags,
            bandpass=bpas,
            stis008=True,
            method=self.config["method"],
        )
        # Loading GaiaDR3 Hipparcos supplementary catalog
        supp_catalog = np.load(self.config["supp_file"])
        mags = [supp_catalog["Vmag"], supp_catalog["Bmag"], supp_catalog["Hpmag"]]
        bpas = [bandpass.OSN_V(), bandpass.OSN_B(), bandpass.Hipp_M()]

        supp = StarCatalog.from_photometry(
            supp_catalog["RAJ2000"],
            supp_catalog["DEJ2000"],
            magnitudes=mags,
            bandpass=bpas,
            stis008=True,
            method=self.config["method"],
        )
        # Combining catalogs
        self.catalog = gaia + supp
        # Building tree
        self.catalog.build_balltree()


class GaiaDR3Mag15(GenericStarMap):
    """
    Gaia DR3 star map for stars with G>15
    """

    def compile(self):
        mag_map = np.load(self.config["magnitude_maps"])
        mags = [mag_map[0], mag_map[1], mag_map[2]]
        bpas = [bandpass.GaiaDR3_G(), bandpass.GaiaDR3_BP(), bandpass.GaiaDR3_RP()]

        self.catalog = StarMap.from_photometry_map(
            magnitudes=mags, bandpass=bpas, stis008=True
        )
