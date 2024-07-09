import sys
sys.path.insert(0,'/home/gerritr/ECAP/nsb_simulation/blacksky/')

import numpy as np
import pickle

from astropy.coordinates import SkyCoord
import astropy.units as u
import astropy.constants as c

import blacksky.bandpass as bandpass
from blacksky.catalog import StarCatalog, StarMap

from nsb.core import Ray
from nsb.core.emitter import Emitter, Diffuse

class GenericStarCatalog(Emitter):
    def emit(self, frame):
        target = frame.target.transform_to('icrs')
        wvl = frame.obswl.to(u.nm).value
        E_p = c.h.value*c.c.value / (wvl*1e-9)
        
        ind, coords, spec = self.catalog.query(wvl, np.asarray([[target.dec.rad, target.ra.rad]]), np.deg2rad(frame.fov))
        coords = SkyCoord(coords[:,1], coords[:,0], unit='rad', frame='icrs').transform_to(frame.AltAz)
        
        return Ray(coords, weight=spec/E_p, source=type(self), parent=ind, direction='forward')

class GenericStarMap(Diffuse):
    def evaluate(self, frame, rays):
        r_g = rays.transform_to('icrs')
        wvl = frame.obswl.to(u.nm).value
        E_p = c.h.value*c.c.value / (wvl*1e-9)
        
        ind, spec = self.catalog.query(wvl, np.vstack([r_g.coords.dec.deg, r_g.coords.ra.deg]).T)
        rays.source = type(self)
        
        return rays*(spec/E_p)

class GaiaDR3(GenericStarCatalog):
    def compile(self):
        # Loading GaiaDR3 main catalog file
        main_catalog = np.load(self.config['gaia_file'])
        mags = [main_catalog['phot_g_mean_mag'], main_catalog['phot_bp_mean_mag'], main_catalog['phot_rp_mean_mag']]
        bpas = [bandpass.GaiaDR3_G(), bandpass.GaiaDR3_BP(), bandpass.GaiaDR3_RP()]
        
        gaia = StarCatalog.from_photometry(main_catalog['ra'], main_catalog['dec'],
                                           magnitudes=mags, bandpass=bpas, stis008=True)
        # Loading GaiaDR3 Tycho supplementary catalog
        supp_catalog = np.load(self.config['supp_file'])
        mags = [supp_catalog['BTmag'], supp_catalog['VTmag'], supp_catalog['Vmag'], supp_catalog['Bmag'], supp_catalog['Hmag']]
        bpas = [bandpass.Tycho_B(), bandpass.Tycho_V(), bandpass.OSN_V(), bandpass.OSN_B(), bandpass.Hipp_M()]
        
        supp = StarCatalog.from_photometry(supp_catalog['RA_ICRS_'], supp_catalog['DE_ICRS_'],
                                           magnitudes=mags, bandpass=bpas, stis008=True)
        # Combining catalogs
        catalog = gaia+supp
        # Filtering by visual magnitude
        v_mag = catalog.spectral.apply_bandpass(bandpass.OSN_V())
        vmask = (v_mag > self.config['magmin']) & (v_mag <= self.config['magmax'])

        self.catalog = catalog[vmask]
        self.catalog.build_balltree()

class GaiaDR3Mag15(GenericStarMap):
    def compile(self):
        mag_map = np.load(self.config['magnitude_maps'])
        mags = [mag_map[0], mag_map[1], mag_map[2]]
        bpas = [bandpass.GaiaDR3_G(), bandpass.GaiaDR3_BP(), bandpass.GaiaDR3_RP()]
        
        self.catalog = StarMap.from_photometry_map(magnitudes=mags, bandpass=bpas, stis008=True)





