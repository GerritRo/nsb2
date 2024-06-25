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

class GaiaDR3(Emitter):
    def compile(self):
        catalog = np.load(self.config['catalog_file'])
        mags = [catalog['phot_g_mean_mag'], catalog['phot_bp_mean_mag'], catalog['phot_rp_mean_mag']]
        bpas = [bandpass.GaiaDR3_G(), bandpass.GaiaDR3_BP(), bandpass.GaiaDR3_RP()]
        
        catalog = StarCatalog.from_photometry(catalog['ra'], catalog['dec'], magnitudes=mags, bandpass=bpas, stis008=True)
        v_mag = catalog.spectral.apply_bandpass(bandpass.OSN_V())
        vmask = (v_mag > self.config['magmin']) & (v_mag <= self.config['magmax'])

        self.catalog = catalog[vmask]
        self.catalog.build_balltree()

    def emit(self, frame):
        target = frame.target.transform_to('icrs')
        wvl = frame.obswl.to(u.nm).value
        ind, coords, spec = self.catalog.query(wvl, np.asarray([[target.dec.rad, target.ra.rad]]), np.deg2rad(frame.fov))

        coords = SkyCoord(coords[:,1], coords[:,0], unit='rad', frame='icrs').transform_to(frame.AltAz)
        E_p = c.h.value*c.c.value / (wvl*1e-9)
        
        return Ray(coords, weight=spec/E_p, source=type(self), parent=ind, direction='forward')

class GaiaDR3Mag15(Diffuse):
    def compile(self):
        mag_map = np.load(self.config['magnitude_maps'])
        mags = [mag_map[0], mag_map[1], mag_map[2]]
        bpas = [bandpass.GaiaDR3_G(), bandpass.GaiaDR3_BP(), bandpass.GaiaDR3_RP()]
        
        self.catalog = StarMap.from_photometry_map(magnitudes=mags, bandpass=bpas, stis008=True)

    def evaluate(self, frame, rays):
        r_g = rays.transform_to('icrs')
        wvl = frame.obswl.to(u.nm).value
        E_p = c.h.value*c.c.value / (wvl*1e-9)
        
        ind, spec = self.catalog.query(wvl, np.vstack([r_g.coords.dec.deg, r_g.coords.ra.deg]).T)
        rays.source = type(self)
        
        return rays*(spec/E_p)





