import sys
sys.path.insert(0,'/home/gerritr/ECAP/nsb_simulation/blacksky/')

import numpy as np
import pickle

from astropy.coordinates import SkyCoord
import astropy.units as u
import astropy.constants as c

import blacksky.bandpass as bandpass

from nsb.core import Ray
from nsb.core.emitter import Emitter

class StarCatalog(Emitter):
    def compile(self):
        with open(self.config['catalog_file'], "rb") as input_file:
            self.catalog = pickle.load(input_file)

        v_mag = self.catalog.apply_bandpass(bandpass.OSN_V())
        self.vmask = (v_mag > self.config['magmin']) & (v_mag < self.config['magmax'])

    def emit(self, frame):
        target = frame.target.transform_to('icrs')

        wvl = frame.obswl.to(u.nm).value
        ind, coords, spec = self.catalog.query(wvl, np.asarray([[target.dec.rad, target.ra.rad]]), np.deg2rad(frame.fov))

        coords = SkyCoord(coords[:,1], coords[:,0], unit='rad', frame='icrs').transform_to(frame.AltAz)

        E_p = c.h.value*c.c.value / (wvl*1e-9)
        
        return Ray(coords, weight=spec/E_p, source=type(self), parent=ind, direction='forward')

if __name__ == "__main__":
    None
