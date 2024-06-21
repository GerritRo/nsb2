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
        catalog = self.config['catalog']

        v_mag = catalog.spectral.apply_bandpass(bandpass.OSN_V())
        vmask = (v_mag > self.config['magmin']) & (v_mag < self.config['magmax'])

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
        spath = '/home/gerritr/ECAP/nsb_simulation/nsb2/nsb/utils/assets/'
        self.m_b = -2.5*np.log10(np.load(spath + 'gaia_m_b_15plus.npy')+1e-31)
        self.m_g = -2.5*np.log10(np.load(spath + 'gaia_m_g_15plus.npy')+1e-31)
        self.NSIDE = hp.npix2nside(len(self.m_g))
        
        self.T = ballesteros(np.clip(self.m_b - self.m_g, -0.45, None),
                                     4261, 0.77, 12.0, 0.445)
        
        self.norm = self._norm_mag_spectrum()    



