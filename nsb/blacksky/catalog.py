import numpy as np
import healpy as hp
from itertools import combinations
from sklearn.neighbors import BallTree

from . import ASSETS_PATH
from .photometric import BlackBody, SynthSpec

class StarCatalog():
    def __init__(self, ra, dec, spectral):
        self.coords = np.vstack([np.deg2rad(dec), np.deg2rad(ra)]).T
        self.spectral = spectral

    @classmethod
    def from_photometry(cls, ra, dec, magnitudes=[], bandpass=[], stis008=True, method='blackbody'):
        if method == 'blackbody':
            spectral = BlackBody(magnitudes=magnitudes, bandpass=bandpass, stis008=stis008)
        elif method == 'synthetic':
            spectral = SynthSpec(magnitudes=magnitudes, bandpass=bandpass)

        return cls(ra, dec, spectral)

    def query(self, wvl, coords, radius):
        inds = self.balltree.query_radius(coords, r=radius)
        ind = np.unique(np.concatenate(inds))
        return ind, self.coords[ind], self.spectral.SED(ind, wvl)

    def build_balltree(self):
        self.balltree = BallTree(self.coords, metric='haversine')

    def __add__(self, catalog):
        coords = np.rad2deg(np.vstack((self.coords, catalog.coords)))
        return StarCatalog(coords[:,1], coords[:,0], self.spectral.append(catalog.spectral))

    def __getitem__(self, item):
        i_coords = np.rad2deg(self.coords[item])
        return StarCatalog(i_coords[:,1], i_coords[:,0], self.spectral[item])

class StarMap():
    def __init__(self, spectral):
        self.spectral = spectral
        self.nside = hp.npix2nside(spectral.flux.shape[0])
        self.ncorr = hp.nside2pixarea(self.nside)
        
    @classmethod
    def from_photometry(cls, ra, dec, magnitudes=[], bandpass=[], nside=2**10, stis008=True, method='blackbody'):
        hp_inds = hp.ang2pix(nside, ra, dec, nest=True, lonlat=True)
        m_map = [cls.create_mag_map(hp_inds, m, nside) for m in cls.replace_nan(magnitudes)]
        
        if method == 'blackbody':
            spectral = BlackBody(magnitudes=m_map, bandpass=bandpass, stis008=stis008)
        else:
            raise NotImplementedError

        return cls(spectral)

    @classmethod
    def from_photometry_map(cls, magnitudes=[], bandpass=[], stis008=False, method='blackbody'):
        if method == 'blackbody':
            spectral = BlackBody(magnitudes=magnitudes, bandpass=bandpass, stis008=stis008)
        return cls(spectral)

    def query(self, wvl, coords):
        ind = hp.ang2pix(self.nside, coords[:,1], coords[:,0], nest=True, lonlat=True)
        return ind, self.spectral.SED(ind, wvl)/self.ncorr

    @staticmethod
    def replace_nan(magnitudes):
        for j in combinations(range(len(magnitudes)), 2):
            m1, m2 = magnitudes[j[0]], magnitudes[j[1]]
            mag_diff = np.nanmean(m1-m2)
            magnitudes[j[0]] = np.where(np.isfinite(m1), m1, m2+mag_diff)
            magnitudes[j[1]] = np.where(np.isfinite(m2), m2, m1-mag_diff)
        magnitudes = [np.nan_to_num(m, nan=np.nanmax(m)) for m in magnitudes] 
        return magnitudes

    @staticmethod
    def create_mag_map(hp_inds, m, nside):
        return -2.5*np.log10(np.bincount(hp_inds, 10**(-0.4*m), hp.nside2npix(nside)))

    def __add__(self, starmap):
        return StarMap(self.spectral+starmap.spectral)