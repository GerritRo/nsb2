import numpy as np
import healpy as hp
from itertools import combinations

from .assigner import assign_from_string
from .spatial import method_from_string
from .spectral import spectra_from_string

class StarCatalog():
    def __init__(self, spatial, spectra):
        self.spatial = spatial
        self.spectra = spectra

    @classmethod
    def from_dataframe(cls, dataframe, obstime, method='kdtree',
                       spec_library='blackbody', assign_spectra='nearest'):
        raise NotImplementedError
    
    @classmethod
    def from_parameters(cls, coords, parameters, method='kdtree',
                        spec_library='blackbody', assign_spectra='nearest'):
        raise NotImplementedError
    
    @classmethod
    def from_magnitudes(cls, coords, magnitudes, parameters, method='kdtree',
                        spec_library='blackbody', assign_spectra='nearest'):
        '''
        Create a spectral star catalog based on magnitudes + spectral parameters.
        '''
        # Get methods from strings if not provided and object
        if isinstance(method, str):
            method = method_from_string(method)
        if isinstance(spec_library, str):
            spec_library = spectra_from_string(spec_library)
        if isinstance(assign_spectra, str):
            assign_spectra = assign_from_string(assign_spectra)

        # Assign the spatial object
        spatial = method(coords)

        # Assign the spectra
        spectra = assign_spectra(spatial, spec_library, magnitudes, parameters)

        return cls(spatial, spectral)

    def query(self, wvls, coords, radius):
        ind = self.spatial.query(coords, r=radius)
        return ind, self.spatial.coords[ind], self.spectral.SED(ind, wvls)

    def __add__(self, catalog):
        return StarCatalog(self.spatial.append(catalog.spatial), self.spectral.append(catalog.spectral))

    def __getitem__(self, item):
        return StarCatalog(self.spatial[item], self.spectral[item])

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