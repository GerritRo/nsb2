import logging

import numpy as np
from abc import abstractmethod

from astropy.coordinates import SkyCoord
import astropy.units as u
import astropy.constants as c
from sklearn.neighbors import BallTree

from scipy.interpolate import UnivariateSpline
import scipy.integrate as integrate

from nsb.core import Ray
from nsb.core.emitter import Emitter
from nsb.utils.formulas import blackbody, ballesteros
from nsb.utils import bandpass

from astroquery.utils.tap.core import TapPlus
vizier = TapPlus(url="http://tapvizier.u-strasbg.fr/TAPVizieR/tap")

import requests
import pyvo

service_name = "Gaia@AIP"
url = "https://gaia.aip.de/tap"
token = 'd233b34a76deb2face168fa20300a3d67d01a32f'
tap_session = requests.Session()
tap_session.headers['Authorization'] = token

tap_service = pyvo.dal.TAPService(url, session=tap_session)

logger = logging.getLogger(__name__)

class StarCatalog(Emitter):       
    def compile(self):
        self.mag     = [self.config['magmin'], self.config['magmax']]
        self.path    = self.config['cache_path'] + self.__class__.__name__ + ".npy"

        try:
            self.catalog = self.__load_from_cache()
        except FileNotFoundError as e:
            logger.info("No cached data at {}, querying from Catalog".format(self.path))
            self.catalog = self._database_query(self.mag)

        self._update_catalog()
        if self.config['cache']:
            self.__save_to_cache()

        self.norm = self._norm_mag_spectrum()
        self.catalog = self._filter_catalog()
        self._preprocess()
        
        self.balltree = self._generate_tree()
        self.StarCoords = self._to_SkyCoord()
        
    def SPF(self, lam, T):
        E_p = c.h.value*c.c.value / lam
        return blackbody(lam, T[:,np.newaxis]) / self.norm(T[:,np.newaxis]) / E_p

    def emit(self, frame):
        target = frame.target.transform_to('icrs')
        ind = self.balltree.query_radius(np.asarray([[target.dec.rad, target.ra.rad]]),
                                         r=np.deg2rad(frame.fov))
        coords = self.StarCoords[ind[0]].transform_to(frame.AltAz)
        
        F = self.calc_flux(ind[0], frame.obswl.to(u.m).value)
        
        return Ray(coords, 
                   weight=F, 
                   source=type(self), 
                   parent=ind[0],
                   direction='forward')
    
    def calc_flux(self, ind, lam):
        T = self.catalog[self._cat_dictionary('Teff')][ind]
        M = self.catalog[self._cat_dictionary('mag')][ind]
        
        return 10**(-0.4*M[:,np.newaxis])*self.config['Mag_0'] * self.SPF(lam, T) 
    
    @abstractmethod
    def _norm_mag_spectrum(self):
        return NotImplementedError

    def __load_from_cache(self):
        return np.load(self.path)

    def __save_to_cache(self):
        np.save(self.path, self.catalog)

    def _update_catalog(self):
        max_mag = np.max(self.catalog[self._cat_dictionary('mag')])
        while max_mag < self.mag[-1]:
            print('Achieved max Mag lower than expected {:.2f} < {}'.format(max_mag, self.mag[-1]))
            new_add = self._database_query([max_mag, self.mag[-1]])
            if len(new_add) == 0:
                break
            self.catalog = np.append(self.catalog, new_add)
            max_mag = np.max(self.catalog[self._cat_dictionary('mag')])

    def _filter_catalog(self):
        mask = (self.catalog[self._cat_dictionary('mag')]>=self.mag[0])&(self.catalog[self._cat_dictionary('mag')]<=self.mag[1])
        return self.catalog[mask]

    def _to_SkyCoord(self):
        '''
        Conversion to skycoords from Catalog
        '''
        ra, dec = [self._cat_dictionary(c) for c in ['ra','dec']]
        return SkyCoord(self.catalog[ra], self.catalog[dec], unit='deg')

    def _generate_tree(self):
        ra, dec = [self._cat_dictionary(c) for c in ['ra','dec']]
        coords  = np.vstack([np.deg2rad(self.catalog[dec]),
                             np.deg2rad(self.catalog[ra])]).T
        return BallTree(coords, metric='haversine', leaf_size=1000)

    @abstractmethod
    def _cat_dictionary(self, entry):
        return NotImplementedError
    
    @abstractmethod
    def _database_query(self, mag):
        return NotImplementedError
    
    @abstractmethod
    def _preprocess(self):
        return NotImplementedError

    
class GaiaDR3(StarCatalog):
    def _cat_dictionary(self, entry):
        cat_dict = {'ra':'ra', 'dec':'dec', 'mag':'phot_g_mean_mag', 'Teff':'teff_gspphot'}
        return cat_dict[entry]
    
    def _preprocess(self):
        T = self.catalog[self._cat_dictionary('Teff')]
        b_g  = self.catalog['phot_bp_mean_mag'] - self.catalog['phot_g_mean_mag']
        
        mask = ~np.isnan(T)

        b = lambda d: ballesteros(np.clip(d, -0.45, None), 4261, 0.77, 12.0, 0.445)
        
        self.catalog[self._cat_dictionary('Teff')] = np.nan_to_num(np.where(mask, T, b(b_g)), nan=6000)
        
    def _norm_mag_spectrum(self):
        x = np.logspace(2, 5, 200)
        y = []
        s = bandpass.GaiaDR3()        
        
        for T_eff in x:
            y.append(integrate.quad(lambda lam: s(lam)*blackbody(1e-9*lam, T_eff), 250, 1100, limit=150)[0])
            
        return UnivariateSpline(x, y, s=0, ext=1, k=1)

    def _database_query(self, mag):
        try:
            query = ('SELECT lite.ra, lite.dec, lite.teff_gspphot, lite.phot_g_mean_mag, sp.b_jkc_mag, sp.v_jkc_mag  '
                    'FROM gaiadr3.gaia_source_lite AS lite '
                    'INNER JOIN gaiadr3.synthetic_photometry_gspc AS sp USING (source_id) '
                    'lite.phot_g_mean_mag  > {} '
                    'AND '
                    'lite.phot_g_mean_mag  <= {} ORDER BY sp.v_jkc_mag').format(mag[0], mag[1])

            job = tap_service.submit_job(query, language='ADQL', runid='gaia_mag', queue="5m", maxrec=300000)
            job.run()
            job.wait(phases=["COMPLETED", "ERROR", "ABORTED"], timeout=300.0)
            job.raise_if_error()
            results = job.fetch_result().to_table()
        except IOError as e:
            raise e

        return results


class Hipparcos(StarCatalog):
    def _cat_dictionary(self, entry):
        cat_dict = {'ra':'RAICRS', 'dec':'DEICRS', 'mag':'Vmag', 'Teff':'Teff'}
        return cat_dict[entry]

    def _preprocess(self):
        T = self.catalog[self._cat_dictionary('Teff')]
        b_v = self.catalog['B-V']
        
        mask = (~np.isnan(T)) & (~(T==0))
        
        b = lambda d: ballesteros(d, 4600, 0.92, 1.7, 0.62)
        
        self.catalog[self._cat_dictionary('Teff')] = np.nan_to_num(np.where(mask, T, b(b_v)), nan=6000)
        self.catalog = self.catalog[~np.isnan(self.catalog['RAICRS'])]
        
    def _norm_mag_spectrum(self):
        x = np.logspace(2, 5, 200)
        y = []
        s = bandpass.JohnsonV()        
        
        for T_eff in x:
            y.append(integrate.quad(lambda lam: s(lam)*blackbody(1e-9*lam, T_eff), 450, 750, limit=150)[0])
            
        return UnivariateSpline(x, y, s=0, ext=1, k=1)

    def _database_query(self, mag):
        try:
            query = ('SELECT RAICRS, DEICRS, Teff, Vmag, "B-V" '
                     'FROM "J/MNRAS/471/770/table2" AS sp '
                     'FULL JOIN "I/239/hip_main" AS hip USING (HIP) '
                     'WHERE Vmag > {} '
                     'AND '
                     'Vmag <= {} ORDER BY Vmag').format(mag[0], mag[1])
            job     = vizier.launch_job_async(query)
            return job.get_results()
        except IOError as e:
            raise e

if __name__ == "__main__":
    None
