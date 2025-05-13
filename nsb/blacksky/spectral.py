from itertools import combinations, permutations
import os.path

import numpy as np
from scipy.spatial import KDTree
from scipy.interpolate import interp1d, UnivariateSpline, RectBivariateSpline
import scipy.integrate as si
import astropy.units as u

from . import ASSETS_PATH
from .formulas import blackbody
from .constants import T_vega, F_vega, sigma, h, c
from .utils import Vega_STIS008

class BlackBody():
    def __init__(self, flux=None, teff=None, magnitudes=[], bandpass=[], stis008=False):
        self.stis008 = stis008
        self.vegazero = [self._bpxvg(bp) for bp in bandpass]
        self.response = [self._bpxbb(bp) for bp in bandpass]

        self.flux = flux
        self.teff = teff
        
        if not isinstance(self.teff, np.ndarray):
            self.teff = np.full(magnitudes[0].shape, np.nan)
        if not isinstance(self.flux, np.ndarray):
            self.flux = np.full(magnitudes[0].shape, np.nan)

        # Determine the temperature for all stars where teff is missing and magnitudes exist
        for j in combinations(range(len(magnitudes)), 2):
            color = magnitudes[j[0]] - magnitudes[j[1]]
            cmask = np.isnan(self.teff) & np.isfinite(color)
            self.teff[cmask] = self._teff_from_color(color[cmask], [self.response[i] for i in j], [self.vegazero[i] for i in j])

        # Set the temperature for undetermined stars to the mean
        mask10 = np.isnan(self.teff)
        if np.sum(mask10) != 0:
            self.teff[mask10] = np.nanmean(self.teff)

        for j in range(len(magnitudes)):
            tmask = np.isnan(self.flux) & np.isfinite(self.teff) & np.isfinite(magnitudes[j])
            self.flux[tmask] = self._flux_from_teff_mag(self.teff[tmask], magnitudes[j][tmask], self.response[j], self.vegazero[j])
    
    def SED(self, ind, wvl):
        return self.flux[ind][:, np.newaxis] * blackbody(1e-9*wvl, self.teff[ind][:, np.newaxis])

    def apply_bandpass(self, bandpass):
        vega_ref = self._bpxvg(bandpass)
        response = self._bpxbb(bandpass)
        
        return -2.5*np.log10(self.flux * response(self.teff) / vega_ref)
        
    def _teff_from_color(self, color, response, zero_point):
        # Determine InvertSpline
        t_arr = np.logspace(3, 6, 500)
        ratio = np.log10(response[1](t_arr)/response[0](t_arr))
        i_spline = interp1d(ratio, t_arr, bounds_error=False)
        # Determine the Zero Point using VegaMag
        col_zero = np.log10(zero_point[1]/zero_point[0])
        return i_spline(np.asarray(color)/2.5 + col_zero)

    def _flux_from_teff_mag(self, teff, mag, response, zero_point):
        return 10**(-0.4*mag) * zero_point/response(teff)

    def _bpxbb(self, bandpass):
        x = np.linspace(bandpass.min, bandpass.max, 2**10+1)
        y = np.logspace(3, 6, 500)
    
        z = x[:,np.newaxis]*bandpass(x[:,np.newaxis])*blackbody(1e-9*x[:,np.newaxis], y)
        r = si.romb(z, dx=x[1]-x[0], axis=0)
        
        return UnivariateSpline(y, r, s=0, ext=1, k=1)

    def append(self, blackbody):
        return BlackBody(flux = np.append(self.flux, blackbody.flux), teff=np.append(self.teff, blackbody.teff))

    def __add__(self, blackbody):
        w1 = np.nan_to_num(self.flux*self.teff*self.teff*self.teff*self.teff)
        w2 = np.nan_to_num(blackbody.flux*blackbody.teff*blackbody.teff*blackbody.teff*blackbody.teff)
        nt = (self.teff*w1 + blackbody.teff*w2)/(w1+w2)

        return BlackBody(flux=(w1+w2)/(nt*nt*nt*nt), teff=nt)

    def __getitem__(self, item):
        return BlackBody(flux = self.flux[item], teff=self.teff[item])

class SynthSpec():
    def __init__(self, flux=None, ind=None, ebv=None, dust_extinction=None, magnitudes=[], bandpass=[]):
        if not os.path.isfile(ASSETS_PATH + 'coelho2014.npy'):
            raise FileNotFoundError('Coelho2014 catalog file not found, download via blacksky.utils.download_coelho2014')
        coelho_arr = np.load(ASSETS_PATH + 'coelho2014.npy')
        self.coelho_arr = coelho_arr[1:]
        self.coelho_wvl = coelho_arr[0]/10
        self.dust_extinction = dust_extinction
        self.d_int = interp1d(self.coelho_wvl, self.coelho_arr, axis=-1)
        
        self.vegazero = [self._bpxvg(bp) for bp in bandpass]

        self.flux = flux
        self.ind  = ind
        self.ebv  = ebv

        if not isinstance(self.flux, np.ndarray):
            self.flux = np.full(magnitudes[0].shape, np.nan)
        if not isinstance(self.ind, np.ndarray):
            self.ind = np.full(magnitudes[0].shape, np.nan)
        if not isinstance(self.ebv, np.ndarray):
            self.ebv = np.full(magnitudes[0].shape, 0)

        # Determine best fitting spectrum including ebv contribution
        for j in combinations(range(len(magnitudes)), 2):
            color = magnitudes[j[0]] - magnitudes[j[1]]
            cmask = np.isfinite(color)&np.isnan(self.flux)&np.isnan(self.ind)
            if cmask.sum() != 0:
                res = self._calculate_kdtree(color[cmask], self.ebv[cmask], [bandpass[j[0]], bandpass[j[1]]])
                self.ind[cmask], self.flux[cmask], self.ebv[cmask] = res[0], 10**(-0.4*magnitudes[j[0]][cmask])*res[1], res[2]

        # For Stars with no dual magnitude data, use average color difference
        for j in permutations(range(len(magnitudes)), 2):
            avg_c = np.nanmean(magnitudes[j[0]] - magnitudes[j[1]])
            cmask = np.isnan(self.flux)&np.isnan(self.ind)&np.isfinite(magnitudes[j[0]])
            if cmask.sum() != 0 and np.isfinite(avg_c):
                res = self._calculate_kdtree(np.full((cmask.sum(),), avg_c), self.ebv[cmask], [bandpass[j[0]], bandpass[j[1]]])
                self.ind[cmask], self.flux[cmask], self.ebv[cmask] = res[0], 10**(-0.4*magnitudes[j[0]][cmask])*res[1], res[2]

        self.ind = self.ind.astype(int)

    def SED(self, ind, wvl):
        if self.dust_extinction != None:
            d_ext = self.dust_extinction.extinguish(wvl*u.nm, Ebv=self.ebv[ind][:,np.newaxis])
        else:
            d_ext = 1
        return self.flux[ind][:,np.newaxis]*self.d_int(wvl)[self.ind[ind]]*d_ext

    def apply_bandpass(self, bandpass):
        r = self.SED(np.arange(self.flux.shape[0]), self.coelho_wvl)
        mnorm = self._bpxcs(bandpass, r)
        vzero = self._bpxvg(bandpass)

        return -2.5*np.log10(mnorm/vzero)

    def _calculate_kdtree(self, bcolor, bebv, passbands, eps=1e-2, N=100):
        if self.dust_extinction != None:
            ars = []
            for j in range(N):
                x = np.clip(np.random.lognormal(0.000655, 2.0515, 3727), 0, 100)
                r = self.coelho_arr*self.dust_extinction.extinguish(self.coelho_wvl*u.nm, Ebv=x[:,np.newaxis])
                mnorm = [self._bpxcs(bp, r) for bp in passbands]
                vzero = [self._bpxvg(bp) for bp in passbands]
                color = 2.5*np.log10(mnorm[1]/mnorm[0]*vzero[0]/vzero[1])
                ars.append(np.asarray([np.arange(3727), color, x, vzero[0]/mnorm[0]]))
    
            ars = np.hstack(ars)
            coords = np.vstack([ars[1], eps*ars[2]]).T
            kdtree = KDTree(coords)
            dist, ind = kdtree.query(np.vstack([bcolor, eps*bebv]).T)
    
            return ars[0][ind.flatten()], ars[3][ind.flatten()], ars[2][ind.flatten()]
        else:
            ars = []
            mnorm = [self._bpxcs(bp, self.coelho_arr) for bp in passbands]
            vzero = [self._bpxvg(bp) for bp in passbands]
            color = 2.5*np.log10(mnorm[1]/mnorm[0]*vzero[0]/vzero[1])
            ars = np.vstack([np.arange(3727), color, np.zeros(3727), vzero[0]/mnorm[0]])
    
            coords = np.atleast_2d(ars[1]).T
            kdtree = KDTree(coords)
            dist, ind = kdtree.query(np.vstack([bcolor]).T)
    
            return ars[0][ind.flatten()], ars[3][ind.flatten()], ars[2][ind.flatten()]
            
    def _bpxcs(self, bandpass, spectra):
        BP_x = bandpass(self.coelho_wvl)*self.coelho_wvl
        return si.simpson(y = spectra*BP_x, x = self.coelho_wvl)

    def append(self, synthspec):
        return SynthSpec(flux = np.append(self.flux, synthspec.flux),
                         ind = np.append(self.ind, synthspec.ind),
                         ebv = np.append(self.ebv, synthspec.ebv))

    def __getitem__(self, item):
        return SynthSpec(flux = self.flux[item], ind=self.ind[item], ebv=self.ebv[item])