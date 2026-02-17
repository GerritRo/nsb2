from __future__ import annotations

from functools import cached_property

import astropy.units as u
import numpy as np
import numpy.lib.recfunctions as recfc
import scipy.integrate as si
from astropy.io import fits, votable
from astropy.utils.data import download_file
from scipy.interpolate import RegularGridInterpolator, UnivariateSpline

SVO_TABLE_URL = "https://svo2.cab.inta-csic.es/theory/fps/fps.php?ID="
CALSPEC_URL = "https://archive.stsci.edu/hlsps/reference-atlases/cdbs/current_calspec/"


class Bandpass:
    def __init__(self, wvl: u.Quantity, transmission: np.ndarray) -> None:
        self.lam = wvl
        self.trx = transmission
        self.min = self.lam.min()
        self.max = self.lam.max()
        self._spline = UnivariateSpline(self.lam, self.trx, s=0, ext=1)

    def __call__(self, lam: u.Quantity) -> np.ndarray:
        return self._spline(lam.to(self.lam.unit))

    @cached_property
    def vegazero(self) -> u.Quantity:
        """Vega zeropoint flux — computed on first access, not at construction."""
        f_down = download_file(CALSPEC_URL + 'alpha_lyr_stis_011.fits', cache=True)
        hdul = fits.open(f_down)
        wvl = hdul[1].data['WAVELENGTH'] * u.angstrom
        flx = hdul[1].data['FLUX'] * u.erg / u.second / u.cm**2 / u.angstrom
        return si.simpson(x=wvl, y=wvl * self(wvl) * flx) * u.erg / u.second / u.cm**2

    @classmethod
    def from_SVO(cls, filter_id: str, cache: bool = True) -> Bandpass:
        f_down = download_file(SVO_TABLE_URL + filter_id, cache=cache)
        table = votable.parse_single_table(f_down)
        return cls(table.array.data['Wavelength'] * u.angstrom, table.array.data['Transmission'])

    @classmethod
    def from_csv(cls, file) -> Bandpass:
        arr = np.genfromtxt(file, delimiter=",", names=True)
        lam = arr['wvl'] * u.nm
        trx = recfc.drop_fields(arr, "wvl", usemask=False)
        return cls(lam, np.array(trx.tolist()).prod(axis=1))


class SpectralGrid:
    """N-dimensional grid of spectra, callable on data coordinates."""

    def __init__(self, points: list, wvl: u.Quantity, flx: u.Quantity) -> None:
        self.points = points
        self.wvl = wvl
        self.flx = flx

    def __call__(self, xi: np.ndarray) -> u.Quantity:
        if xi.size == 0:
            return self.flx
        rgi = RegularGridInterpolator(self.points, self.flx, bounds_error=False)
        return rgi(xi) * self.flx.unit

    def apply_bandpass(self, bandpass: Bandpass) -> SpectralGrid:
        mask = (self.wvl >= bandpass.min) & (self.wvl <= bandpass.max)
        wvl = self.wvl[mask]
        flx = np.einsum("a,...ab->...ab", bandpass(wvl), self.flx[..., mask, :])
        return SpectralGrid(self.points, wvl, flx)

    def integrate(self) -> RateGrid:
        rate = si.simpson(self.flx, x=self.wvl, axis=-2) * self.flx.unit * self.wvl.unit
        return RateGrid(self.points, rate)

    def __mul__(self, value) -> SpectralGrid:
        v = np.asarray(value)
        for _ in range(self.flx.ndim - v.ndim):
            v = v[..., np.newaxis]
        return SpectralGrid(self.points, self.wvl, self.flx * v)


class RateGrid:
    """N-dimensional grid of integrated rates, callable on data coordinates."""

    def __init__(self, points: list, rate: u.Quantity) -> None:
        self.points = points
        self.rate = rate

    def __call__(self, xi: np.ndarray) -> u.Quantity:
        if xi.size == 0:
            return self.rate
        rgi = RegularGridInterpolator(self.points, self.rate, bounds_error=False)
        return rgi(xi) * self.rate.unit

    def __mul__(self, value) -> RateGrid:
        return RateGrid(self.points, np.einsum('b...,ab->ab...', self.rate, value))
