from __future__ import annotations

import logging
from functools import cached_property
from urllib.error import HTTPError

import astropy.units as u
import numpy as np
import numpy.lib.recfunctions as recfc
from astropy.io import fits, votable
from astropy.utils.data import download_file
from scipy.integrate import simpson
from scipy.interpolate import RegularGridInterpolator, UnivariateSpline

__all__ = [
    "Bandpass",
    "RateGrid",
    "SpectralGrid",
    "integrate_wavelength",
]


logger = logging.getLogger(__name__)

SVO_TABLE_URL = "https://svo2.cab.inta-csic.es/theory/fps/fps.php?ID="
CALSPEC_URL = "https://archive.stsci.edu/hlsps/reference-atlases/cdbs/current_calspec/"

VEGA_CALSPEC_FILE = "alpha_lyr_stis_012.fits"


def integrate_wavelength(
    flux: u.Quantity, wavelength: u.Quantity, axis: int = -1
) -> u.Quantity:
    """Integrate a spectral quantity over wavelength.

    Composite Simpson's rule is applied to the plain numerical values, and the
    unit of the result is reconstructed as ``flux.unit * wavelength.unit``.

    Parameters
    ----------
    flux : astropy.units.Quantity
        Spectral quantity to integrate.  The wavelength axis is given by
        ``axis``; all other axes are integrated independently.
    wavelength : astropy.units.Quantity
        Wavelength grid, of the same length as ``flux`` along ``axis``.  Need
        not be equidistant, but must be strictly monotonic.
    axis : int, optional
        Axis of ``flux`` holding the wavelength samples.  Default is the last
        axis.

    Returns
    -------
    astropy.units.Quantity
        Integral of ``flux`` over wavelength, with ``axis`` removed and unit
        ``flux.unit * wavelength.unit``.

    Raises
    ------
    TypeError
        If either argument is not a `~astropy.units.Quantity`.

    Examples
    --------
    >>> import numpy as np, astropy.units as u
    >>> wvl = np.linspace(400, 500, 11) * u.nm
    >>> flx = np.ones(11) * u.erg / u.s / u.cm**2 / u.nm
    >>> integrate_wavelength(flx, wvl)  # doctest: +FLOAT_CMP
    <Quantity 100. erg / (s cm2)>
    """
    if not isinstance(flux, u.Quantity):
        raise TypeError("`flux` must be an astropy Quantity")
    if not isinstance(wavelength, u.Quantity):
        raise TypeError("`wavelength` must be an astropy Quantity")

    integral = simpson(flux.value, x=wavelength.value, axis=axis)
    return integral * (flux.unit * wavelength.unit)


class Bandpass:
    """Wavelength-dependent transmission of an optical passband.

    The transmission curve is interpolated with an interpolating cubic
    spline that evaluates to zero outside the tabulated range, so a
    `Bandpass` can be called on any wavelength grid.

    Parameters
    ----------
    wvl : astropy.units.Quantity
        Wavelength grid of the transmission curve, shape ``(W,)``.
    transmission : numpy.ndarray
        Dimensionless transmission at each wavelength, shape ``(W,)``.

    Attributes
    ----------
    lam : astropy.units.Quantity
        The tabulated wavelength grid.
    trx : numpy.ndarray
        The tabulated transmission values.
    min, max : astropy.units.Quantity
        Lower and upper edge of the tabulated wavelength range.

    Examples
    --------
    >>> import numpy as np, astropy.units as u
    >>> bp = Bandpass(np.linspace(400, 600, 5) * u.nm, np.ones(5))
    >>> float(bp(np.array([500]) * u.nm)[0])  # doctest: +FLOAT_CMP
    1.0
    """

    def __init__(self, wvl: u.Quantity, transmission: np.ndarray) -> None:
        self.lam = wvl
        self.trx = transmission
        self.min = self.lam.min()
        self.max = self.lam.max()
        self._spline = UnivariateSpline(self.lam, self.trx, s=0, ext=1)

    def __call__(self, lam: u.Quantity) -> np.ndarray:
        """Evaluate the transmission at the given wavelengths.

        Parameters
        ----------
        lam : astropy.units.Quantity
            Wavelengths at which to evaluate the transmission.  Converted to
            the unit of the tabulated grid before evaluation.

        Returns
        -------
        numpy.ndarray
            Dimensionless transmission, zero outside the tabulated range.
        """
        return self._spline(lam.to(self.lam.unit))

    @cached_property
    def vegazero(self) -> u.Quantity:
        """astropy.units.Quantity: Vega zeropoint of this passband.

        The band-integrated photon-weighted flux of Vega, computed from the
        CALSPEC reference spectrum ``alpha_lyr_stis_012`` [Bohlin2014]_.  It
        is the denominator of the Vega magnitude system, so a source with
        ``vegazero``-equal integrated flux has magnitude zero in this band.

        Raises
        ------
        RuntimeError
            If the pinned reference spectrum is no longer available from
            STScI.  See :data:`VEGA_CALSPEC_FILE`.
        """
        logger.debug("downloading CALSPEC Vega reference spectrum")
        try:
            path = download_file(CALSPEC_URL + VEGA_CALSPEC_FILE, cache=True)
        except HTTPError as err:
            if err.code != 404:
                raise
            raise RuntimeError(
                f"The CALSPEC reference spectrum {VEGA_CALSPEC_FILE!r} is no "
                f"longer available at {CALSPEC_URL}. STScI retires superseded "
                f"revisions; set nsb2.core.spectral.VEGA_CALSPEC_FILE to a "
                f"revision that is still published."
            ) from err
        with fits.open(path) as hdul:
            wvl = hdul[1].data["WAVELENGTH"] * u.angstrom
            flx = hdul[1].data["FLUX"] * u.erg / u.second / u.cm**2 / u.angstrom
        return integrate_wavelength(wvl * self(wvl) * flx, wvl)

    @classmethod
    def from_SVO(cls, filter_id: str, cache: bool = True) -> Bandpass:
        """Construct a `Bandpass` from the SVO Filter Profile Service.

        Parameters
        ----------
        filter_id : str
            SVO filter identifier, e.g. ``"GAIA/GAIA3.G"``.
        cache : bool, optional
            Whether to cache the downloaded profile in the astropy download
            cache.  Default is `True`.

        Returns
        -------
        Bandpass
            The requested passband.

        Notes
        -----
        Requires network access on the first call for a given filter.
        """
        logger.debug("downloading SVO filter profile %s", filter_id)
        path = download_file(SVO_TABLE_URL + filter_id, cache=cache)
        table = votable.parse_single_table(path)
        return cls(
            table.array.data["Wavelength"] * u.angstrom,
            table.array.data["Transmission"],
        )

    @classmethod
    def from_csv(cls, file) -> Bandpass:
        """Construct a `Bandpass` from a comma-separated table.

        The file must have a header row.  One column must be named ``wvl``
        and hold wavelengths in nanometres; the transmission is the product
        of all remaining columns, which lets separate optical elements
        (mirror, window, photodetector, ...) be tabulated side by side.

        Parameters
        ----------
        file : str or pathlib.Path or file-like
            Anything accepted by :func:`numpy.genfromtxt`.

        Returns
        -------
        Bandpass
            The combined passband.
        """
        arr = np.genfromtxt(file, delimiter=",", names=True)
        lam = arr["wvl"] * u.nm
        trx = recfc.drop_fields(arr, "wvl", usemask=False)
        return cls(lam, np.array(trx.tolist()).prod(axis=1))


class SpectralGrid:
    """Spectra tabulated on an N-dimensional grid of source parameters.

    A grid maps a point in some parameter space onto a spectrum.
    Calling the grid interpolates between the tabulated spectra.

    Parameters
    ----------
    points : list of numpy.ndarray
        Grid coordinates along each parameter axis, one array per axis.  An
        empty list denotes a grid with no parameter axes, i.e. a single
        spectrum shared by all sources.
    wvl : astropy.units.Quantity
        Wavelength grid, shape ``(W,)``.
    flx : astropy.units.Quantity
        Tabulated spectra, shape ``(*grid_shape, W, C)``, where ``C`` is the
        number of spectral components carried alongside each other (e.g.
        minimum/median/maximum variants of a model).

    See Also
    --------
    RateGrid : The same structure after integration over wavelength.
    """

    def __init__(self, points: list, wvl: u.Quantity, flx: u.Quantity) -> None:
        self.points = points
        self.wvl = wvl
        self.flx = flx

    def __call__(self, xi: np.ndarray) -> u.Quantity:
        """Interpolate the grid at the given parameter coordinates.

        Parameters
        ----------
        xi : numpy.ndarray
            Parameter coordinates, shape ``(N, D)`` where ``D`` matches the
            number of grid axes.  An empty array returns the tabulated
            spectra unchanged, which is the correct behaviour for grids
            without parameter axes.

        Returns
        -------
        astropy.units.Quantity
            Interpolated spectra, shape ``(N, W, C)``.  Points outside the
            grid produce ``nan``.
        """
        if xi.size == 0:
            return self.flx
        rgi = RegularGridInterpolator(self.points, self.flx, bounds_error=False)
        return rgi(xi) * self.flx.unit

    def apply_bandpass(self, bandpass: Bandpass) -> SpectralGrid:
        """Restrict to a passband and weight by its transmission.

        Parameters
        ----------
        bandpass : Bandpass
            The passband to apply.

        Returns
        -------
        SpectralGrid
            A new grid, truncated to the wavelength range of ``bandpass`` and
            multiplied by its transmission.  The input grid is not modified.
        """
        mask = (self.wvl >= bandpass.min) & (self.wvl <= bandpass.max)
        wvl = self.wvl[mask]
        flx = np.einsum("a,...ab->...ab", bandpass(wvl), self.flx[..., mask, :])
        return SpectralGrid(self.points, wvl, flx)

    def integrate(self) -> RateGrid:
        """Integrate every tabulated spectrum over wavelength.

        Returns
        -------
        RateGrid
            Grid of band-integrated rates, with the wavelength axis removed.
        """
        rate = integrate_wavelength(self.flx, self.wvl, axis=-2)
        return RateGrid(self.points, rate)

    def __mul__(self, value) -> SpectralGrid:
        """Scale the tabulated spectra.

        Parameters
        ----------
        value : array_like
            Scale factor.  Trailing axes are added as needed so that ``value``
            broadcasts against the leading axes of ``flx``.

        Returns
        -------
        SpectralGrid
            A new, scaled grid.  The input grid is not modified.
        """
        v = np.asarray(value)
        for _ in range(self.flx.ndim - v.ndim):
            v = v[..., np.newaxis]
        return SpectralGrid(self.points, self.wvl, self.flx * v)


class RateGrid:
    """Band-integrated rates tabulated on an N-dimensional parameter grid.

    Parameters
    ----------
    points : list of numpy.ndarray
        Grid coordinates along each parameter axis, one array per axis.
    rate : astropy.units.Quantity
        Tabulated rates, shape ``(*grid_shape, C)``.

    See Also
    --------
    SpectralGrid : The wavelength-resolved counterpart.
    """

    def __init__(self, points: list, rate: u.Quantity) -> None:
        self.points = points
        self.rate = rate

    def __call__(self, xi: np.ndarray) -> u.Quantity:
        """Interpolate the grid at the given parameter coordinates.

        Parameters
        ----------
        xi : numpy.ndarray
            Parameter coordinates, shape ``(N, D)``.  An empty array returns
            the tabulated rates unchanged.

        Returns
        -------
        astropy.units.Quantity
            Interpolated rates, shape ``(N, C)``.
        """
        if xi.size == 0:
            return self.rate
        rgi = RegularGridInterpolator(self.points, self.rate, bounds_error=False)
        return rgi(xi) * self.rate.unit

    def __mul__(self, value) -> RateGrid:
        """Scale the tabulated rates per source and grid point.

        Parameters
        ----------
        value : array_like
            Scale factors, shape ``(N, G)`` for ``G`` grid points.

        Returns
        -------
        RateGrid
            A new, scaled grid.  The input grid is not modified.
        """
        return RateGrid(self.points, np.einsum("b...,ab->ab...", self.rate, value))
