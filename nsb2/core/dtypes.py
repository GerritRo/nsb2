from __future__ import annotations

from dataclasses import dataclass

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord

from nsb2.core.spectral import Bandpass, SpectralGrid, integrate_wavelength

__all__ = [
    "PixelRefs",
    "Prediction",
    "ResolvedField",
    "SourceField",
]


@dataclass
class SourceField:
    """Sources on the sky together with a reference to their spectra.

    Attributes
    ----------
    coords : astropy.coordinates.SkyCoord
        Source positions, shape ``(N,)``.
    weights : astropy.units.Quantity
        Per-source brightness weights, shape ``(N, C_w)``. For radiance
        fields these carry the solid-angle-dependent scaling.
    spectral_data : numpy.ndarray
        Coordinates into ``spectral_grid``, shape ``(N, D)``. ``D`` is zero
        when all sources share a single spectrum.
    spectral_grid : nsb2.core.spectral.SpectralGrid
        The grid the spectral coordinates refer to.
    radiance_field : bool, optional
        `True` if the field is a radiance (per unit solid angle) that maps
        one-to-one onto instrument pixels, `False` for discrete sources.
        Default is `False`.
    """

    coords: SkyCoord
    weights: u.Quantity
    spectral_data: np.ndarray
    spectral_grid: SpectralGrid
    radiance_field: bool = False

    def resolve_spectra(self, bandpass: Bandpass) -> ResolvedField:
        """Look up the full spectrum of every source in a passband.

        Parameters
        ----------
        bandpass : nsb2.core.spectral.Bandpass
            Passband to restrict the spectra to.

        Returns
        -------
        ResolvedField
            A new field carrying wavelength-resolved fluxes.  This field is
            not modified.
        """
        grid = self.spectral_grid.apply_bandpass(bandpass)
        return ResolvedField(
            coords=self.coords,
            weights=self.weights,
            wvl=grid.wvl,
            flx=grid(self.spectral_data),
            radiance_field=self.radiance_field,
        )


@dataclass
class ResolvedField:
    """Sources whose spectra have been resolved onto a wavelength grid.

    Attributes
    ----------
    coords : astropy.coordinates.SkyCoord
        Source positions, shape ``(N,)``.
    weights : astropy.units.Quantity
        Per-source brightness weights, shape ``(N, C_w)``.
    wvl : astropy.units.Quantity
        Wavelength grid, shape ``(W,)``.
    flx : astropy.units.Quantity
        Per-source spectra, shape ``(N, W, C_comp)``.
    radiance_field : bool
        `True` if the field is a radiance rather than a set of discrete
        sources.

    See Also
    --------
    SourceField.resolve_spectra : Produces instances of this class.
    """

    coords: SkyCoord
    weights: u.Quantity
    wvl: u.Quantity
    flx: u.Quantity
    radiance_field: bool

    def integrate(self, extra_weights: np.ndarray | None = None) -> u.Quantity:
        """Integrate the weighted spectra over wavelength.

        Parameters
        ----------
        extra_weights : array_like, optional
            Additional multiplicative, wavelength-dependent weights such as
            atmospheric extinction.  Must broadcast against
            ``(N, W, C_comp)``.  Default is no extra weighting.

        Returns
        -------
        astropy.units.Quantity
            Per-source rates, shape ``(N, C_comp)``.
        """
        integrand = self.weights[:, None] * self.flx
        if extra_weights is not None:
            integrand = extra_weights * integrand
        return integrate_wavelength(integrand, self.wvl, axis=-2)


@dataclass
class PixelRefs:
    """Assignment of sources to instrument pixels.

    Sources are stored once and referenced by index, because a single source
    can fall into the acceptance of several neighbouring pixels.

    Attributes
    ----------
    indices : list of numpy.ndarray
        One integer array per pixel, holding the indices of the sources that
        contribute to it.
    weights : list of astropy.units.Quantity or None, optional
        One array per pixel, holding the instrument response weight of each
        contributing source, in ``m2`` for point sources or ``m2 sr`` for
        radiance fields.  `None` until an instrument has filled them in via
        :meth:`nsb2.core.instrument.Instrument.compute_pixel_weights`.
        Default is `None`.
    """

    indices: list[np.ndarray]
    weights: list[u.Quantity] | None = None


@dataclass
class Prediction:
    """Predicted pixel rates from one source along one light path.

    Attributes
    ----------
    rates : astropy.units.Quantity
        Per-pixel rates, shape ``(N_pix, 3)``.  The trailing axis holds the
        minimum, median and maximum of the spectral component variants, which
        bracket the model uncertainty.
    indirect : bool
        `True` if the rates come from a scattering path, `False` for direct
        light.
    source_name : str, optional
        Name of the source that produced the prediction.  Default is ``""``.
    path_name : str, optional
        Name of the light path that produced the prediction.  Default is
        ``""``.
    """

    rates: u.Quantity
    indirect: bool
    source_name: str = ""
    path_name: str = ""
