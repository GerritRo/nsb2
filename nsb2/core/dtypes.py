from __future__ import annotations

from dataclasses import dataclass

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord
from scipy.integrate import simpson

from nsb2.core.spectral import Bandpass, SpectralGrid


@dataclass
class SourceField:
    """Sources on the sky with associated spectral data.

    This is the unified output of all source queries — point sources,
    diffuse fields, and ephemeris bodies all produce this type.
    """
    coords: SkyCoord              # (N,) positions
    weights: np.ndarray           # (N, C_w) brightness weights (Quantity)
    spectral_data: np.ndarray     # (N, D) indices into spectral grid
    spectral_grid: SpectralGrid   # shared grid for spectral resolution
    radiance_field: bool = False  # Marking if it is a radiance

    def resolve_spectra(self, bandpass: Bandpass) -> ResolvedField:
        """Resolve spectral indices to full wavelength-dependent fluxes."""
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
    """Sources with fully resolved spectra (wavelength axis present)."""
    coords: SkyCoord
    weights: u.Quantity     # (N, C_w)
    wvl: u.Quantity         # (W,) wavelength grid (Quantity)
    flx: u.Quantity         # (N, W, C_comp) flux array (Quantity)
    radiance_field: bool

    def integrate(self, extra_weights: np.ndarray | None = None) -> u.Quantity:
        """Integrate flux over wavelength, returning per-source rates.

        Parameters
        ----------
        extra_weights : array-like, optional
            Additional multiplicative weights (e.g. atmospheric extinction).
            Shape must broadcast with (N, W, C_comp).

        Returns
        -------
        rates : Quantity, shape (N, C_comp)
        """
        integrand = self.weights[:, None] * self.flx
        if extra_weights is not None:
            integrand = extra_weights * integrand
        rate_unit = self.flx.unit * self.wvl.unit * self.weights.unit
        return simpson(integrand, x=self.wvl, axis=-2) * rate_unit


@dataclass
class PixelRefs:
    """Maps sources to instrument pixels.

    For N_pix pixels and N_sources unique sources:
    - indices[i] is an array of source indices assigned to pixel i
    - weights[i] is the corresponding instrument response weights
    """
    indices: list[np.ndarray]                # list of N_pix arrays of int
    weights: list[u.Quantity] | None = None  # list of N_pix arrays of Quantity (m^2 or m^2*sr), or None before instrument fills them


@dataclass
class Prediction:
    """A single prediction result from one source per one light path."""
    rates: np.ndarray     # (N_pix, 3) min/med/max
    indirect: bool        # True if from scattering path
    source_name: str = ""
    path_name: str = ""
