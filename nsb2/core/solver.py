from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord
from scipy.integrate import simpson

from nsb2.core.interpolation import UnitRegularGridInterpolator

if TYPE_CHECKING:
    from nsb2.core.atmosphere import Atmosphere
    from nsb2.core.dtypes import SourceField
    from nsb2.core.instrument import Instrument
    from nsb2.core.sources import Source
    from nsb2.core.spectral import Bandpass


def _trapz_einsum(a, b, wvl, eins_str):
    """Trapezoidal integration via einsum (used for LUT compilation)."""
    delta_x = np.diff(wvl) / 2
    return (np.einsum(eins_str, a[..., :-1], b[..., :-1, :], delta_x) +
            np.einsum(eins_str, a[..., 1:], b[..., 1:, :], delta_x))


# ---------------------------------------------------------------------------
# Abstract base classes
# ---------------------------------------------------------------------------

class Solver:
    def compile(
        self,
        source: Source,
        instrument: Instrument,
        atmosphere: Atmosphere,
        **kwargs,
    ) -> float:
        """Optional pre-computation step (e.g. LUT generation).

        Returns
        -------
        cost : float
            Compilation cost metric (0 for no-ops).
        """
        return 0


class DirectSolver(Solver):
    """Strategy for computing extinction-weighted rates (direct light)."""

    @abstractmethod
    def compute_rates(
        self,
        source: Source,
        field: SourceField,
        atmosphere: Atmosphere,
        bandpass: Bandpass,
    ) -> u.Quantity:
        """Compute per-source rates via the direct (extinction) path.

        Parameters
        ----------
        source : Source
        field : SourceField
        atmosphere : Atmosphere
        bandpass : Bandpass

        Returns
        -------
        rates : Quantity, shape (N_sources, C_comp)
            Weight-inclusive rates (``field.weights`` already applied).
        """
        ...


class ScatteredSolver(Solver):
    """Strategy for computing scattered light."""

    @abstractmethod
    def compute_rates(
        self,
        source: Source,
        field: SourceField,
        atmosphere: Atmosphere,
        bandpass: Bandpass,
        eval_coords: SkyCoord,
    ) -> u.Quantity:
        """Compute in-scattering rates on the evaluation grid.

        Parameters
        ----------
        source : Source
        field : SourceField
        atmosphere : Atmosphere
        bandpass : Bandpass
        eval_coords : SkyCoord, shape (M_lat, M_lon)

        Returns
        -------
        rates : Quantity, shape (M_lat, M_lon, N_sources, C_comp)
            Weight-inclusive rates (``field.weights`` already applied).
        """
        ...


class ExplicitDirectSolver(DirectSolver):
    """Direct extinction via full spectral integration."""

    def compute_rates(self, source: Source, field: SourceField, atmosphere: Atmosphere, bandpass: Bandpass) -> u.Quantity:
        resolved = field.resolve_spectra(bandpass)
        coords = np.atleast_1d(resolved.coords)
        ext_weights = atmosphere.extinction(
            coords.alt.rad, coords.az.rad, resolved.wvl)
        return resolved.integrate(extra_weights=ext_weights[..., None])


class ExplicitScatteredSolver(ScatteredSolver):
    """Scattered light via full spectral integration."""

    def compute_rates(self, source: Source, field: SourceField, atmosphere: Atmosphere, bandpass: Bandpass, eval_coords: SkyCoord) -> u.Quantity:
        resolved = field.resolve_spectra(bandpass)
        scat_weights = atmosphere.scattering(
            eval_coords.alt.rad[..., None, None],
            eval_coords.az.rad[..., None, None],
            resolved.coords.alt.rad[None, ..., None],
            resolved.coords.az.rad[None, ..., None],
            resolved.wvl)

        rate_unit = (resolved.flx.unit * resolved.wvl.unit
                     * resolved.weights.unit * scat_weights.unit)
        integrand = (scat_weights * resolved.weights)[..., None] * resolved.flx
        return simpson(integrand, x=resolved.wvl, axis=-2) * rate_unit


class LUTDirectSolver(DirectSolver):
    """Direct extinction via pre-compiled azimuthally-symmetric LUT."""

    def __init__(self) -> None:
        self._luts: dict[Source, UnitRegularGridInterpolator] = {}

    def compile(self, source: Source, instrument: Instrument, atmosphere: Atmosphere, *,
                extinction_z_bins: int = 90, **kwargs) -> float:
        spectral_grid = source.spectral_grid.apply_bandpass(instrument.bandpass)
        Z_range = np.linspace(0, np.pi / 2, extinction_z_bins)
        res = _trapz_einsum(
            atmosphere.extinction(Z_range, 0, spectral_grid.wvl),
            spectral_grid.flx, spectral_grid.wvl, 'zN,...Nc,N->z...c')
        self._luts[source] = UnitRegularGridInterpolator(
            (Z_range, *spectral_grid.points), res,
            method='linear', bounds_error=False)
        return 0

    def compute_rates(self, source: Source, field: SourceField, atmosphere: Atmosphere, bandpass: Bandpass) -> u.Quantity:
        try:
            lut = self._luts[source]
        except KeyError:
            raise RuntimeError(
                f"No compiled LUT for {type(source).__name__}. "
                f"Call compile() before predict()."
            ) from None
        coords = np.atleast_1d(field.coords)
        raw_rates = lut(np.column_stack([coords.alt.rad, field.spectral_data]))
        return field.weights * raw_rates


class LUTScatteredSolver(ScatteredSolver):
    """Scattered light via pre-compiled azimuthally-symmetric LUT."""

    def __init__(self) -> None:
        self._luts: dict[Source, UnitRegularGridInterpolator] = {}

    def compile(self, source: Source, instrument: Instrument, atmosphere: Atmosphere, *,
                scattering_z_bins: int = 10, scattering_theta_bins: int = 10, **kwargs) -> float:
        spectral_grid = source.spectral_grid.apply_bandpass(instrument.bandpass)
        Z_range = np.linspace(0, np.pi / 2, scattering_z_bins)
        theta_range = np.linspace(0, np.sqrt(np.pi), scattering_theta_bins)**2
        res = _trapz_einsum(
            atmosphere.scattering(
                Z_range[:, None, None, None], 0,
                Z_range[None, :, None, None],
                theta_range[None, None, :, None],
                spectral_grid.wvl),
            spectral_grid.flx, spectral_grid.wvl, 'abcN,...Nd,N->abc...d')
        self._luts[source] = UnitRegularGridInterpolator(
            (Z_range, Z_range, theta_range, *spectral_grid.points),
            res, method='linear', bounds_error=False)
        return 0

    def compute_rates(self, source: Source, field: SourceField, atmosphere: Atmosphere, bandpass: Bandpass, eval_coords: SkyCoord) -> u.Quantity:
        try:
            lut = self._luts[source]
        except KeyError:
            raise RuntimeError(
                f"No compiled LUT for {type(source).__name__}. "
                f"Call compile() before predict()."
            ) from None
        coord_array = np.broadcast_arrays(
            eval_coords.alt.rad[..., np.newaxis],
            field.coords.alt.rad[np.newaxis, np.newaxis, :],
            np.pi - np.abs(np.abs(eval_coords.az.rad[..., np.newaxis] - field.coords.az.rad) - np.pi))
        lut_input = np.concatenate([
            np.stack(coord_array, axis=-1),
            np.broadcast_to(field.spectral_data,
                            eval_coords.shape + field.spectral_data.shape)
        ], axis=-1)
        raw_rates = lut(lut_input)
        return raw_rates * field.weights
