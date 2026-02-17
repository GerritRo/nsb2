from __future__ import annotations

from abc import ABC, abstractmethod

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord
from scipy.integrate import simpson as simps
from scipy.ndimage import map_coordinates

from nsb2.core.dtypes import PixelRefs, SourceField
from nsb2.core.interpolation import UnitRegularGridInterpolator
from nsb2.core.spectral import Bandpass


def _min_med_max(arr, axis=-1):
    """Stack min / median / max along *axis* into a new trailing dimension."""
    return np.stack([f(arr, axis=axis) for f in
                     (np.nanmin, np.nanmedian, np.nanmax)], axis=-1)


class Instrument(ABC):
    """Base class for telescope/camera instruments."""

    bandpass: Bandpass
    _pix_pos: np.ndarray
    _pix_area_sr: np.ndarray

    @abstractmethod
    def pixel_coords(self, observation) -> SkyCoord:
        """Pixel center positions as SkyCoord in the observation frame."""
        ...

    @abstractmethod
    def pixel_radii(self) -> np.ndarray:
        """Search radii for spatial source queries (radians)."""
        ...

    @abstractmethod
    def fov_range(self) -> tuple[tuple[float, float], tuple[float, float]]:
        """Field of view as ((lon_min, lon_max), (lat_min, lat_max)) in radians."""
        ...

    def eval_grid(self, observation, n: int = 2) -> SkyCoord:
        """Evaluation grid for scattered light interpolation.

        Parameters
        ----------
        observation : coordinate frame
        n : int
            Number of grid points per axis (n x n grid).
        """
        lon_range, lat_range = self.fov_range()
        X, Y = np.meshgrid(np.linspace(*lon_range, n), np.linspace(*lat_range, n))
        return SkyCoord(X, Y, unit='rad', frame=observation).transform_to(observation.origin)

    def project_discrete(self, rates: np.ndarray, pixel_refs: PixelRefs) -> u.Quantity:
        """Project rates to pixels via discrete assignment.

        Parameters
        ----------
        rates : ndarray, shape (N_sources, C_comp)
            Per-source rates (already weighted by source brightness).
        pixel_refs : PixelRefs
            Mapping from sources to pixels.

        Returns
        -------
        pixel_rates : Quantity, shape (N_pix, 3)
            Per-pixel min/med/max rates.
        """
        rstack = _min_med_max(rates)

        n_pix = len(pixel_refs.indices)
        if pixel_refs.weights is None:
            raise ValueError("pixel_refs.weights must not be None")
        weights = pixel_refs.weights
        results = []
        for i in range(n_pix):
            idx = pixel_refs.indices[i]
            w = weights[i]
            if len(idx) == 0:
                results.append(np.full(3, 0)* w.unit * rstack.unit)
            else:
                results.append(np.nansum(w[:, None] * rstack[idx], axis=0))

        return u.Quantity(results)


    def project_continuous(self, rates: np.ndarray, eval_coords: SkyCoord) -> u.Quantity:
        """Project a grid rate field to pixels via interpolation.

        Parameters
        ----------
        rates : ndarray, shape (M_lat, M_lon, N_sources, C_comp)
            Rate field on the evaluation grid.
        eval_coords : SkyCoord, shape (M_lat, M_lon)
            The evaluation grid coordinates (used for shape only).

        Returns
        -------
        pixel_rates : Quantity, shape (N_pix, 3)
        """
        rstack = _min_med_max(rates)
        # Use known FOV grid (same linspace as eval_grid)
        lon_range, lat_range = self.fov_range()
        lon = np.linspace(*lon_range, eval_coords.shape[1])
        lat = np.linspace(*lat_range, eval_coords.shape[0])
        summed = np.nansum(rstack, axis=-2)  # sum over sources
        rgi = UnitRegularGridInterpolator([lat, lon], summed,
                                          bounds_error=False, fill_value=None)
        return rgi(self._pix_pos[:, ::-1]) * self._pix_area_sr[:, None] * u.m**2 * u.radian**2

    @abstractmethod
    def compute_pixel_weights(self, field: SourceField, pixel_refs: PixelRefs, observation) -> PixelRefs:
        """Fill in instrument-specific pixel weights for a PixelRefs.

        Parameters
        ----------
        field : SourceField
            The queried source field.
        pixel_refs : PixelRefs
            Pixel refs with indices populated; weights to be filled.
        observation : coordinate frame

        Returns
        -------
        pixel_refs : PixelRefs
            New PixelRefs with weights filled in.
        """
        ...


class EffectiveApertureInstrument(Instrument):
    """Instrument with per-pixel effective aperture response functions."""

    def __init__(self, response: dict, bandpass: Bandpass) -> None:
        x = np.asarray(response['x'])          
        y = np.asarray(response['y'])          
        vals = np.asarray(response['values']) 

        self._pix_pos = np.stack([np.mean(x, axis=1),
                                  np.mean(y, axis=1)]).T
        self._pix_bins = np.stack([x[:, [0, -1]],
                                   y[:, [0, -1]]], axis=1)
        self._pix_rad = np.max(np.diff(self._pix_bins, axis=2), axis=(1, 2)) / np.sqrt(2)

        # Precompute FOV range
        self._fov = ((self._pix_bins[:, 1].min(), self._pix_bins[:, 1].max()),
                     (self._pix_bins[:, 0].min(), self._pix_bins[:, 0].max()))

        inner = simps(vals, x=x[:, np.newaxis, :], axis=-1)  
        self._pix_area_sr = np.asarray(simps(inner, x=y, axis=-1)) 

        self._response_values = vals
        self._resp_x0 = x[:, 0]
        self._resp_y0 = y[:, 0]
        self._resp_x_scale = (x.shape[1] - 1) / (x[:, -1] - x[:, 0])
        self._resp_y_scale = (y.shape[1] - 1) / (y[:, -1] - y[:, 0])
        self.bandpass = bandpass

    @property
    def n_pixels(self) -> int:
        return len(self._pix_pos)

    def pixel_coords(self, observation) -> SkyCoord:
        return SkyCoord(self._pix_pos[:, 0], self._pix_pos[:, 1],
                        unit='rad', frame=observation)

    def pixel_radii(self) -> np.ndarray:
        return self._pix_rad

    def fov_range(self) -> tuple[tuple[float, float], tuple[float, float]]:
        return self._fov

    def compute_pixel_weights(self, field: SourceField, pixel_refs: PixelRefs, observation) -> PixelRefs:
        from dataclasses import replace

        if field.radiance_field:
            weights = list(self._pix_area_sr[:, np.newaxis] * (u.m**2 * u.radian**2))
        else:
            s_coords = field.coords.transform_to(observation.origin).transform_to(observation)
            all_lon = s_coords.lon.rad
            all_lat = s_coords.lat.rad

            counts = np.fromiter(
                (len(idx) for idx in pixel_refs.indices),
                dtype=int, count=self.n_pixels)
            all_idx = np.concatenate(
                [np.asarray(idx, dtype=int) for idx in pixel_refs.indices])

            pix_ids = np.repeat(np.arange(self.n_pixels), counts)
            fx = (all_lon[all_idx] - self._resp_x0[pix_ids]) * self._resp_x_scale[pix_ids]
            fy = (all_lat[all_idx] - self._resp_y0[pix_ids]) * self._resp_y_scale[pix_ids]

            flat_vals = map_coordinates(
                self._response_values,
                np.array([pix_ids.astype(np.float64), fx, fy]),
                order=1, mode='constant', cval=0.0)

            weights = [chunk * u.m**2
                       for chunk in np.split(flat_vals, np.cumsum(counts[:-1]))]

        return replace(pixel_refs, weights=weights)