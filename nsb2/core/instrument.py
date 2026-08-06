"""Telescope and camera models.

An instrument knows how a photon arriving from a given direction is collected
into a given pixel, and nothing about where the photon came from.  It is
responsible for two things: weighting each source by the effective collection
area of the pixel it lands in, and projecting rates onto the pixel grid.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import replace

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord
from scipy.integrate import simpson
from scipy.ndimage import map_coordinates

from nsb2.core.dtypes import PixelRefs, SourceField
from nsb2.core.interpolation import UnitRegularGridInterpolator
from nsb2.core.spectral import Bandpass

__all__ = [
    "EffectiveApertureInstrument",
    "Instrument",
]


logger = logging.getLogger(__name__)


def _min_med_max(arr, axis=-1):
    """Reduce the spectral component axis to its minimum, median and maximum.

    The component axis holds model variants that bracket the uncertainty of a
    source; collapsing it to these three summary statistics is what lets a
    prediction be reported as a range.

    Parameters
    ----------
    arr : array_like
        Array whose component axis is to be reduced.
    axis : int, optional
        Axis holding the components.  Default is the last axis.

    Returns
    -------
    array_like
        ``arr`` with ``axis`` replaced by a trailing axis of length three,
        holding the minimum, median and maximum.  ``nan`` values are ignored.
    """
    return np.stack(
        [f(arr, axis=axis) for f in (np.nanmin, np.nanmedian, np.nanmax)], axis=-1
    )


class Instrument(ABC):
    """Base class for telescope and camera models.

    Attributes
    ----------
    bandpass : nsb2.core.spectral.Bandpass
        Wavelength response of the full optical chain.
    """

    bandpass: Bandpass
    _pix_pos: np.ndarray
    _pix_area_sr: np.ndarray

    @abstractmethod
    def pixel_coords(self, observation) -> SkyCoord:
        """Return the pixel centre positions.

        Parameters
        ----------
        observation : astropy.coordinates.BaseCoordinateFrame
            The telescope pointing frame.

        Returns
        -------
        astropy.coordinates.SkyCoord
            Pixel centres in the observation frame, shape ``(N_pix,)``.
        """
        ...

    @abstractmethod
    def pixel_radii(self) -> np.ndarray:
        """Return the per-pixel search radius for spatial source queries.

        Returns
        -------
        numpy.ndarray
            Radii in radians, shape ``(N_pix,)``.  Large enough to cover the
            pixel's full acceptance, so that no contributing source is missed.
        """
        ...

    @abstractmethod
    def fov_range(self) -> tuple[tuple[float, float], tuple[float, float]]:
        """Return the extent of the field of view.

        Returns
        -------
        tuple
            ``((lon_min, lon_max), (lat_min, lat_max))`` in radians, relative
            to the pointing direction.
        """
        ...

    def eval_grid(self, observation, n: int = 2) -> SkyCoord:
        """Build the grid on which scattered light is evaluated.

        Scattered light varies slowly across a field of view a few degrees
        wide, so it is evaluated on a coarse grid and interpolated onto the
        pixels rather than computed per pixel.

        Parameters
        ----------
        observation : astropy.coordinates.BaseCoordinateFrame
            The telescope pointing frame.
        n : int, optional
            Number of grid points per axis, giving an ``n`` by ``n`` grid.
            Default is 2, i.e. the corners of the field of view.

        Returns
        -------
        astropy.coordinates.SkyCoord
            Grid positions, shape ``(n, n)``.
        """
        lon_range, lat_range = self.fov_range()
        x, y = np.meshgrid(np.linspace(*lon_range, n), np.linspace(*lat_range, n))
        return SkyCoord(x, y, unit="rad", frame=observation).transform_to(
            observation.origin
        )

    def project_discrete(self, rates: np.ndarray, pixel_refs: PixelRefs) -> u.Quantity:
        """Sum discrete source rates into the pixels they belong to.

        Parameters
        ----------
        rates : astropy.units.Quantity
            Per-source rates, shape ``(N_sources, C_comp)``, already weighted
            by source brightness.
        pixel_refs : nsb2.core.dtypes.PixelRefs
            Source-to-pixel assignment, with ``weights`` filled in.

        Returns
        -------
        astropy.units.Quantity
            Per-pixel rates, shape ``(N_pix, 3)``, holding the minimum,
            median and maximum over the spectral components.

        Raises
        ------
        ValueError
            If ``pixel_refs.weights`` has not been filled in.
        """
        if pixel_refs.weights is None:
            raise ValueError("pixel_refs.weights must not be None")

        rstack = _min_med_max(rates)
        weights = pixel_refs.weights

        results = []
        for idx, weight in zip(pixel_refs.indices, weights, strict=True):
            if len(idx) == 0:
                results.append(np.zeros(3) * weight.unit * rstack.unit)
            else:
                results.append(np.nansum(weight[:, None] * rstack[idx], axis=0))

        return u.Quantity(results)

    def project_continuous(
        self, rates: np.ndarray, eval_coords: SkyCoord
    ) -> u.Quantity:
        """Interpolate a rate field from the evaluation grid onto the pixels.

        Parameters
        ----------
        rates : astropy.units.Quantity
            Rate field on the evaluation grid, shape
            ``(M_lat, M_lon, N_sources, C_comp)``.
        eval_coords : astropy.coordinates.SkyCoord
            The evaluation grid, shape ``(M_lat, M_lon)``.  Only its shape is
            used; the positions are reconstructed from
            :meth:`fov_range` to match :meth:`eval_grid`.

        Returns
        -------
        astropy.units.Quantity
            Per-pixel rates, shape ``(N_pix, 3)``, scaled by each pixel's
            solid angle since the field is a radiance.
        """
        rstack = _min_med_max(rates)
        lon_range, lat_range = self.fov_range()
        lon = np.linspace(*lon_range, eval_coords.shape[1])
        lat = np.linspace(*lat_range, eval_coords.shape[0])
        summed = np.nansum(rstack, axis=-2)
        rgi = UnitRegularGridInterpolator(
            [lat, lon], summed, bounds_error=False, fill_value=None
        )
        return (
            rgi(self._pix_pos[:, ::-1])
            * self._pix_area_sr[:, None]
            * u.m**2
            * u.radian**2
        )

    @abstractmethod
    def compute_pixel_weights(
        self, field: SourceField, pixel_refs: PixelRefs, observation
    ) -> PixelRefs:
        """Fill in the instrument response weight of every source-pixel pair.

        Parameters
        ----------
        field : nsb2.core.dtypes.SourceField
            The queried source field.
        pixel_refs : nsb2.core.dtypes.PixelRefs
            Assignment with ``indices`` populated and ``weights`` unset.
        observation : astropy.coordinates.BaseCoordinateFrame
            The telescope pointing frame.

        Returns
        -------
        nsb2.core.dtypes.PixelRefs
            A new assignment with ``weights`` filled in.  The input is not
            modified.
        """
        ...


class EffectiveApertureInstrument(Instrument):
    """Instrument described by a per-pixel effective aperture map.

    The response of each pixel is tabulated as an effective collection area
    over a small grid of offsets from the pointing direction.  This folds the
    mirror area, the optical transmission, the shadowing of the camera body
    and the light guide acceptance into a single quantity, which is what
    ray-tracing simulations of a telescope produce.

    Parameters
    ----------
    response : dict or numpy.lib.npyio.NpzFile
        Mapping with keys ``"x"``, ``"y"`` and ``"values"``.  ``x`` and ``y``
        hold the offset grid of each pixel in radians, shape ``(N_pix, G)``;
        ``values`` holds the effective area in square metres per steradian,
        shape ``(N_pix, G, G)``.
    bandpass : nsb2.core.spectral.Bandpass
        Wavelength response of the full optical chain.

    See Also
    --------
    nsb2.instrument.CTAO : Ready-made CTAO telescope models.
    nsb2.instrument.HESS : Ready-made H.E.S.S. telescope models.
    """

    def __init__(self, response: dict, bandpass: Bandpass) -> None:
        x = np.asarray(response["x"])
        y = np.asarray(response["y"])
        vals = np.asarray(response["values"])

        self._pix_pos = np.stack([np.mean(x, axis=1), np.mean(y, axis=1)]).T
        self._pix_bins = np.stack([x[:, [0, -1]], y[:, [0, -1]]], axis=1)
        self._pix_rad = np.max(np.diff(self._pix_bins, axis=2), axis=(1, 2)) / np.sqrt(
            2
        )

        self._fov = (
            (self._pix_bins[:, 1].min(), self._pix_bins[:, 1].max()),
            (self._pix_bins[:, 0].min(), self._pix_bins[:, 0].max()),
        )

        inner = simpson(vals, x=x[:, np.newaxis, :], axis=-1)
        self._pix_area_sr = np.asarray(simpson(inner, x=y, axis=-1))

        self._response_values = vals
        self._resp_x0 = x[:, 0]
        self._resp_y0 = y[:, 0]
        self._resp_x_scale = (x.shape[1] - 1) / (x[:, -1] - x[:, 0])
        self._resp_y_scale = (y.shape[1] - 1) / (y[:, -1] - y[:, 0])
        self.bandpass = bandpass

        logger.debug("loaded instrument response for %d pixels", self.n_pixels)

    @property
    def n_pixels(self) -> int:
        """int: Number of camera pixels."""
        return len(self._pix_pos)

    def pixel_coords(self, observation) -> SkyCoord:
        """Return the pixel centre positions.

        See :meth:`Instrument.pixel_coords` for the parameters and returns.
        """
        return SkyCoord(
            self._pix_pos[:, 0], self._pix_pos[:, 1], unit="rad", frame=observation
        )

    def pixel_radii(self) -> np.ndarray:
        """Return the per-pixel search radius.

        The radius is half the diagonal of the pixel's response grid, so that
        the whole tabulated acceptance is covered.

        See :meth:`Instrument.pixel_radii` for the returns.
        """
        return self._pix_rad

    def fov_range(self) -> tuple[tuple[float, float], tuple[float, float]]:
        """Return the extent of the field of view.

        See :meth:`Instrument.fov_range` for the returns.
        """
        return self._fov

    def compute_pixel_weights(
        self, field: SourceField, pixel_refs: PixelRefs, observation
    ) -> PixelRefs:
        """Look up the effective aperture for every source-pixel pair.

        For a radiance field every pixel gets its solid-angle-integrated
        aperture, since the emission fills the pixel.  For discrete sources
        the response map is interpolated at the source's offset from the
        pixel centre.

        See :meth:`Instrument.compute_pixel_weights` for the parameters and
        returns.
        """
        if field.radiance_field:
            weights = list(self._pix_area_sr[:, np.newaxis] * (u.m**2 * u.radian**2))
        else:
            s_coords = field.coords.transform_to(observation.origin).transform_to(
                observation
            )
            all_lon = s_coords.lon.rad
            all_lat = s_coords.lat.rad

            counts = np.fromiter(
                (len(idx) for idx in pixel_refs.indices),
                dtype=int,
                count=self.n_pixels,
            )
            all_idx = np.concatenate(
                [np.asarray(idx, dtype=int) for idx in pixel_refs.indices]
            )

            pix_ids = np.repeat(np.arange(self.n_pixels), counts)
            fx = (all_lon[all_idx] - self._resp_x0[pix_ids]) * self._resp_x_scale[
                pix_ids
            ]
            fy = (all_lat[all_idx] - self._resp_y0[pix_ids]) * self._resp_y_scale[
                pix_ids
            ]

            flat_vals = map_coordinates(
                self._response_values,
                np.array([pix_ids.astype(np.float64), fx, fy]),
                order=1,
                mode="constant",
                cval=0.0,
            )

            weights = [
                chunk * u.m**2 for chunk in np.split(flat_vals, np.cumsum(counts[:-1]))
            ]

        return replace(pixel_refs, weights=weights)
