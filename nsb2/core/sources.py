from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable

import astropy.coordinates
import astropy.units as u
import healpy as hp
import numpy as np
from astropy.coordinates import SkyCoord
from sklearn.neighbors import BallTree

from nsb2.core.dtypes import PixelRefs, SourceField
from nsb2.core.photometry import create_color_grid
from nsb2.core.spectral import SpectralGrid


def _transform_to_frame(skycoord: SkyCoord, frame) -> SkyCoord:
    """Transform a SkyCoord to the given frame, or return as-is if frame is None."""
    if frame is not None:
        skycoord = skycoord.transform_to(frame)
    return skycoord


class Source(ABC):
    """A sky brightness source."""

    name: str = ""

    @property
    @abstractmethod
    def spectral_grid(self) -> SpectralGrid:
        """The SpectralGrid associated with this source."""
        ...

    @abstractmethod
    def query_direct(
        self,
        observation,
        pixel_coords: SkyCoord,
        pixel_radii: np.ndarray,
    ) -> tuple[SourceField, PixelRefs]:
        """Query sources for the extinction/direct-light path.

        Parameters
        ----------
        observation : coordinate frame
            The telescope pointing frame.
        pixel_coords : SkyCoord
            Pixel center positions in the observation's origin frame.
        pixel_radii : array
            Search radii per pixel (for catalog sources).

        Returns
        -------
        field : SourceField
        pixel_refs : PixelRefs
        """
        ...

    @abstractmethod
    def query_scattered(self, observation, nside: int = 64) -> SourceField:
        """Query sources for the scattering/indirect-light path.

        Returns all sources visible in the upper hemisphere.

        Parameters
        ----------
        observation : coordinate frame
        nside : int
            HEALPix resolution for discretised hemisphere queries.

        Returns
        -------
        field : SourceField
        """
        ...


class RadianceSource(Source):
    """Base for diffuse/radiance sources that map 1:1 onto pixels.
    """

    _spectral_grid: SpectralGrid

    @property
    def spectral_grid(self) -> SpectralGrid:
        return self._spectral_grid

    @abstractmethod
    def _query_coords(self, coords: SkyCoord) -> tuple[np.ndarray, np.ndarray]:
        """Return (weight, data) arrays for the given sky coordinates."""
        ...

    def query_direct(
        self,
        observation,
        pixel_coords: SkyCoord,
        pixel_radii: np.ndarray,
    ) -> tuple[SourceField, PixelRefs]:
        sky_coords = pixel_coords.transform_to(observation.origin)
        weight, data = self._query_coords(sky_coords)
        n_pix = len(pixel_coords)

        field = SourceField(
            coords=sky_coords,
            weights=weight,
            spectral_data=data,
            spectral_grid=self.spectral_grid,
            radiance_field=True,
        )
        pixel_refs = PixelRefs(
            indices=[np.array([x]) for x in range(n_pix)],
            weights=None,
        )
        return field, pixel_refs


class CatalogSource(Source):
    """Point sources from a photometric catalog."""

    def __init__(
        self,
        coords: SkyCoord,
        weight: np.ndarray,
        data: np.ndarray,
        spectral_grid: SpectralGrid,
        name: str = "",
    ) -> None:
        self.name = name or type(self).__name__
        self.frame = coords.frame
        self.coords = coords
        self.weight = weight[:, None] if weight.ndim == 1 else weight
        self.data = data[:, None] if data.ndim == 1 else data
        self._spectral_grid = spectral_grid

    @property
    def spectral_grid(self) -> SpectralGrid:
        return self._spectral_grid

    def build_balltree(self) -> None:
        self.balltree = BallTree(self._skycoord2latlon(self.coords), metric='haversine')

    def apply_space_motion(self, time) -> None:
        self.coords = self.coords.apply_space_motion(new_obstime=time)
        self.build_balltree()

    def query_direct(self, observation, pixel_coords: SkyCoord, pixel_radii: np.ndarray) -> tuple[SourceField, PixelRefs]:
        sky_coords = pixel_coords.transform_to(observation.origin)
        refs = self.balltree.query_radius(self._skycoord2latlon(sky_coords), pixel_radii)
        unique_indices, inverse_indices = np.unique(np.concatenate(refs), return_inverse=True)
        split_indices = np.cumsum([len(ref) for ref in refs])[:-1]
        new_refs = np.split(inverse_indices, split_indices)

        s_coords = self.coords[unique_indices]

        field = SourceField(
            coords=s_coords.transform_to(observation.origin),
            weights=self.weight[unique_indices],
            spectral_data=self.data[unique_indices],
            spectral_grid=self._spectral_grid,
        )
        pixel_refs = PixelRefs(indices=new_refs, weights=None)  # weights filled by instrument
        return field, pixel_refs

    def query_scattered(self, observation, nside: int = 64) -> SourceField:
        zenith = SkyCoord(0, 90, unit='deg', frame=observation.origin)
        refs = self.balltree.query_radius(self._skycoord2latlon(zenith), np.pi / 2)
        unique_indices = np.unique(np.concatenate(refs))

        s_coords = self.coords[unique_indices]
        return SourceField(
            coords=s_coords.transform_to(observation.origin),
            weights=self.weight[unique_indices],
            spectral_data=self.data[unique_indices],
            spectral_grid=self._spectral_grid,
        )

    def to_map(self, nside: int) -> HEALPixSource:
        npix = hp.nside2npix(nside)
        hp_inds = hp.ang2pix(nside, self.coords.spherical.lon.deg,
                             self.coords.spherical.lat.deg, nest=True, lonlat=True)
        weight = np.vstack([np.bincount(hp_inds, self.weight[:, i], npix)
                            for i in range(self.weight.shape[1])])
        if self.data.shape[1] == 0:
            data = np.empty((0, npix))
        else:
            data = np.where(np.isnan(self.data), np.nanmean(self.data, axis=0), self.data)
            data = np.vstack([np.bincount(hp_inds, data[:, i] * self.weight[:, i], npix) /
                              np.bincount(hp_inds, self.weight[:, i], npix)
                              for i in range(data.shape[1])])
        area_corr = hp.nside2pixarea(nside) * u.radian**2
        return HEALPixSource(self.frame, weight / area_corr, data, self._spectral_grid, name=self.name)

    def __getitem__(self, item) -> CatalogSource:
        return CatalogSource(self.coords[item], self.weight[item],
                             self.data[item], self._spectral_grid, name=self.name)

    def _skycoord2latlon(self, skycoord: SkyCoord) -> np.ndarray:
        skycoord = _transform_to_frame(skycoord, self.frame)
        return np.vstack([skycoord.spherical.lat.rad, skycoord.spherical.lon.rad]).T

    @classmethod
    def from_photometric_catalog(cls, coords, magnitude, color, spectral_library, name="") -> CatalogSource:
        color_range = [np.nanmin(color[1]), np.nanmax(color[1])]
        color_grid = create_color_grid(magnitude[0], color[0], color_range, spectral_library)
        return cls(coords, 10**(-0.4 * magnitude[1]) * u.dimensionless_unscaled,
                   color[1], color_grid, name=name)


class EphemerisSource(Source):
    """A solar system body (e.g. Moon) with time-dependent position."""

    def __init__(
        self,
        body: str,
        weight_function: Callable,
        data_function: Callable,
        spectral_grid: SpectralGrid,
        name: str = "",
    ) -> None:
        self.name = name or body
        self.body = body
        self.weight_function = weight_function
        self.data_function = data_function
        self._spectral_grid = spectral_grid

    @property
    def spectral_grid(self) -> SpectralGrid:
        return self._spectral_grid

    def _query_body(self, obstime) -> tuple[SkyCoord, np.ndarray, np.ndarray]:
        body_coords = astropy.coordinates.get_body(self.body, obstime)
        weight = self.weight_function(obstime)
        data = self.data_function(obstime)
        if body_coords.isscalar:
            body_coords = body_coords.reshape(1)
        weight = np.atleast_2d(weight)
        data = np.atleast_2d(data)
        return body_coords, weight, data

    def _empty_field(self) -> SourceField:
        """Return an empty SourceField (no sources above horizon)."""
        return SourceField(
            coords=SkyCoord([], [], unit='deg', frame='altaz'),
            weights=np.empty((0, 1)) * u.dimensionless_unscaled,
            spectral_data=np.empty((0, 1)),
            spectral_grid=self._spectral_grid,
        )

    def query_direct(self, observation, pixel_coords: SkyCoord, pixel_radii: np.ndarray) -> tuple[SourceField, PixelRefs]:
        b_coord, weight, data = self._query_body(observation.origin.obstime)
        b_coord = b_coord.transform_to(observation.origin)

        above_horizon = b_coord.alt.rad > 0
        if not np.any(above_horizon):
            n_pix = len(pixel_coords)
            field = self._empty_field()
            pixel_refs = PixelRefs(
                indices=[np.array([], dtype=int) for _ in range(n_pix)],
                weights=None,
            )
            return field, pixel_refs

        body_visible = b_coord[above_horizon]
        field = SourceField(
            coords=body_visible,
            weights=weight[above_horizon],
            spectral_data=data[above_horizon],
            spectral_grid=self._spectral_grid,
        )

        sky_coords = pixel_coords.transform_to(observation.origin)
        n_pix = len(pixel_coords)
        len(body_visible)
        indices = []
        for i in range(n_pix):
            seps = sky_coords[i].separation(body_visible).rad
            within = np.where(seps < pixel_radii[i])[0]
            indices.append(within.astype(int))

        pixel_refs = PixelRefs(indices=indices, weights=None)
        return field, pixel_refs

    def query_scattered(self, observation, nside: int = 64) -> SourceField:
        b_coord, weight, data = self._query_body(observation.origin.obstime)
        b_coord = b_coord.transform_to(observation.origin)

        above_horizon = b_coord.alt.rad > 0
        if not np.any(above_horizon):
            return self._empty_field()

        return SourceField(
            coords=b_coord[above_horizon],
            weights=weight[above_horizon],
            spectral_data=data[above_horizon],
            spectral_grid=self._spectral_grid,
        )


class LonLatSource(RadianceSource):
    """Diffuse source defined by functions of longitude/latitude."""

    def __init__(
        self,
        frame,
        weight_function: Callable,
        data_function: Callable,
        spectral_grid: SpectralGrid,
        name: str = "",
    ) -> None:
        self.name = name or type(self).__name__
        self.frame = frame
        self.weight_function = weight_function
        self.data_function = data_function
        self._spectral_grid = spectral_grid

    def _query_coords(self, coords: SkyCoord) -> tuple[np.ndarray, np.ndarray]:
        local = self._skycoord2localcoord(coords)
        weight = self.weight_function(*local)
        data = self.data_function(*local)
        return weight[:, None], data

    def _skycoord2localcoord(self, skycoord: SkyCoord) -> tuple[np.ndarray, np.ndarray]:
        skycoord = _transform_to_frame(skycoord, self.frame)
        return skycoord.spherical.lon.rad, skycoord.spherical.lat.rad

    def query_scattered(self, observation, nside: int = 64) -> SourceField:
        lon, lat = hp.pix2ang(nside, np.arange(hp.nside2npix(nside)), lonlat=True)
        lon, lat = lon[lat > 0], lat[lat > 0]
        h_coords = SkyCoord(lon, lat, unit='deg', frame=observation.origin)
        weight, data = self._query_coords(h_coords)
        weight = weight * hp.nside2pixarea(nside) * u.radian**2

        return SourceField(
            coords=h_coords,
            weights=weight,
            spectral_data=data,
            spectral_grid=self._spectral_grid,
            radiance_field=True,
        )


class HEALPixSource(RadianceSource):
    """Diffuse source stored as a HEALPix map."""

    def __init__(
        self,
        frame,
        weight: u.Quantity,
        data: np.ndarray,
        spectral_grid: SpectralGrid,
        name: str = "",
    ) -> None:
        self.name = name or type(self).__name__
        self.frame = frame
        self.weight = weight
        self.data = data
        self._spectral_grid = spectral_grid

    def _query_coords(self, coords: SkyCoord) -> tuple[np.ndarray, np.ndarray]:
        skycoord = _transform_to_frame(coords, self.frame)
        lon, lat = skycoord.spherical.lon.deg, skycoord.spherical.lat.deg
        weight = hp.get_interp_val(self.weight, lon, lat, nest=True, lonlat=True)
        data = hp.get_interp_val(self.data, lon, lat, nest=True, lonlat=True)
        return np.atleast_2d(weight).T, np.atleast_2d(data).T

    def query_scattered(self, observation, nside: int = 64) -> SourceField:
        lon, lat = hp.pix2ang(nside, np.arange(hp.nside2npix(nside)), nest=True, lonlat=True)
        h_coords = SkyCoord(lon, lat, unit='deg', frame=self.frame).transform_to(observation.origin)

        weight = hp.ud_grade(self.weight.value, nside,
                             order_in='NESTED', order_out='NESTED') * self.weight.unit
        data = hp.ud_grade(self.data, nside, order_in='NESTED', order_out='NESTED')

        weight = np.atleast_2d(weight).T
        data = np.atleast_2d(data).T

        above_horizon = h_coords.alt.rad > 0
        weight = weight[above_horizon] * hp.nside2pixarea(nside) * u.radian**2

        return SourceField(
            coords=h_coords[above_horizon],
            weights=weight,
            spectral_data=data[above_horizon],
            spectral_grid=self._spectral_grid,
            radiance_field=True,
        )

    @classmethod
    def from_photometric_map(cls, frame, magnitude, color, spectral_library, name="") -> HEALPixSource:
        color_range = [np.nanmin(color[1]), np.nanmax(color[1])]
        color_grid = create_color_grid(magnitude[0], color[0], color_range, spectral_library)
        area_corr = hp.nside2pixarea(hp.npix2nside(magnitude[1].shape[0])) * u.radian**2
        return cls(frame, 10**(-0.4 * magnitude[1]) / area_corr, color[1], color_grid, name=name)
