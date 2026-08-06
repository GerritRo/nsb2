"""Sky brightness sources.

A source knows where its emitters are on the sky and how bright they are; it
does not know anything about the telescope or the atmosphere.  Sources answer
two kinds of query.  :meth:`Source.query_direct` returns only the emitters
that fall inside the instrument's pixels, which is what the direct light path
needs.  :meth:`Source.query_scattered` returns everything above the horizon,
because any of it can be scattered into the field of view.

Three shapes of source cover the models in this package: catalogues of
resolved point sources, diffuse fields that vary smoothly across the sky, and
solar system bodies whose position depends on time.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Callable

import astropy.units as u
import healpy as hp
import numpy as np
from astropy.coordinates import SkyCoord, get_body
from sklearn.neighbors import BallTree

from nsb2.core.dtypes import PixelRefs, SourceField
from nsb2.core.photometry import create_color_grid
from nsb2.core.spectral import SpectralGrid

__all__ = [
    "CatalogSource",
    "EphemerisSource",
    "HEALPixSource",
    "LonLatSource",
    "RadianceSource",
    "Source",
]


logger = logging.getLogger(__name__)

#: Default HEALPix resolution for discretising the sky in scattering queries.
DEFAULT_NSIDE = 64


def _transform_to_frame(skycoord: SkyCoord, frame) -> SkyCoord:
    """Transform coordinates to a frame, or pass them through.

    Parameters
    ----------
    skycoord : astropy.coordinates.SkyCoord
        Coordinates to transform.
    frame : str or astropy.coordinates.BaseCoordinateFrame or None
        Target frame.  `None` means the source is defined directly in the
        observation frame, e.g. airglow, which is fixed to the local horizon.

    Returns
    -------
    astropy.coordinates.SkyCoord
        The transformed coordinates, or ``skycoord`` itself if ``frame`` is
        `None`.
    """
    if frame is not None:
        skycoord = skycoord.transform_to(frame)
    return skycoord


class Source(ABC):
    """Base class for sky brightness sources.

    Attributes
    ----------
    name : str
        Human-readable identifier, carried through to
        :class:`nsb2.core.dtypes.Prediction` so that the contributions of
        several sources can be told apart.
    """

    name: str = ""

    @property
    @abstractmethod
    def spectral_grid(self) -> SpectralGrid:
        """nsb2.core.spectral.SpectralGrid: Spectra of this source's emitters."""
        ...

    @abstractmethod
    def query_direct(
        self,
        observation,
        pixel_coords: SkyCoord,
        pixel_radii: np.ndarray,
    ) -> tuple[SourceField, PixelRefs]:
        """Find the emitters seen directly by each pixel.

        Parameters
        ----------
        observation : astropy.coordinates.BaseCoordinateFrame
            The telescope pointing frame, carrying the observation time and
            location on its ``origin``.
        pixel_coords : astropy.coordinates.SkyCoord
            Pixel centre positions, shape ``(N_pix,)``.
        pixel_radii : numpy.ndarray
            Search radius per pixel in radians, shape ``(N_pix,)``.

        Returns
        -------
        field : nsb2.core.dtypes.SourceField
            The emitters found.
        pixel_refs : nsb2.core.dtypes.PixelRefs
            Which of them contribute to which pixel, with ``weights`` left
            unset for the instrument to fill in.
        """
        ...

    @abstractmethod
    def query_scattered(self, observation, nside: int = DEFAULT_NSIDE) -> SourceField:
        """Find every emitter above the horizon.

        Parameters
        ----------
        observation : astropy.coordinates.BaseCoordinateFrame
            The telescope pointing frame.
        nside : int, optional
            HEALPix resolution used to discretise diffuse emission.  Default
            is :data:`DEFAULT_NSIDE`.

        Returns
        -------
        nsb2.core.dtypes.SourceField
            All emitters that can scatter light into the field of view.
        """
        ...


class RadianceSource(Source):
    """Base class for diffuse sources that map one-to-one onto pixels.

    A radiance varies smoothly enough across the sky that it can be evaluated
    once per pixel centre, so the source-to-pixel assignment is trivial and
    each pixel gets exactly one emitter.

    Subclasses implement :meth:`_query_coords` and
    :meth:`Source.query_scattered`.
    """

    _spectral_grid: SpectralGrid

    @property
    def spectral_grid(self) -> SpectralGrid:
        """nsb2.core.spectral.SpectralGrid: Spectra of this source's emitters."""
        return self._spectral_grid

    @abstractmethod
    def _query_coords(self, coords: SkyCoord) -> tuple[np.ndarray, np.ndarray]:
        """Evaluate the radiance at the given sky positions.

        Parameters
        ----------
        coords : astropy.coordinates.SkyCoord
            Positions to evaluate at, shape ``(N,)``.

        Returns
        -------
        weight : astropy.units.Quantity
            Brightness weights, shape ``(N, C_w)``.
        data : numpy.ndarray
            Spectral grid coordinates, shape ``(N, D)``.
        """
        ...

    def query_direct(
        self,
        observation,
        pixel_coords: SkyCoord,
        pixel_radii: np.ndarray,
    ) -> tuple[SourceField, PixelRefs]:
        """Evaluate the radiance at each pixel centre.

        See :meth:`Source.query_direct` for the parameters and returns.
        ``pixel_radii`` is unused, since a radiance is sampled pointwise.
        """
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
    """Resolved point sources from a photometric catalogue.

    Positions are indexed by a ball tree on the unit sphere, so that the
    per-pixel cone searches stay tractable for catalogues of millions of
    stars.  :meth:`build_balltree` must be called before querying.

    Parameters
    ----------
    coords : astropy.coordinates.SkyCoord
        Source positions, shape ``(N,)``.
    weight : astropy.units.Quantity
        Brightness weights, shape ``(N,)`` or ``(N, C_w)``.
    data : numpy.ndarray
        Spectral grid coordinates, shape ``(N,)`` or ``(N, D)``.
    spectral_grid : nsb2.core.spectral.SpectralGrid
        Grid the spectral coordinates refer to.
    name : str, optional
        Identifier for the source.  Defaults to the class name.

    See Also
    --------
    from_photometric_catalog : Build one from magnitudes and colours.
    """

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
        """nsb2.core.spectral.SpectralGrid: Spectra of this source's emitters."""
        return self._spectral_grid

    def build_balltree(self) -> None:
        """Build the spatial index used by the query methods.

        Must be called once before :meth:`query_direct` or
        :meth:`query_scattered`, and again after the positions change.
        """
        logger.debug(
            "building ball tree for %d sources in %s", len(self.coords), self.name
        )
        self.balltree = BallTree(self._skycoord2latlon(self.coords), metric="haversine")

    def apply_space_motion(self, time) -> None:
        """Propagate the catalogue positions to a new epoch.

        Rebuilds the spatial index afterwards.  Unlike the query methods,
        this mutates the source in place.

        Parameters
        ----------
        time : astropy.time.Time
            Epoch to propagate the positions to.
        """
        self.coords = self.coords.apply_space_motion(new_obstime=time)
        self.build_balltree()

    def query_direct(
        self,
        observation,
        pixel_coords: SkyCoord,
        pixel_radii: np.ndarray,
    ) -> tuple[SourceField, PixelRefs]:
        """Cone-search the catalogue around each pixel centre.

        See :meth:`Source.query_direct` for the parameters and returns.
        """
        sky_coords = pixel_coords.transform_to(observation.origin)
        refs = self.balltree.query_radius(
            self._skycoord2latlon(sky_coords), pixel_radii
        )
        unique_indices, inverse_indices = np.unique(
            np.concatenate(refs), return_inverse=True
        )
        split_indices = np.cumsum([len(ref) for ref in refs])[:-1]
        new_refs = np.split(inverse_indices, split_indices)

        s_coords = self.coords[unique_indices]

        field = SourceField(
            coords=s_coords.transform_to(observation.origin),
            weights=self.weight[unique_indices],
            spectral_data=self.data[unique_indices],
            spectral_grid=self._spectral_grid,
        )
        pixel_refs = PixelRefs(indices=new_refs, weights=None)
        return field, pixel_refs

    def query_scattered(self, observation, nside: int = DEFAULT_NSIDE) -> SourceField:
        """Select every catalogue source above the horizon.

        See :meth:`Source.query_scattered` for the parameters and returns.
        ``nside`` is unused, since the catalogue is already discrete.
        """
        zenith = SkyCoord(0, 90, unit="deg", frame=observation.origin)
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
        """Bin the catalogue into a HEALPix radiance map.

        Faint stars are far too numerous to trace individually but too dim to
        resolve, so binning them into a map and treating them as diffuse
        emission is both cheaper and closer to what the instrument sees.
        Spectral coordinates are averaged with brightness weighting, and
        missing ones are replaced by the catalogue mean.

        Parameters
        ----------
        nside : int
            HEALPix resolution of the output map.

        Returns
        -------
        HEALPixSource
            The binned map, with brightness converted to a radiance by
            dividing by the pixel solid angle.  This source is not modified.
        """
        npix = hp.nside2npix(nside)
        hp_inds = hp.ang2pix(
            nside,
            self.coords.spherical.lon.deg,
            self.coords.spherical.lat.deg,
            nest=True,
            lonlat=True,
        )
        weight = np.vstack(
            [
                np.bincount(hp_inds, self.weight[:, i], npix)
                for i in range(self.weight.shape[1])
            ]
        )
        if self.data.shape[1] == 0:
            data = np.empty((0, npix))
        else:
            data = np.where(
                np.isnan(self.data), np.nanmean(self.data, axis=0), self.data
            )
            # Cells with no sources have zero total weight; the resulting nan
            # is the intended "no spectral information here" value.
            with np.errstate(invalid="ignore"):
                data = np.vstack(
                    [
                        np.bincount(hp_inds, data[:, i] * self.weight[:, i], npix)
                        / np.bincount(hp_inds, self.weight[:, i], npix)
                        for i in range(data.shape[1])
                    ]
                )
        area_corr = hp.nside2pixarea(nside) * u.radian**2
        return HEALPixSource(
            self.frame, weight / area_corr, data, self._spectral_grid, name=self.name
        )

    def __getitem__(self, item) -> CatalogSource:
        """Select a subset of the catalogue.

        Parameters
        ----------
        item : slice or array_like
            Anything accepted as a numpy index.

        Returns
        -------
        CatalogSource
            A new source holding the selected entries.  The spatial index is
            not carried over; call :meth:`build_balltree` before querying.
        """
        return CatalogSource(
            self.coords[item],
            self.weight[item],
            self.data[item],
            self._spectral_grid,
            name=self.name,
        )

    def _skycoord2latlon(self, skycoord: SkyCoord) -> np.ndarray:
        """Convert coordinates to the ``(lat, lon)`` radians the ball tree expects."""
        skycoord = _transform_to_frame(skycoord, self.frame)
        return np.vstack([skycoord.spherical.lat.rad, skycoord.spherical.lon.rad]).T

    @classmethod
    def from_photometric_catalog(
        cls, coords, magnitude, color, spectral_library, name=""
    ) -> CatalogSource:
        """Build a catalogue source from magnitudes and colour indices.

        Parameters
        ----------
        coords : astropy.coordinates.SkyCoord
            Source positions, shape ``(N,)``.
        magnitude : list
            ``[bandpass, values]``: the passband the magnitudes are measured
            in, and the magnitude of each source.
        color : list
            ``[[blue_band, red_band], values]``: the two passbands defining
            the colour index, and the index of each source.
        spectral_library : nsb2.core.spectral.SpectralGrid
            Template spectra, e.g. from
            :func:`nsb2.core.photometry.PicklesTRDSAtlas1998`.
        name : str, optional
            Identifier for the source.  Defaults to the class name.

        Returns
        -------
        CatalogSource
            Source whose spectra are interpolated by colour index.
        """
        color_range = [np.nanmin(color[1]), np.nanmax(color[1])]
        color_grid = create_color_grid(
            magnitude[0], color[0], color_range, spectral_library
        )
        return cls(
            coords,
            10 ** (-0.4 * magnitude[1]) * u.dimensionless_unscaled,
            color[1],
            color_grid,
            name=name,
        )


class EphemerisSource(Source):
    """A solar system body, whose position depends on the observation time.

    Parameters
    ----------
    body : str
        Body name understood by :func:`astropy.coordinates.get_body`, e.g.
        ``"moon"``.
    weight_function : callable
        Called with the observation time; returns the brightness weights.
    data_function : callable
        Called with the observation time; returns the spectral grid
        coordinates, e.g. the lunar phase angle.
    spectral_grid : nsb2.core.spectral.SpectralGrid
        Grid the spectral coordinates refer to.
    name : str, optional
        Identifier for the source.  Defaults to ``body``.
    """

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
        """nsb2.core.spectral.SpectralGrid: Spectra of this source's emitters."""
        return self._spectral_grid

    def _query_body(self, obstime) -> tuple[SkyCoord, np.ndarray, np.ndarray]:
        """Locate the body and evaluate its brightness at ``obstime``."""
        body_coords = get_body(self.body, obstime)
        weight = self.weight_function(obstime)
        data = self.data_function(obstime)
        if body_coords.isscalar:
            body_coords = body_coords.reshape(1)
        return body_coords, np.atleast_2d(weight), np.atleast_2d(data)

    def _empty_field(self) -> SourceField:
        """Return a field with no sources, for when the body is below the horizon."""
        return SourceField(
            coords=SkyCoord([], [], unit="deg", frame="altaz"),
            weights=np.empty((0, 1)) * u.dimensionless_unscaled,
            spectral_data=np.empty((0, 1)),
            spectral_grid=self._spectral_grid,
        )

    def query_direct(
        self,
        observation,
        pixel_coords: SkyCoord,
        pixel_radii: np.ndarray,
    ) -> tuple[SourceField, PixelRefs]:
        """Find the pixels the body falls into, if it is above the horizon.

        See :meth:`Source.query_direct` for the parameters and returns.
        """
        b_coord, weight, data = self._query_body(observation.origin.obstime)
        b_coord = b_coord.transform_to(observation.origin)

        above_horizon = b_coord.alt.rad > 0
        if not np.any(above_horizon):
            logger.debug("%s is below the horizon", self.name)
            n_pix = len(pixel_coords)
            pixel_refs = PixelRefs(
                indices=[np.array([], dtype=int) for _ in range(n_pix)],
                weights=None,
            )
            return self._empty_field(), pixel_refs

        body_visible = b_coord[above_horizon]
        field = SourceField(
            coords=body_visible,
            weights=weight[above_horizon],
            spectral_data=data[above_horizon],
            spectral_grid=self._spectral_grid,
        )

        sky_coords = pixel_coords.transform_to(observation.origin)
        indices = []
        for i in range(len(pixel_coords)):
            seps = sky_coords[i].separation(body_visible).rad
            indices.append(np.where(seps < pixel_radii[i])[0].astype(int))

        return field, PixelRefs(indices=indices, weights=None)

    def query_scattered(self, observation, nside: int = DEFAULT_NSIDE) -> SourceField:
        """Return the body if it is above the horizon, else an empty field.

        See :meth:`Source.query_scattered` for the parameters and returns.
        ``nside`` is unused, since the body is a single point source.
        """
        b_coord, weight, data = self._query_body(observation.origin.obstime)
        b_coord = b_coord.transform_to(observation.origin)

        above_horizon = b_coord.alt.rad > 0
        if not np.any(above_horizon):
            logger.debug("%s is below the horizon", self.name)
            return self._empty_field()

        return SourceField(
            coords=b_coord[above_horizon],
            weights=weight[above_horizon],
            spectral_data=data[above_horizon],
            spectral_grid=self._spectral_grid,
        )


class LonLatSource(RadianceSource):
    """Diffuse source defined by analytic functions of longitude and latitude.

    Parameters
    ----------
    frame : str or astropy.coordinates.BaseCoordinateFrame or None
        Frame the model functions are defined in.  `None` means the
        observation's own frame, as for airglow, which is fixed to the local
        horizon.
    weight_function : callable
        Called with ``(lon, lat)`` in radians; returns brightness weights.
    data_function : callable
        Called with ``(lon, lat)`` in radians; returns spectral grid
        coordinates.
    spectral_grid : nsb2.core.spectral.SpectralGrid
        Grid the spectral coordinates refer to.
    name : str, optional
        Identifier for the source.  Defaults to the class name.
    """

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
        """Evaluate the model functions at the given sky positions."""
        local = self._skycoord2localcoord(coords)
        weight = self.weight_function(*local)
        data = self.data_function(*local)
        return weight[:, None], data

    def _skycoord2localcoord(self, skycoord: SkyCoord) -> tuple[np.ndarray, np.ndarray]:
        """Convert coordinates to ``(lon, lat)`` radians in the model's frame."""
        skycoord = _transform_to_frame(skycoord, self.frame)
        return skycoord.spherical.lon.rad, skycoord.spherical.lat.rad

    def query_scattered(self, observation, nside: int = DEFAULT_NSIDE) -> SourceField:
        """Sample the model over a HEALPix grid of the visible hemisphere.

        See :meth:`Source.query_scattered` for the parameters and returns.
        """
        lon, lat = hp.pix2ang(nside, np.arange(hp.nside2npix(nside)), lonlat=True)
        lon, lat = lon[lat > 0], lat[lat > 0]
        h_coords = SkyCoord(lon, lat, unit="deg", frame=observation.origin)
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
    """Diffuse source stored as a HEALPix radiance map.

    Parameters
    ----------
    frame : str or astropy.coordinates.BaseCoordinateFrame
        Frame the map is defined in.
    weight : astropy.units.Quantity
        Radiance per HEALPix cell, in nested ordering.
    data : numpy.ndarray
        Spectral grid coordinates per cell, in nested ordering.
    spectral_grid : nsb2.core.spectral.SpectralGrid
        Grid the spectral coordinates refer to.
    name : str, optional
        Identifier for the source.  Defaults to the class name.

    See Also
    --------
    CatalogSource.to_map : Build one by binning a point source catalogue.
    from_photometric_map : Build one from magnitude and colour maps.
    """

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
        """Interpolate the map at the given sky positions."""
        skycoord = _transform_to_frame(coords, self.frame)
        lon, lat = skycoord.spherical.lon.deg, skycoord.spherical.lat.deg
        weight = hp.get_interp_val(self.weight, lon, lat, nest=True, lonlat=True)
        data = hp.get_interp_val(self.data, lon, lat, nest=True, lonlat=True)
        return np.atleast_2d(weight).T, np.atleast_2d(data).T

    def query_scattered(self, observation, nside: int = DEFAULT_NSIDE) -> SourceField:
        """Resample the map onto the visible hemisphere at the given resolution.

        See :meth:`Source.query_scattered` for the parameters and returns.
        """
        lon, lat = hp.pix2ang(
            nside, np.arange(hp.nside2npix(nside)), nest=True, lonlat=True
        )
        h_coords = SkyCoord(lon, lat, unit="deg", frame=self.frame).transform_to(
            observation.origin
        )

        weight = (
            hp.ud_grade(self.weight.value, nside, order_in="NESTED", order_out="NESTED")
            * self.weight.unit
        )
        data = hp.ud_grade(self.data, nside, order_in="NESTED", order_out="NESTED")

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
    def from_photometric_map(
        cls, frame, magnitude, color, spectral_library, name=""
    ) -> HEALPixSource:
        """Build a HEALPix source from magnitude and colour maps.

        Parameters
        ----------
        frame : str or astropy.coordinates.BaseCoordinateFrame
            Frame the maps are defined in.
        magnitude : list
            ``[bandpass, map]``: the passband the magnitudes are measured in,
            and the integrated magnitude per HEALPix cell.
        color : list
            ``[[blue_band, red_band], map]``: the two passbands defining the
            colour index, and the index per cell.
        spectral_library : nsb2.core.spectral.SpectralGrid
            Template spectra, e.g. from
            :func:`nsb2.core.photometry.PicklesTRDSAtlas1998`.
        name : str, optional
            Identifier for the source.  Defaults to the class name.

        Returns
        -------
        HEALPixSource
            Source whose spectra are interpolated by colour index, with
            brightness converted to a radiance by dividing by the cell solid
            angle.
        """
        color_range = [np.nanmin(color[1]), np.nanmax(color[1])]
        color_grid = create_color_grid(
            magnitude[0], color[0], color_range, spectral_library
        )
        area_corr = hp.nside2pixarea(hp.npix2nside(magnitude[1].shape[0])) * u.radian**2
        return cls(
            frame,
            10 ** (-0.4 * magnitude[1]) / area_corr,
            color[1],
            color_grid,
            name=name,
        )
