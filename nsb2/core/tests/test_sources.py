"""Tests for :mod:`nsb2.core.sources`."""

import astropy.units as u
import numpy as np
import pytest
from astropy.coordinates import SkyCoord

from nsb2.conftest import make_spectral_grid
from nsb2.core.dtypes import PixelRefs, SourceField
from nsb2.core.sources import (
    CatalogSource,
    EphemerisSource,
    HEALPixSource,
    LonLatSource,
    _transform_to_frame,
)


class TestTransformToFrame:
    def test_none_frame_returns_input(self):
        sc = SkyCoord(10, 20, unit="deg", frame="icrs")
        assert _transform_to_frame(sc, None) is sc

    def test_icrs_to_galactic(self):
        sc = SkyCoord(0, 0, unit="deg", frame="icrs")
        assert _transform_to_frame(sc, "galactic").frame.name == "galactic"


class TestCatalogSource:
    @pytest.fixture
    def catalog(self):
        coords = SkyCoord(
            np.arange(10) * 10 * u.deg, np.zeros(10) * u.deg, frame="icrs"
        )
        src = CatalogSource(
            coords,
            np.ones(10) * u.dimensionless_unscaled,
            np.zeros((10, 0)),
            make_spectral_grid(),
        )
        src.build_balltree()
        return src

    def test_name_defaults_to_class_name(self, catalog):
        assert catalog.name == "CatalogSource"

    def test_one_dimensional_weight_is_promoted(self, catalog):
        assert catalog.weight.ndim == 2

    def test_query_direct_returns_field_and_refs(self, catalog, observation):
        pix_coords = SkyCoord(
            [0, 0.01, -0.01], [0, 0.01, -0.01], unit="rad", frame=observation
        )
        field, refs = catalog.query_direct(
            observation, pix_coords, np.full(3, np.deg2rad(5))
        )
        assert isinstance(field, SourceField)
        assert isinstance(refs, PixelRefs)
        assert field.radiance_field is False
        assert len(refs.indices) == 3

    def test_query_direct_indices_point_into_field(self, catalog, observation):
        pix_coords = SkyCoord([0, 0.01], [0, 0.01], unit="rad", frame=observation)
        field, refs = catalog.query_direct(
            observation, pix_coords, np.full(2, np.deg2rad(20))
        )
        for idx in refs.indices:
            assert np.all(idx < len(field.coords))

    def test_query_scattered_returns_field(self, catalog, observation):
        field = catalog.query_scattered(observation)
        assert isinstance(field, SourceField)
        assert field.radiance_field is False

    def test_getitem_slices(self, catalog):
        assert len(catalog[:3].coords) == 3

    def test_getitem_keeps_name(self, catalog):
        assert catalog[:3].name == catalog.name

    def test_to_map(self, catalog):
        assert isinstance(catalog.to_map(nside=8), HEALPixSource)

    def test_to_map_does_not_modify_source(self, catalog):
        before = catalog.weight.copy()
        catalog.to_map(nside=8)
        np.testing.assert_array_equal(catalog.weight.value, before.value)

    def test_to_map_conserves_total_brightness(self, catalog):
        healpix = catalog.to_map(nside=8)
        pixel_area = 4 * np.pi / len(healpix.weight[0])
        total = (healpix.weight[0] * pixel_area).sum()
        assert total.value == pytest.approx(catalog.weight[:, 0].sum().value, rel=1e-6)


class TestLonLatSource:
    @pytest.fixture
    def diffuse(self):
        return LonLatSource(
            None,
            lambda lon, lat: np.ones(len(lon)) * u.dimensionless_unscaled,
            lambda lon, lat: np.empty((len(lon), 0)),
            make_spectral_grid(),
        )

    def test_query_direct(self, diffuse, observation):
        pix_coords = SkyCoord([0.01, 0.02], [0.01, 0.02], unit="rad", frame=observation)
        field, refs = diffuse.query_direct(
            observation, pix_coords, np.array([0.1, 0.1])
        )
        assert isinstance(field, SourceField)
        assert len(refs.indices) == 2

    def test_query_direct_maps_one_source_per_pixel(self, diffuse, observation):
        pix_coords = SkyCoord([0.01, 0.02], [0.01, 0.02], unit="rad", frame=observation)
        _, refs = diffuse.query_direct(observation, pix_coords, np.array([0.1, 0.1]))
        assert [idx.tolist() for idx in refs.indices] == [[0], [1]]

    def test_query_scattered(self, diffuse, observation):
        field = diffuse.query_scattered(observation, nside=8)
        assert isinstance(field, SourceField)
        assert field.radiance_field is True

    def test_query_scattered_covers_upper_hemisphere(self, diffuse, observation):
        field = diffuse.query_scattered(observation, nside=8)
        assert len(field.coords) == pytest.approx(12 * 8**2 / 2, rel=0.05)


class TestEphemerisSource:
    @pytest.fixture
    def moon_like(self):
        return EphemerisSource(
            "moon",
            lambda t: np.array([1.0]) * u.dimensionless_unscaled,
            lambda t: np.array([0.5]),
            make_spectral_grid(),
        )

    def test_name_defaults_to_body(self, moon_like):
        assert moon_like.name == "moon"

    def test_query_direct_returns_pixel_refs_for_every_pixel(
        self, moon_like, observation
    ):
        pix_coords = SkyCoord([0.0, 0.01], [0.0, 0.01], unit="rad", frame=observation)
        _, refs = moon_like.query_direct(observation, pix_coords, np.array([0.1, 0.1]))
        assert len(refs.indices) == 2

    def test_query_scattered_returns_field(self, moon_like, observation):
        field = moon_like.query_scattered(observation)
        assert isinstance(field, SourceField)

    def test_empty_field_when_below_horizon(self, moon_like, observation):
        """A body below the horizon must yield an empty field, not an error."""
        below = EphemerisSource(
            "sun",
            lambda t: np.array([1.0]) * u.dimensionless_unscaled,
            lambda t: np.array([0.5]),
            make_spectral_grid(),
        )
        field = below.query_scattered(observation)
        # The test observation is at local night, so the Sun is down.
        assert field.spectral_data.shape[0] == 0


class TestHEALPixSource:
    @pytest.fixture
    def healpix(self):
        npix = 12 * 8**2
        return HEALPixSource(
            "icrs",
            np.ones(npix) * u.dimensionless_unscaled,
            np.zeros(npix),
            make_spectral_grid(),
        )

    def test_query_direct_returns_one_source_per_pixel(self, healpix, observation):
        pix_coords = SkyCoord([0.0, 0.01], [0.0, 0.01], unit="rad", frame=observation)
        field, refs = healpix.query_direct(
            observation, pix_coords, np.array([0.1, 0.1])
        )
        assert field.radiance_field is True
        assert len(refs.indices) == 2

    def test_query_scattered_only_returns_visible_sky(self, healpix, observation):
        field = healpix.query_scattered(observation, nside=8)
        assert np.all(field.coords.alt.rad > 0)

    def test_query_scattered_resamples_to_requested_nside(self, healpix, observation):
        field = healpix.query_scattered(observation, nside=4)
        assert len(field.coords) == pytest.approx(12 * 4**2 / 2, rel=0.1)
