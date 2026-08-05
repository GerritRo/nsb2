"""Extended source tests: EphemerisSource, HEALPixSource, to_map else branch, etc."""

from unittest.mock import patch

import astropy.units as u
import healpy as hp
import numpy as np
import pytest
from astropy.coordinates import EarthLocation, SkyCoord
from astropy.time import Time

from nsb2.core.dtypes import PixelRefs, SourceField
from nsb2.core.sources import (
    CatalogSource,
    EphemerisSource,
    HEALPixSource,
    LonLatSource,
)
from nsb2.core.spectral import SpectralGrid
from tests.conftest import make_bandpass, make_observation, make_spectral_grid


def make_day_observation(alt=60, az=180):
    """Observation during daytime — sun is above horizon."""
    from astropy.coordinates import AltAz

    loc = EarthLocation(lat=-23.27 * u.deg, lon=16.5 * u.deg, height=1800 * u.m)
    t = Time("2024-06-15T10:00:00")  # ~noon local time at lon=16.5°
    altaz = AltAz(obstime=t, location=loc)
    pointing = SkyCoord(alt=alt * u.deg, az=az * u.deg, frame=altaz)
    return pointing.skyoffset_frame()


# ---------------------------------------------------------------------------
# EphemerisSource
# ---------------------------------------------------------------------------


class TestEphemerisSource:
    @pytest.fixture
    def simple_source(self):
        """EphemerisSource for 'moon' with trivial weight/data functions."""
        sg = make_spectral_grid()

        def weight_fn(obstime):
            return np.ones(1) * u.dimensionless_unscaled

        def data_fn(obstime):
            return np.empty((1, 0))

        return EphemerisSource("moon", weight_fn, data_fn, sg, name="test_moon")

    def test_init_stores_attributes(self, simple_source):
        assert simple_source.body == "moon"
        assert simple_source.name == "test_moon"

    def test_spectral_grid_property(self, simple_source):
        sg = make_spectral_grid()
        src = EphemerisSource("moon", lambda t: np.ones(1), lambda t: np.empty((1, 0)), sg)
        assert src.spectral_grid is sg

    def test_query_body_returns_coords_weight_data(self, simple_source):
        obs = make_observation()
        coords, weight, data = simple_source._query_body(obs.origin.obstime)
        assert isinstance(coords, SkyCoord)
        assert weight.ndim == 2
        assert data.ndim == 2

    def test_query_direct_below_horizon_returns_empty_field(self, simple_source):
        """Sun is below the horizon at nighttime — returns empty SourceField."""
        sg = make_spectral_grid()

        def weight_fn(t):
            return np.ones(1) * u.dimensionless_unscaled

        def data_fn(t):
            return np.empty((1, 0))

        src = EphemerisSource("sun", weight_fn, data_fn, sg)
        obs = make_observation()  # nighttime — sun below horizon
        pix_coords = SkyCoord([0.0], [0.0], unit="rad", frame=obs)
        pixel_radii = np.array([0.1])
        field, refs = src.query_direct(obs, pix_coords, pixel_radii)
        assert isinstance(field, SourceField)
        assert len(field.coords) == 0

    def test_query_direct_above_horizon_returns_field(self, simple_source):
        """Sun is above the horizon at daytime — may return non-empty SourceField."""
        sg = make_spectral_grid()

        def weight_fn(t):
            return np.ones(1) * u.dimensionless_unscaled

        def data_fn(t):
            return np.empty((1, 0))

        src = EphemerisSource("sun", weight_fn, data_fn, sg)
        obs = make_day_observation()  # daytime — sun above horizon
        pix_coords = SkyCoord([0.0], [0.0], unit="rad", frame=obs)
        pixel_radii = np.array([np.pi / 2])  # large radius
        field, refs = src.query_direct(obs, pix_coords, pixel_radii)
        assert isinstance(field, SourceField)
        assert isinstance(refs, PixelRefs)

    def test_query_scattered_below_horizon_returns_empty(self, simple_source):
        """Sun below horizon → empty scattered field."""
        sg = make_spectral_grid()
        src = EphemerisSource(
            "sun",
            lambda t: np.ones(1) * u.dimensionless_unscaled,
            lambda t: np.empty((1, 0)),
            sg,
        )
        obs = make_observation()  # nighttime
        field = src.query_scattered(obs)
        assert isinstance(field, SourceField)
        assert len(field.coords) == 0

    def test_query_scattered_above_horizon_returns_field(self):
        """Sun above horizon → non-empty scattered field."""
        sg = make_spectral_grid()
        src = EphemerisSource(
            "sun",
            lambda t: np.ones(1) * u.dimensionless_unscaled,
            lambda t: np.empty((1, 0)),
            sg,
        )
        obs = make_day_observation()
        field = src.query_scattered(obs)
        assert isinstance(field, SourceField)
        assert len(field.coords) > 0

    def test_empty_field_returns_source_field(self, simple_source):
        field = simple_source._empty_field()
        assert isinstance(field, SourceField)
        assert len(field.coords) == 0

    def test_query_direct_with_moon(self, simple_source):
        """Moon can be above or below horizon — just ensure no crash."""
        obs = make_observation()
        pix_coords = SkyCoord([0.0, 0.01], [0.0, 0.0], unit="rad", frame=obs)
        pixel_radii = np.full(2, 0.1)
        field, refs = simple_source.query_direct(obs, pix_coords, pixel_radii)
        assert isinstance(field, SourceField)
        assert len(refs.indices) == 2


# ---------------------------------------------------------------------------
# HEALPixSource
# ---------------------------------------------------------------------------


def _make_healpix_source(nside=4):
    """Create a minimal HEALPixSource for testing."""
    npix = hp.nside2npix(nside)
    weight = np.ones(npix) * u.dimensionless_unscaled / u.radian**2
    data = np.zeros((1, npix))
    sg = make_spectral_grid()
    return HEALPixSource("icrs", weight, data, sg, name="test_healpix")


class TestHEALPixSource:
    def test_init_stores_name(self):
        src = _make_healpix_source()
        assert src.name == "test_healpix"

    def test_spectral_grid_property(self):
        src = _make_healpix_source()
        sg = make_spectral_grid()
        src2 = HEALPixSource("icrs", src.weight, src.data, sg)
        assert src2.spectral_grid is sg

    def test_query_direct_returns_field_and_refs(self):
        src = _make_healpix_source()
        obs = make_observation()
        pix_coords = SkyCoord([0.0, 0.01], [0.0, 0.0], unit="rad", frame=obs)
        pixel_radii = np.array([0.1, 0.1])
        field, refs = src.query_direct(obs, pix_coords, pixel_radii)
        assert isinstance(field, SourceField)
        assert field.radiance_field is True
        assert isinstance(refs, PixelRefs)

    def test_query_scattered_returns_field(self):
        src = _make_healpix_source()
        obs = make_observation()
        field = src.query_scattered(obs, nside=4)
        assert isinstance(field, SourceField)
        assert field.radiance_field is True

    def test_query_scattered_only_above_horizon(self):
        src = _make_healpix_source()
        obs = make_observation()
        field = src.query_scattered(obs, nside=4)
        # All returned coordinates should be above the horizon (alt > 0)
        assert np.all(field.coords.alt.rad > 0)

    def test_query_direct_shape(self):
        src = _make_healpix_source()
        obs = make_observation()
        n_pix = 3
        pix_coords = SkyCoord(np.zeros(n_pix), np.zeros(n_pix), unit="rad", frame=obs)
        pixel_radii = np.full(n_pix, 0.05)
        field, refs = src.query_direct(obs, pix_coords, pixel_radii)
        assert len(refs.indices) == n_pix


# ---------------------------------------------------------------------------
# CatalogSource — additional coverage
# ---------------------------------------------------------------------------


class TestCatalogSourceExtended:
    @pytest.fixture
    def catalog_with_data(self):
        """CatalogSource with non-zero spectral data columns (for to_map else branch)."""
        sg = make_spectral_grid(n_comp=3, n_wvl=20)
        n_stars = 12
        coords = SkyCoord(
            np.arange(n_stars) * 30.0 * u.deg, np.zeros(n_stars) * u.deg, frame="icrs"
        )
        weight = np.ones(n_stars) * u.dimensionless_unscaled
        data = np.random.rand(n_stars, 1) * 0.5  # 1 spectral param
        return CatalogSource(coords, weight, data, sg)

    def test_to_map_else_branch_with_data(self, catalog_with_data):
        """to_map with non-zero data columns exercises the else branch (lines 187-191)."""
        catalog_with_data.build_balltree()
        healpix = catalog_with_data.to_map(nside=8)
        assert isinstance(healpix, HEALPixSource)

    def test_apply_space_motion(self):
        """apply_space_motion updates coords."""
        sg = make_spectral_grid()
        coords = SkyCoord(
            ra=[10, 20] * u.deg,
            dec=[5, -5] * u.deg,
            frame="icrs",
            pm_ra_cosdec=[1.0, 0.0] * u.mas / u.yr,
            pm_dec=[0.5, 0.0] * u.mas / u.yr,
            obstime=Time("2000-01-01"),
        )
        weight = np.ones(2) * u.dimensionless_unscaled
        data = np.empty((2, 0))
        src = CatalogSource(coords, weight, data, sg)
        src.build_balltree()
        t_new = Time("2010-01-01")
        src.apply_space_motion(t_new)
        assert src.coords.obstime == t_new

    def test_skycoord2latlon_shape(self):
        sg = make_spectral_grid()
        coords = SkyCoord([10, 20, 30] * u.deg, [5, -5, 0] * u.deg, frame="icrs")
        weight = np.ones(3) * u.dimensionless_unscaled
        data = np.empty((3, 0))
        src = CatalogSource(coords, weight, data, sg)
        result = src._skycoord2latlon(coords)
        assert result.shape == (3, 2)

    @patch("nsb2.core.sources.create_color_grid")
    def test_from_photometric_catalog(self, mock_ccg):
        mock_ccg.return_value = make_spectral_grid()
        coords = SkyCoord([10, 20] * u.deg, [0, 5] * u.deg, frame="icrs")
        magnitude = [make_bandpass(), np.array([10.0, 11.0])]
        color = [[make_bandpass(), make_bandpass()], np.array([0.5, 0.5])]
        spec_lib = make_spectral_grid()
        result = CatalogSource.from_photometric_catalog(coords, magnitude, color, spec_lib)
        assert isinstance(result, CatalogSource)
        assert len(result.coords) == 2

    @patch("nsb2.core.sources.create_color_grid")
    def test_healpix_from_photometric_map(self, mock_ccg):
        mock_ccg.return_value = make_spectral_grid()
        nside = 4
        npix = hp.nside2npix(nside)
        magnitude = [make_bandpass(), np.zeros(npix)]
        color = [[make_bandpass(), make_bandpass()], np.zeros(npix)]
        spec_lib = make_spectral_grid()
        result = HEALPixSource.from_photometric_map("icrs", magnitude, color, spec_lib)
        assert isinstance(result, HEALPixSource)


# ---------------------------------------------------------------------------
# LonLatSource — additional coverage for _skycoord2localcoord with non-None frame
# ---------------------------------------------------------------------------


class TestLonLatSourceWithFrame:
    def test_query_scattered_with_galactic_frame(self):
        """LonLatSource with a non-None frame exercises _transform_to_frame."""
        sg = make_spectral_grid()

        def weight_fn(lon, lat):
            return np.ones(len(lon)) * u.dimensionless_unscaled

        def data_fn(lon, lat):
            return np.empty((len(lon), 0))

        src = LonLatSource("galactic", weight_fn, data_fn, sg)
        obs = make_observation()
        field = src.query_scattered(obs, nside=4)
        assert field.radiance_field is True
