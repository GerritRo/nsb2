import astropy.units as u
import numpy as np
import pytest
from astropy.coordinates import SkyCoord
from astropy.time import Time

from nsb2.conftest import make_spectral_grid
from nsb2.core.dtypes import PixelRefs, SourceField
from nsb2.core.sources import (
    CatalogSource,
    EphemerisSource,
    HEALPixSource,
    LonLatSource,
    _transform_to_frame,
)
from nsb2.core.spectral import SpectralGrid


def _coloured_grid(n_colors=5, n_wvl=20, n_comp=2):
    """A spectral grid with one parameter axis, so spectral_data is non-empty."""
    colors = np.linspace(-1, 1, n_colors)
    wvl = np.linspace(300, 700, n_wvl) * u.nm
    flx = np.ones((n_colors, n_wvl, n_comp)) * u.erg / u.s / u.cm**2 / u.nm
    return SpectralGrid([colors], wvl, flx)


def _moon_like():
    return EphemerisSource(
        "moon",
        lambda t: np.array([1.0]) * u.dimensionless_unscaled,
        lambda t: np.array([0.5]),
        make_spectral_grid(),
    )


class TestTransformToFrame:
    def test_passes_none_through_and_transforms_otherwise(self):
        sc = SkyCoord(10, 20, unit="deg", frame="icrs")
        assert _transform_to_frame(sc, None) is sc
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

    def test_queries_return_a_field_and_indices_into_it(self, catalog, observation):
        pix_coords = SkyCoord(
            [0, 0.01, -0.01], [0, 0.01, -0.01], unit="rad", frame=observation
        )
        field, refs = catalog.query_direct(
            observation, pix_coords, np.full(3, np.deg2rad(20))
        )
        assert isinstance(field, SourceField)
        assert isinstance(refs, PixelRefs)
        assert field.radiance_field is False
        assert len(refs.indices) == 3
        for idx in refs.indices:
            assert np.all(idx < len(field.coords))

        scattered = catalog.query_scattered(observation)
        assert isinstance(scattered, SourceField)
        assert scattered.radiance_field is False

    def test_getitem_selects_a_subset_and_keeps_the_metadata(self, catalog):
        assert catalog.name == "CatalogSource"
        assert catalog.weight.ndim == 2, "a 1-D weight is promoted to (N, 1)"
        subset = catalog[:3]
        assert len(subset.coords) == 3
        assert subset.name == catalog.name

    def test_to_map_conserves_total_brightness(self, catalog):
        before = catalog.weight.copy()
        healpix = catalog.to_map(nside=8)
        assert isinstance(healpix, HEALPixSource)

        pixel_area = 4 * np.pi / len(healpix.weight[0])
        total = (healpix.weight[0] * pixel_area).sum()
        assert total.value == pytest.approx(catalog.weight[:, 0].sum().value, rel=1e-6)
        np.testing.assert_array_equal(catalog.weight.value, before.value)

    def test_to_map_averages_the_colour_of_each_cell(self):
        """The non-empty-data arm weights each colour by source brightness.

        Cells with no sources divide zero by zero, and the resulting nan is
        the intended "no spectral information here" value.  A source whose
        own colour is nan takes the catalogue mean before averaging.
        """
        rng = np.random.default_rng(3)
        coords = SkyCoord(
            rng.uniform(0, 360, 20) * u.deg,
            rng.uniform(-60, 60, 20) * u.deg,
            frame="icrs",
        )
        source = CatalogSource(
            coords,
            np.ones(20) * u.dimensionless_unscaled,
            rng.uniform(-1, 1, 20),
            _coloured_grid(),
        )
        before = source.data.copy()

        healpix = source.to_map(nside=4)
        assert healpix.data.shape == (1, 12 * 4**2)
        occupied = healpix.data[np.isfinite(healpix.data)]
        assert len(occupied) > 0
        assert np.all((occupied >= -1) & (occupied <= 1))
        assert np.any(np.isnan(source.to_map(nside=16).data))
        np.testing.assert_array_equal(source.data, before)

        with_gap = CatalogSource(
            SkyCoord([10, 11, 12] * u.deg, [0, 0, 0] * u.deg, frame="icrs"),
            np.ones(3) * u.dimensionless_unscaled,
            np.array([0.0, np.nan, 1.0]),
            _coloured_grid(),
        )
        filled = with_gap.to_map(nside=2).data
        assert len(filled[np.isfinite(filled)]) > 0

    def test_apply_space_motion_moves_the_stars_and_rebuilds_the_index(self):
        coords = SkyCoord(
            ra=[10.0, 200.0] * u.deg,
            dec=[20.0, -30.0] * u.deg,
            pm_ra_cosdec=[500.0, -500.0] * u.mas / u.yr,
            pm_dec=[500.0, -500.0] * u.mas / u.yr,
            distance=[100.0, 200.0] * u.pc,
            obstime=Time("2000-01-01"),
            frame="icrs",
        )
        source = CatalogSource(
            coords,
            np.ones(2) * u.dimensionless_unscaled,
            np.zeros((2, 0)),
            make_spectral_grid(),
        )
        source.build_balltree()
        before = source.coords.ra.deg.copy()
        original_tree = source.balltree

        source.apply_space_motion(Time("2020-01-01"))

        moved = np.abs(source.coords.ra.deg - before)
        assert np.all(moved > 0)
        # 500 mas/yr over 20 years is about 10 arcsec, i.e. ~0.003 degrees.
        assert np.all(moved < 0.05)
        assert source.balltree is not original_tree


class TestLonLatSource:
    @pytest.fixture
    def diffuse(self):
        return LonLatSource(
            None,
            lambda lon, lat: np.ones(len(lon)) * u.dimensionless_unscaled,
            lambda lon, lat: np.empty((len(lon), 0)),
            make_spectral_grid(),
        )

    def test_maps_one_source_per_pixel_and_samples_the_visible_hemisphere(
        self, diffuse, observation
    ):
        pix_coords = SkyCoord([0.01, 0.02], [0.01, 0.02], unit="rad", frame=observation)
        field, refs = diffuse.query_direct(
            observation, pix_coords, np.array([0.1, 0.1])
        )
        assert isinstance(field, SourceField)
        assert [idx.tolist() for idx in refs.indices] == [[0], [1]]

        scattered = diffuse.query_scattered(observation, nside=8)
        assert scattered.radiance_field is True
        assert len(scattered.coords) == pytest.approx(12 * 8**2 / 2, rel=0.05)


class TestEphemerisSource:
    def test_queries_a_body_above_the_horizon(self, observation):
        source = _moon_like()
        assert source.name == "moon", "the name defaults to the body"
        assert source.spectral_grid is source._spectral_grid

        pix_coords = SkyCoord([0.0, 0.01], [0.0, 0.01], unit="rad", frame=observation)
        _, refs = source.query_direct(observation, pix_coords, np.array([0.1, 0.1]))
        assert len(refs.indices) == 2

        field = source.query_scattered(observation)
        assert isinstance(field, SourceField)
        # A scalar body position must be promoted to a length-one array.
        assert field.coords.isscalar is False

    def test_body_below_the_horizon_gives_an_empty_field(self, observation):
        """A body below the horizon must yield an empty field, not an error."""
        below = EphemerisSource(
            "sun",
            lambda t: np.array([1.0]) * u.dimensionless_unscaled,
            lambda t: np.array([0.5]),
            make_spectral_grid(),
        )
        # The test observation is at local night, so the Sun is down.
        assert below.query_scattered(observation).spectral_data.shape[0] == 0


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

    def test_interpolates_per_pixel_and_resamples_the_visible_sky(
        self, healpix, observation
    ):
        pix_coords = SkyCoord([0.0, 0.01], [0.0, 0.01], unit="rad", frame=observation)
        field, refs = healpix.query_direct(
            observation, pix_coords, np.array([0.1, 0.1])
        )
        assert field.radiance_field is True
        assert len(refs.indices) == 2

        scattered = healpix.query_scattered(observation, nside=4)
        assert np.all(scattered.coords.alt.rad > 0)
        assert len(scattered.coords) == pytest.approx(12 * 4**2 / 2, rel=0.1)
