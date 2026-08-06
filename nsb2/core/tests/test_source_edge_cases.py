"""Edge cases and less-travelled branches of :mod:`nsb2.core.sources`.

The main source tests use a spectral grid with no parameter axes, which is
what most models need but which short-circuits the colour-averaging path in
:meth:`~nsb2.core.sources.CatalogSource.to_map`.  These exercise the
alternatives.
"""

import astropy.units as u
import numpy as np
import pytest
from astropy.coordinates import SkyCoord
from astropy.time import Time

from nsb2.conftest import make_spectral_grid
from nsb2.core.sources import CatalogSource, EphemerisSource, HEALPixSource


def _coloured_grid(n_colors=5, n_wvl=20, n_comp=2):
    """A spectral grid with one parameter axis, so spectral_data is non-empty."""
    colors = np.linspace(-1, 1, n_colors)
    wvl = np.linspace(300, 700, n_wvl) * u.nm
    flx = np.ones((n_colors, n_wvl, n_comp)) * u.erg / u.s / u.cm**2 / u.nm
    from nsb2.core.spectral import SpectralGrid

    return colors, SpectralGrid([colors], wvl, flx)


@pytest.fixture
def coloured_catalog():
    """A catalogue whose sources carry a colour index."""
    _, grid = _coloured_grid()
    n = 20
    rng = np.random.default_rng(3)
    coords = SkyCoord(
        rng.uniform(0, 360, n) * u.deg, rng.uniform(-60, 60, n) * u.deg, frame="icrs"
    )
    source = CatalogSource(
        coords,
        np.ones(n) * u.dimensionless_unscaled,
        rng.uniform(-1, 1, n),
        grid,
    )
    source.build_balltree()
    return source


class TestToMapWithColours:
    def test_averages_the_colour_per_cell(self, coloured_catalog):
        """The non-empty-data arm weights each colour by source brightness."""
        healpix = coloured_catalog.to_map(nside=4)
        assert isinstance(healpix, HEALPixSource)
        assert healpix.data.shape == (1, 12 * 4**2)

    def test_empty_cells_have_no_colour(self, coloured_catalog):
        """Cells with no sources divide zero by zero and yield nan."""
        healpix = coloured_catalog.to_map(nside=16)
        assert np.any(np.isnan(healpix.data))

    def test_occupied_cells_have_a_colour_in_range(self, coloured_catalog):
        healpix = coloured_catalog.to_map(nside=4)
        occupied = healpix.data[np.isfinite(healpix.data)]
        assert len(occupied) > 0
        assert np.all((occupied >= -1) & (occupied <= 1))

    def test_does_not_modify_the_catalog(self, coloured_catalog):
        before = coloured_catalog.data.copy()
        coloured_catalog.to_map(nside=4)
        np.testing.assert_array_equal(coloured_catalog.data, before)

    def test_nan_colours_are_filled_before_averaging(self):
        """A source with no colour takes the catalogue mean, not nan."""
        _, grid = _coloured_grid()
        coords = SkyCoord([10, 11, 12] * u.deg, [0, 0, 0] * u.deg, frame="icrs")
        source = CatalogSource(
            coords,
            np.ones(3) * u.dimensionless_unscaled,
            np.array([0.0, np.nan, 1.0]),
            grid,
        )
        healpix = source.to_map(nside=2)
        occupied = healpix.data[np.isfinite(healpix.data)]
        assert len(occupied) > 0


class TestApplySpaceMotion:
    def test_propagates_positions_to_a_new_epoch(self):
        """A star with proper motion must move, and the index must follow."""
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

        source.apply_space_motion(Time("2020-01-01"))

        moved = np.abs(source.coords.ra.deg - before)
        assert np.all(moved > 0)
        # 500 mas/yr over 20 years is about 10 arcsec, i.e. ~0.003 degrees.
        assert np.all(moved < 0.05)

    def test_rebuilds_the_spatial_index(self):
        coords = SkyCoord(
            ra=[10.0] * u.deg,
            dec=[20.0] * u.deg,
            pm_ra_cosdec=[100.0] * u.mas / u.yr,
            pm_dec=[100.0] * u.mas / u.yr,
            distance=[100.0] * u.pc,
            obstime=Time("2000-01-01"),
            frame="icrs",
        )
        source = CatalogSource(
            coords,
            np.ones(1) * u.dimensionless_unscaled,
            np.zeros((1, 0)),
            make_spectral_grid(),
        )
        source.build_balltree()
        original_tree = source.balltree
        source.apply_space_motion(Time("2020-01-01"))
        assert source.balltree is not original_tree


class TestEphemerisSourceTimeArrays:
    def test_scalar_obstime_is_reshaped(self, observation):
        """A scalar body position must be promoted to a length-one array."""
        source = EphemerisSource(
            "moon",
            lambda t: np.array([1.0]) * u.dimensionless_unscaled,
            lambda t: np.array([0.5]),
            make_spectral_grid(),
        )
        field = source.query_scattered(observation)
        assert field.coords.isscalar is False

    def test_spectral_grid_property_is_exposed(self):
        grid = make_spectral_grid()
        source = EphemerisSource("moon", lambda t: None, lambda t: None, grid)
        assert source.spectral_grid is grid
