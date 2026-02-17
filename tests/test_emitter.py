import astropy.units as u
import numpy as np
import pytest
from astropy.coordinates import SkyCoord

from nsb2.core.dtypes import PixelRefs, ResolvedField, SourceField
from nsb2.core.sources import (
    CatalogSource,
    HEALPixSource,
    LonLatSource,
    _transform_to_frame,
)
from tests.conftest import make_bandpass, make_observation, make_spectral_grid

# ---------------------------------------------------------------------------
# _transform_to_frame
# ---------------------------------------------------------------------------

class TestTransformToFrame:
    def test_none_frame_returns_input(self):
        sc = SkyCoord(10, 20, unit='deg', frame='icrs')
        result = _transform_to_frame(sc, None)
        assert result is sc

    def test_icrs_to_galactic(self):
        sc = SkyCoord(0, 0, unit='deg', frame='icrs')
        result = _transform_to_frame(sc, 'galactic')
        assert result.frame.name == 'galactic'


# ---------------------------------------------------------------------------
# SourceField
# ---------------------------------------------------------------------------

class TestSourceField:
    def test_radiance_field_false_by_default(self):
        sg = make_spectral_grid()
        coords = SkyCoord([10, 20], [30, 40], unit='deg', frame='icrs')
        field = SourceField(
            coords=coords,
            weights=np.ones((2, 1)) * u.dimensionless_unscaled,
            spectral_data=np.empty((2, 0)),
            spectral_grid=sg,
        )
        assert field.radiance_field is False

    def test_radiance_field_true_when_set(self):
        sg = make_spectral_grid()
        coords = SkyCoord([10], [20], unit='deg', frame='icrs')
        field = SourceField(
            coords=coords,
            weights=np.ones((1, 1)) * u.dimensionless_unscaled,
            spectral_data=np.empty((1, 0)),
            spectral_grid=sg,
            radiance_field=True,
        )
        assert field.radiance_field is True

    def test_resolve_spectra_returns_resolved_field(self):
        sg = make_spectral_grid(n_wvl=20, n_comp=3)
        coords = SkyCoord([10], [20], unit='deg', frame='icrs')
        field = SourceField(
            coords=coords,
            weights=np.ones((1, 1)) * u.dimensionless_unscaled,
            spectral_data=np.empty((1, 0)),
            spectral_grid=sg,
        )
        bp = make_bandpass(n=20, lam_min=300, lam_max=700)
        resolved = field.resolve_spectra(bp)
        assert isinstance(resolved, ResolvedField)
        # With no parameter dimensions, flx is (W, C) — broadcasts with sources
        assert resolved.flx.ndim >= 2
        assert resolved.wvl.shape[0] == resolved.flx.shape[-2]


# ---------------------------------------------------------------------------
# ResolvedField
# ---------------------------------------------------------------------------

class TestResolvedField:
    def test_integrate_returns_rates(self):
        wvl = np.linspace(300, 700, 50) * u.nm
        flx = np.ones((3, 50, 2)) * u.erg / u.s / u.cm**2 / u.nm
        weights = np.ones((3, 1)) * u.dimensionless_unscaled
        coords = SkyCoord([0, 10, 20], [0, 10, 20], unit='deg', frame='icrs')
        rf = ResolvedField(coords=coords, weights=weights, wvl=wvl, flx=flx, radiance_field=False)
        rates = rf.integrate()
        assert rates.shape == (3, 2)  # 3 sources × 2 components

    def test_integrate_with_extra_weights(self):
        wvl = np.linspace(300, 700, 50) * u.nm
        flx = np.ones((2, 50, 1)) * u.erg / u.s / u.cm**2 / u.nm
        weights = np.ones((2, 1)) * u.dimensionless_unscaled
        coords = SkyCoord([0, 10], [0, 10], unit='deg', frame='icrs')
        rf = ResolvedField(coords=coords, weights=weights, wvl=wvl, flx=flx, radiance_field=False)

        # Half extinction
        ext = 0.5 * np.ones((2, 50, 1))
        rates_full = rf.integrate()
        rates_ext = rf.integrate(extra_weights=ext)
        np.testing.assert_allclose(rates_ext.value, 0.5 * rates_full.value, rtol=0.01)


# ---------------------------------------------------------------------------
# PixelRefs
# ---------------------------------------------------------------------------

class TestPixelRefs:
    def test_basic_structure(self):
        pr = PixelRefs(
            indices=[np.array([0, 1]), np.array([2])],
            weights=[np.array([1.0, 0.5]) * u.m**2, np.array([0.8]) * u.m**2],
        )
        assert len(pr.indices) == 2
        assert len(pr.weights) == 2
        assert pr.indices[0].shape == (2,)


# ---------------------------------------------------------------------------
# CatalogSource
# ---------------------------------------------------------------------------

class TestCatalogSource:
    @pytest.fixture
    def catalog(self):
        sg = make_spectral_grid()
        coords = SkyCoord(np.arange(10) * 10 * u.deg,
                          np.zeros(10) * u.deg, frame='icrs')
        weight = np.ones(10) * u.dimensionless_unscaled
        data = np.zeros((10, 0))
        src = CatalogSource(coords, weight, data, sg)
        src.build_balltree()
        return src

    def test_query_direct_returns_field_and_refs(self, catalog):
        obs = make_observation()
        pix_coords = SkyCoord([0, 0.01, -0.01], [0, 0.01, -0.01],
                              unit='rad', frame=obs)
        pix_radii = np.full(3, np.deg2rad(5))
        field, refs = catalog.query_direct(obs, pix_coords, pix_radii)
        assert isinstance(field, SourceField)
        assert isinstance(refs, PixelRefs)
        assert field.radiance_field is False

    def test_query_scattered_returns_field(self, catalog):
        obs = make_observation()
        field = catalog.query_scattered(obs)
        assert isinstance(field, SourceField)
        assert field.radiance_field is False

    def test_getitem_slices(self, catalog):
        sub = catalog[:3]
        assert len(sub.coords) == 3

    def test_to_map(self, catalog):
        healpix_src = catalog.to_map(nside=8)
        assert isinstance(healpix_src, HEALPixSource)


# ---------------------------------------------------------------------------
# LonLatSource
# ---------------------------------------------------------------------------

class TestLonLatSource:
    @pytest.fixture
    def diffuse(self):
        sg = make_spectral_grid()

        def weight_fn(lon, lat):
            return np.ones(len(lon)) * u.dimensionless_unscaled

        def data_fn(lon, lat):
            return np.empty((len(lon), 0))

        return LonLatSource(None, weight_fn, data_fn, sg)

    def test_query_direct(self, diffuse):
        obs = make_observation()
        pix_coords = SkyCoord([0.01, 0.02], [0.01, 0.02], unit='rad', frame=obs)
        field, refs = diffuse.query_direct(obs, pix_coords, np.array([0.1, 0.1]))
        assert isinstance(field, SourceField)
        assert len(refs.indices) == 2

    def test_query_scattered(self, diffuse):
        obs = make_observation()
        field = diffuse.query_scattered(obs, nside=8)
        assert isinstance(field, SourceField)
        assert field.radiance_field is True
