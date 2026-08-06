"""Tests for :mod:`nsb2.core.dtypes`."""

import astropy.units as u
import numpy as np
import pytest
from astropy.coordinates import SkyCoord

from nsb2.conftest import make_bandpass, make_spectral_grid
from nsb2.core.dtypes import PixelRefs, Prediction, ResolvedField, SourceField


def _make_source_field(n=2, radiance_field=False):
    return SourceField(
        coords=SkyCoord(np.arange(n) * 10, np.zeros(n), unit="deg", frame="icrs"),
        weights=np.ones((n, 1)) * u.dimensionless_unscaled,
        spectral_data=np.empty((n, 0)),
        spectral_grid=make_spectral_grid(),
        radiance_field=radiance_field,
    )


def _make_resolved_field(n=3, n_wvl=50, n_comp=2):
    return ResolvedField(
        coords=SkyCoord(np.arange(n) * 10, np.zeros(n), unit="deg", frame="icrs"),
        wvl=np.linspace(300, 700, n_wvl) * u.nm,
        flx=np.ones((n, n_wvl, n_comp)) * u.erg / u.s / u.cm**2 / u.nm,
        weights=np.ones((n, 1)) * u.dimensionless_unscaled,
        radiance_field=False,
    )


class TestSourceField:
    def test_radiance_field_false_by_default(self):
        assert _make_source_field().radiance_field is False

    def test_radiance_field_true_when_set(self):
        assert _make_source_field(radiance_field=True).radiance_field is True

    def test_resolve_spectra_returns_resolved_field(self):
        field = _make_source_field(n=1)
        resolved = field.resolve_spectra(make_bandpass(n=20))
        assert isinstance(resolved, ResolvedField)
        assert resolved.flx.ndim >= 2
        assert resolved.wvl.shape[0] == resolved.flx.shape[-2]

    def test_resolve_spectra_preserves_radiance_flag(self):
        field = _make_source_field(n=1, radiance_field=True)
        assert field.resolve_spectra(make_bandpass(n=20)).radiance_field is True

    def test_resolve_spectra_restricts_to_bandpass(self):
        field = _make_source_field(n=1)
        resolved = field.resolve_spectra(make_bandpass(lam_min=400, lam_max=600))
        assert resolved.wvl.min() >= 400 * u.nm
        assert resolved.wvl.max() <= 600 * u.nm


class TestResolvedField:
    def test_integrate_returns_rates(self):
        rates = _make_resolved_field().integrate()
        assert rates.shape == (3, 2)

    def test_integrate_gives_flux_times_bandwidth(self):
        rates = _make_resolved_field(n=1, n_comp=1).integrate()
        assert rates[0, 0].to_value(u.erg / u.s / u.cm**2) == pytest.approx(400.0)

    def test_integrate_with_extra_weights(self):
        rf = _make_resolved_field(n=2, n_comp=1)
        full = rf.integrate()
        halved = rf.integrate(extra_weights=0.5 * np.ones((2, 50, 1)))
        np.testing.assert_allclose(halved.value, 0.5 * full.value, rtol=0.01)

    def test_integrate_applies_source_weights(self):
        rf = _make_resolved_field(n=2, n_comp=1)
        baseline = rf.integrate()
        rf.weights = np.array([[2.0], [3.0]]) * u.dimensionless_unscaled
        weighted = rf.integrate()
        np.testing.assert_allclose(
            weighted.value[:, 0], np.array([2.0, 3.0]) * baseline.value[0, 0]
        )

    def test_integrate_does_not_modify_input(self):
        rf = _make_resolved_field()
        before = rf.flx.copy()
        rf.integrate(extra_weights=0.5 * np.ones((3, 50, 2)))
        np.testing.assert_array_equal(rf.flx.value, before.value)


class TestPixelRefs:
    def test_weights_default_to_none(self):
        assert PixelRefs(indices=[np.array([0])]).weights is None

    def test_basic_structure(self):
        refs = PixelRefs(
            indices=[np.array([0, 1]), np.array([2])],
            weights=[np.array([1.0, 0.5]) * u.m**2, np.array([0.8]) * u.m**2],
        )
        assert len(refs.indices) == 2
        assert len(refs.weights) == 2
        assert refs.indices[0].shape == (2,)


class TestPrediction:
    def test_fields(self):
        pred = Prediction(rates=np.zeros((4, 3)), indirect=False)
        assert pred.indirect is False
        assert pred.rates.shape == (4, 3)
        assert pred.source_name == ""
        assert pred.path_name == ""
