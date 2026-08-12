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
    def test_resolve_spectra_restricts_the_grid_to_the_bandpass(self):
        """The wavelength axis of the resolved flux follows the bandpass.

        The radiance flag is carried across unchanged.
        """
        assert _make_source_field().radiance_field is False

        field = _make_source_field(n=1, radiance_field=True)
        resolved = field.resolve_spectra(make_bandpass(lam_min=400, lam_max=600))
        assert isinstance(resolved, ResolvedField)
        assert resolved.radiance_field is True
        assert resolved.wvl.shape[0] == resolved.flx.shape[-2]
        assert resolved.wvl.min() >= 400 * u.nm
        assert resolved.wvl.max() <= 600 * u.nm


class TestResolvedField:
    def test_integrate_weights_each_source_by_its_brightness(self):
        """A flat spectrum integrates to its height times the bandwidth.

        The result is scaled by the weight of each source.
        """
        rates = _make_resolved_field().integrate()
        assert rates.shape == (3, 2)

        single = _make_resolved_field(n=1, n_comp=1)
        baseline = single.integrate()
        assert baseline[0, 0].to_value(u.erg / u.s / u.cm**2) == pytest.approx(400.0)

        weighted = _make_resolved_field(n=2, n_comp=1)
        weighted.weights = np.array([[2.0], [3.0]]) * u.dimensionless_unscaled
        np.testing.assert_allclose(
            weighted.integrate().value[:, 0],
            np.array([2.0, 3.0]) * baseline.value[0, 0],
        )

    def test_integrate_applies_extra_weights_without_modifying_the_input(self):
        """Extinction arrives as an extra weight per source and wavelength."""
        field = _make_resolved_field(n=2, n_comp=1)
        before = field.flx.copy()
        full = field.integrate()
        halved = field.integrate(extra_weights=0.5 * np.ones((2, 50, 1)))
        np.testing.assert_allclose(halved.value, 0.5 * full.value, rtol=0.01)
        np.testing.assert_array_equal(field.flx.value, before.value)


class TestContainers:
    def test_pixel_refs_and_prediction_defaults(self):
        refs = PixelRefs(indices=[np.array([0, 1]), np.array([2])])
        assert refs.weights is None, "weights are filled in by the instrument"
        assert len(refs.indices) == 2
        assert refs.indices[0].shape == (2,)

        pred = Prediction(rates=np.zeros((4, 3)), indirect=False)
        assert pred.rates.shape == (4, 3)
        assert pred.indirect is False
        assert pred.source_name == ""
        assert pred.path_name == ""
