"""Tests for :mod:`nsb2.core.spectral`."""

import astropy.units as u
import numpy as np
import pytest

from nsb2.conftest import make_bandpass
from nsb2.core.spectral import (
    Bandpass,
    RateGrid,
    SpectralGrid,
    integrate_wavelength,
)


class TestIntegrateWavelength:
    def test_flat_spectrum_gives_width_times_height(self):
        wvl = np.linspace(400, 500, 11) * u.nm
        flx = np.ones(11) * u.erg / u.s / u.cm**2 / u.nm
        result = integrate_wavelength(flx, wvl)
        assert result.unit.is_equivalent(u.erg / u.s / u.cm**2)
        assert result.to_value(u.erg / u.s / u.cm**2) == pytest.approx(100.0)

    def test_unit_is_product_of_input_units(self):
        wvl = np.linspace(400, 500, 11) * u.angstrom
        flx = np.ones(11) * u.W / u.m**2 / u.angstrom
        result = integrate_wavelength(flx, wvl)
        assert result.unit.is_equivalent(u.W / u.m**2)

    def test_result_is_independent_of_wavelength_unit(self):
        """The same spectrum in nm and angstrom must integrate to the same value."""
        flx = np.ones(11) * u.erg / u.s / u.cm**2 / u.nm
        in_nm = integrate_wavelength(flx, np.linspace(400, 500, 11) * u.nm)
        in_angstrom = integrate_wavelength(
            flx.to(u.erg / u.s / u.cm**2 / u.angstrom),
            np.linspace(4000, 5000, 11) * u.angstrom,
        )
        assert in_angstrom.to_value(in_nm.unit) == pytest.approx(in_nm.value)

    def test_integrates_along_requested_axis(self):
        wvl = np.linspace(400, 500, 11) * u.nm
        flx = np.ones((3, 11, 2)) * u.erg / u.s / u.cm**2 / u.nm
        assert integrate_wavelength(flx, wvl, axis=-2).shape == (3, 2)

    def test_rejects_plain_arrays(self):
        wvl = np.linspace(400, 500, 11) * u.nm
        with pytest.raises(TypeError, match="flux"):
            integrate_wavelength(np.ones(11), wvl)
        with pytest.raises(TypeError, match="wavelength"):
            integrate_wavelength(np.ones(11) * u.erg, np.linspace(400, 500, 11))


class TestBandpass:
    def test_init_stores_wavelength_and_transmission(self):
        bp = make_bandpass(n=100)
        assert bp.lam.unit == u.nm
        assert len(bp.trx) == 100

    def test_min_max(self):
        bp = make_bandpass(lam_min=400, lam_max=700)
        assert bp.min == 400 * u.nm
        assert bp.max == 700 * u.nm

    def test_call_returns_transmission(self):
        bp = make_bandpass(lam_min=400, lam_max=700)
        np.testing.assert_allclose(bp(np.array([450, 500, 600]) * u.nm), 1.0, atol=0.02)

    def test_call_outside_band_returns_zero(self):
        bp = make_bandpass(lam_min=400, lam_max=700)
        np.testing.assert_allclose(bp(np.array([200, 800]) * u.nm), 0.0, atol=1e-10)

    def test_call_unit_conversion(self):
        """Calling with angstrom must give the same answer as with nm."""
        bp = make_bandpass(lam_min=400, lam_max=700)
        assert bp(np.array([5000]) * u.angstrom)[0] == pytest.approx(1.0, abs=0.02)

    def test_from_csv(self, tmp_path):
        path = tmp_path / "bandpass.dat"
        path.write_text(
            "wvl,mirror,window\n"
            + "\n".join(f"{w},0.5,0.5" for w in range(300, 701, 10))
            + "\n"
        )
        bp = Bandpass.from_csv(path)
        assert bp.min == 300 * u.nm
        assert bp.max == 700 * u.nm
        # Transmission is the product of the two tabulated columns.
        assert bp(np.array([500]) * u.nm)[0] == pytest.approx(0.25, abs=0.01)


def _make_spectral_grid():
    """A 1D colour grid: 10 colour bins, 50 wavelengths, 3 components."""
    color_pts = np.linspace(-1, 1, 10)
    wvl = np.linspace(300, 700, 50) * u.nm
    flx = np.ones((10, 50, 3)) * u.erg / u.s / u.cm**2 / u.nm
    for i, c in enumerate(color_pts):
        flx[i] *= 1 + 0.5 * c
    return SpectralGrid([color_pts], wvl, flx)


class TestSpectralGrid:
    def test_call_with_empty_xi_returns_raw_flx(self):
        sg = _make_spectral_grid()
        assert sg(np.empty((0,))).shape == sg.flx.shape

    def test_call_interpolates(self):
        sg = _make_spectral_grid()
        result = sg(np.array([[0.0]]))
        assert result.shape == (1, 50, 3)
        np.testing.assert_allclose(result.value, 1.0, atol=0.05)

    def test_call_outside_grid_gives_nan(self):
        sg = _make_spectral_grid()
        assert np.all(np.isnan(sg(np.array([[5.0]])).value))

    def test_apply_bandpass_masks_wavelengths(self):
        sg = _make_spectral_grid()
        filtered = sg.apply_bandpass(make_bandpass(lam_min=400, lam_max=600))
        assert filtered.wvl.min() >= 400 * u.nm
        assert filtered.wvl.max() <= 600 * u.nm
        assert len(filtered.wvl) < len(sg.wvl)

    def test_apply_bandpass_does_not_modify_input(self):
        sg = _make_spectral_grid()
        before = sg.flx.copy()
        sg.apply_bandpass(make_bandpass(lam_min=400, lam_max=600))
        np.testing.assert_array_equal(sg.flx.value, before.value)

    def test_apply_bandpass_multiplies_transmission(self):
        sg = _make_spectral_grid()
        lam = np.linspace(300, 700, 50) * u.nm
        filtered = sg.apply_bandpass(Bandpass(lam, 0.5 * np.ones(50)))
        np.testing.assert_allclose(
            filtered.flx[5, :, 0].value, 0.5 * sg.flx[5, :, 0].value, atol=0.05
        )

    def test_integrate_returns_rate_grid(self):
        rg = _make_spectral_grid().integrate()
        assert isinstance(rg, RateGrid)
        assert rg.rate.shape == (10, 3)

    def test_integrate_gives_flux_times_bandwidth(self):
        rg = _make_spectral_grid().integrate()
        # Centre bin has multiplier ~1 over a 400 nm span.
        expected = 400.0 * (1 + 0.5 * np.linspace(-1, 1, 10)[5])
        assert rg.rate[5, 0].to_value(u.erg / u.s / u.cm**2) == pytest.approx(expected)

    def test_mul_scales_flux(self):
        sg = _make_spectral_grid()
        sg2 = sg * (np.ones(10) * 2.0)
        np.testing.assert_allclose(sg2.flx.value, 2.0 * sg.flx.value)

    def test_mul_does_not_modify_input(self):
        sg = _make_spectral_grid()
        before = sg.flx.copy()
        sg * (np.ones(10) * 2.0)
        np.testing.assert_array_equal(sg.flx.value, before.value)


class TestRateGrid:
    def test_call_with_empty_xi_returns_raw_rate(self):
        rg = RateGrid([], np.array([[1, 2, 3]]) * u.ct / u.s)
        np.testing.assert_array_equal(rg(np.empty((0,))).value, rg.rate.value)

    def test_call_interpolates(self):
        rg = RateGrid(
            [np.linspace(0, 1, 5)], np.linspace(10, 50, 5)[:, None] * u.ct / u.s
        )
        result = rg(np.array([[0.5]]))
        assert result.shape == (1, 1)
        assert result.value[0, 0] == pytest.approx(30.0, rel=0.01)

    def test_mul_broadcasts_over_sources(self):
        rg = RateGrid([np.linspace(0, 1, 3)], np.ones((3, 2)) * u.ct / u.s)
        scaled = rg * np.ones((4, 3))
        assert scaled.rate.shape == (4, 3, 2)
