import astropy.units as u
import numpy as np
import pytest

from nsb2.core.spectral import Bandpass, RateGrid, SpectralGrid

# ---------------------------------------------------------------------------
# Bandpass
# ---------------------------------------------------------------------------

def _make_bandpass(n=100, lam_min=300, lam_max=600):
    """Create a simple top-hat bandpass for testing."""
    lam = np.linspace(lam_min, lam_max, n) * u.nm
    trx = np.ones(n)
    return Bandpass(lam, trx)


class TestBandpass:
    def test_init_stores_wavelength_and_transmission(self):
        bp = _make_bandpass()
        assert bp.lam.unit == u.nm
        assert len(bp.trx) == 100

    def test_min_max(self):
        bp = _make_bandpass(lam_min=400, lam_max=700)
        assert bp.min == 400 * u.nm
        assert bp.max == 700 * u.nm

    def test_call_returns_transmission(self):
        bp = _make_bandpass(lam_min=400, lam_max=700)
        # Inside band
        vals = bp(np.array([450, 500, 600]) * u.nm)
        np.testing.assert_allclose(vals, 1.0, atol=0.02)

    def test_call_outside_band_returns_zero(self):
        bp = _make_bandpass(lam_min=400, lam_max=700)
        vals = bp(np.array([200, 800]) * u.nm)
        np.testing.assert_allclose(vals, 0.0, atol=1e-10)

    def test_call_unit_conversion(self):
        """Calling with angstrom should still work."""
        bp = _make_bandpass(lam_min=400, lam_max=700)
        val = bp(np.array([5000]) * u.angstrom)  # 500 nm
        assert val[0] == pytest.approx(1.0, abs=0.02)


# ---------------------------------------------------------------------------
# SpectralGrid
# ---------------------------------------------------------------------------

def _make_spectral_grid():
    """Simple 1D spectral grid: 10 color bins × 50 wavelengths × 3 components."""
    color_pts = np.linspace(-1, 1, 10)
    wvl = np.linspace(300, 700, 50) * u.nm
    # Flat spectra that scale linearly with color index
    flx = np.ones((10, 50, 3)) * u.erg / u.s / u.cm**2 / u.nm
    for i, c in enumerate(color_pts):
        flx[i] *= (1 + 0.5 * c)
    return SpectralGrid([color_pts], wvl, flx)


class TestSpectralGrid:
    def test_call_with_empty_xi_returns_raw_flx(self):
        sg = _make_spectral_grid()
        result = sg(np.empty((0,)))
        assert result.shape == sg.flx.shape

    def test_call_interpolates(self):
        sg = _make_spectral_grid()
        xi = np.array([[0.0]])  # center of grid
        result = sg(xi)
        assert result.shape == (1, 50, 3)
        # At color=0, multiplier is 1.0
        np.testing.assert_allclose(result.value, 1.0, atol=0.05)

    def test_apply_bandpass_masks_wavelengths(self):
        sg = _make_spectral_grid()
        bp = _make_bandpass(lam_min=400, lam_max=600)
        filtered = sg.apply_bandpass(bp)
        assert filtered.wvl.min() >= 400 * u.nm
        assert filtered.wvl.max() <= 600 * u.nm
        assert len(filtered.wvl) < len(sg.wvl)

    def test_apply_bandpass_multiplies_transmission(self):
        sg = _make_spectral_grid()
        # Half-transmission bandpass
        lam = np.linspace(300, 700, 50) * u.nm
        trx = 0.5 * np.ones(50)
        bp = Bandpass(lam, trx)
        filtered = sg.apply_bandpass(bp)
        # Flux should be halved
        np.testing.assert_allclose(
            filtered.flx[5, :, 0].value,
            0.5 * sg.flx[5, :, 0].value,
            atol=0.05)

    def test_integrate_returns_rate_grid(self):
        sg = _make_spectral_grid()
        rg = sg.integrate()
        assert isinstance(rg, RateGrid)
        assert rg.rate.shape == (10, 3)

    def test_mul_scales_flux(self):
        sg = _make_spectral_grid()
        scale = np.ones(10) * 2.0
        sg2 = sg * scale
        np.testing.assert_allclose(sg2.flx.value, 2.0 * sg.flx.value)


# ---------------------------------------------------------------------------
# RateGrid
# ---------------------------------------------------------------------------

class TestRateGrid:
    def test_call_with_empty_xi_returns_raw_rate(self):
        rg = RateGrid([], np.array([[1, 2, 3]]) * u.ct / u.s)
        result = rg(np.empty((0,)))
        np.testing.assert_array_equal(result.value, rg.rate.value)

    def test_call_interpolates(self):
        pts = np.linspace(0, 1, 5)
        rate = np.linspace(10, 50, 5)[:, None] * u.ct / u.s
        rg = RateGrid([pts], rate)
        result = rg(np.array([[0.5]]))
        assert result.shape == (1, 1)
        assert result.value[0, 0] == pytest.approx(30.0, rel=0.01)
