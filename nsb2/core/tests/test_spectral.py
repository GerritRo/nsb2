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


def _make_spectral_grid():
    """A 1D colour grid: 10 colour bins, 50 wavelengths, 3 components."""
    color_pts = np.linspace(-1, 1, 10)
    wvl = np.linspace(300, 700, 50) * u.nm
    flx = np.ones((10, 50, 3)) * u.erg / u.s / u.cm**2 / u.nm
    for i, c in enumerate(color_pts):
        flx[i] *= 1 + 0.5 * c
    return SpectralGrid([color_pts], wvl, flx)


class TestIntegrateWavelength:
    def test_integrates_to_width_times_height_in_the_product_unit(self):
        """A flat spectrum integrates to its height times the bandwidth.

        The unit is ``flux.unit * wavelength.unit``, and the integration runs
        along whichever axis is asked for.
        """
        wvl = np.linspace(400, 500, 11) * u.nm
        flx = np.ones(11) * u.erg / u.s / u.cm**2 / u.nm
        result = integrate_wavelength(flx, wvl)
        assert result.unit.is_equivalent(u.erg / u.s / u.cm**2)
        assert result.to_value(u.erg / u.s / u.cm**2) == pytest.approx(100.0)

        stacked = np.ones((3, 11, 2)) * u.erg / u.s / u.cm**2 / u.nm
        assert integrate_wavelength(stacked, wvl, axis=-2).shape == (3, 2)

        # The same spectrum in angstrom must integrate to the same value.
        in_angstrom = integrate_wavelength(
            flx.to(u.erg / u.s / u.cm**2 / u.angstrom),
            np.linspace(4000, 5000, 11) * u.angstrom,
        )
        assert in_angstrom.to_value(result.unit) == pytest.approx(result.value)

    def test_rejects_plain_arrays(self):
        wvl = np.linspace(400, 500, 11) * u.nm
        with pytest.raises(TypeError, match="flux"):
            integrate_wavelength(np.ones(11), wvl)
        with pytest.raises(TypeError, match="wavelength"):
            integrate_wavelength(np.ones(11) * u.erg, np.linspace(400, 500, 11))


class TestBandpass:
    def test_interpolates_inside_the_band_and_returns_zero_outside(self):
        bp = make_bandpass(lam_min=400, lam_max=700, n=100)
        assert bp.lam.unit == u.nm
        assert len(bp.trx) == 100
        assert bp.min == 400 * u.nm
        assert bp.max == 700 * u.nm
        np.testing.assert_allclose(bp(np.array([450, 500, 600]) * u.nm), 1.0, atol=0.02)
        np.testing.assert_allclose(bp(np.array([200, 800]) * u.nm), 0.0, atol=1e-10)
        # Calling with angstrom must give the same answer as with nm.
        assert bp(np.array([5000]) * u.angstrom)[0] == pytest.approx(1.0, abs=0.02)

    def test_from_csv_multiplies_the_tabulated_columns(self, tmp_path):
        path = tmp_path / "bandpass.dat"
        path.write_text(
            "wvl,mirror,window\n"
            + "\n".join(f"{w},0.5,0.5" for w in range(300, 701, 10))
            + "\n"
        )
        bp = Bandpass.from_csv(path)
        assert bp.min == 300 * u.nm
        assert bp.max == 700 * u.nm
        assert bp(np.array([500]) * u.nm)[0] == pytest.approx(0.25, abs=0.01)


class TestSpectralGrid:
    def test_call_interpolates_and_gives_nan_outside_the_grid(self):
        sg = _make_spectral_grid()
        # No parameter axes to interpolate over: the raw grid comes back.
        assert sg(np.empty((0,))).shape == sg.flx.shape

        result = sg(np.array([[0.0]]))
        assert result.shape == (1, 50, 3)
        np.testing.assert_allclose(result.value, 1.0, atol=0.05)
        assert np.all(np.isnan(sg(np.array([[5.0]])).value))

    def test_apply_bandpass_masks_wavelengths_and_applies_transmission(self):
        sg = _make_spectral_grid()
        before = sg.flx.copy()

        filtered = sg.apply_bandpass(make_bandpass(lam_min=400, lam_max=600))
        assert filtered.wvl.min() >= 400 * u.nm
        assert filtered.wvl.max() <= 600 * u.nm
        assert len(filtered.wvl) < len(sg.wvl)

        lam = np.linspace(300, 700, 50) * u.nm
        halved = sg.apply_bandpass(Bandpass(lam, 0.5 * np.ones(50)))
        np.testing.assert_allclose(
            halved.flx[5, :, 0].value, 0.5 * sg.flx[5, :, 0].value, atol=0.05
        )
        np.testing.assert_array_equal(sg.flx.value, before.value)

    def test_integrate_gives_flux_times_bandwidth(self):
        rg = _make_spectral_grid().integrate()
        assert isinstance(rg, RateGrid)
        assert rg.rate.shape == (10, 3)
        # Centre bin has multiplier ~1 over a 400 nm span.
        expected = 400.0 * (1 + 0.5 * np.linspace(-1, 1, 10)[5])
        assert rg.rate[5, 0].to_value(u.erg / u.s / u.cm**2) == pytest.approx(expected)

    def test_mul_scales_the_flux_without_modifying_the_input(self):
        sg = _make_spectral_grid()
        before = sg.flx.copy()
        scaled = sg * (np.ones(10) * 2.0)
        np.testing.assert_allclose(scaled.flx.value, 2.0 * sg.flx.value)
        np.testing.assert_array_equal(sg.flx.value, before.value)


class TestRateGrid:
    def test_call_interpolates(self):
        empty = RateGrid([], np.array([[1, 2, 3]]) * u.ct / u.s)
        np.testing.assert_array_equal(empty(np.empty((0,))).value, empty.rate.value)

        rg = RateGrid(
            [np.linspace(0, 1, 5)], np.linspace(10, 50, 5)[:, None] * u.ct / u.s
        )
        result = rg(np.array([[0.5]]))
        assert result.shape == (1, 1)
        assert result.value[0, 0] == pytest.approx(30.0, rel=0.01)

        # Multiplication broadcasts the grid over a set of sources.
        broadcast = RateGrid([np.linspace(0, 1, 3)], np.ones((3, 2)) * u.ct / u.s)
        assert (broadcast * np.ones((4, 3))).rate.shape == (4, 3, 2)
