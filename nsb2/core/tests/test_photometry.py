"""Tests for :mod:`nsb2.core.photometry`."""

import astropy.units as u
import numpy as np
import pytest

from nsb2.core.photometry import (
    PicklesTRDSAtlas1998,
    SolarSpectrumRieke2008,
    create_color_grid,
    synthetic_magnitude,
)
from nsb2.core.spectral import Bandpass, SpectralGrid, integrate_wavelength


def _calibrated_bandpass(lam_min=400, lam_max=600, n=100):
    """A top-hat bandpass with its Vega zeropoint stubbed out.

    ``vegazero`` is a cached property backed by a download; assigning to it
    pins a known value so the photometry can be checked without network
    access.
    """
    lam = np.linspace(lam_min, lam_max, n) * u.nm
    bandpass = Bandpass(lam, np.ones(n))
    reference = np.ones(n) * u.erg / u.s / u.cm**2 / u.nm
    bandpass.vegazero = integrate_wavelength(lam * bandpass(lam) * reference, lam)
    return bandpass


class TestSyntheticMagnitude:
    def test_reference_spectrum_has_magnitude_zero(self):
        bandpass = _calibrated_bandpass()
        wvl = np.linspace(400, 600, 100) * u.nm
        flx = np.ones(100) * u.erg / u.s / u.cm**2 / u.nm
        assert synthetic_magnitude(wvl, flx, bandpass) == pytest.approx(0.0, abs=1e-6)

    def test_a_hundredth_of_the_flux_is_five_magnitudes_fainter(self):
        bandpass = _calibrated_bandpass()
        wvl = np.linspace(400, 600, 100) * u.nm
        flx = 0.01 * np.ones(100) * u.erg / u.s / u.cm**2 / u.nm
        assert synthetic_magnitude(wvl, flx, bandpass) == pytest.approx(5.0, abs=1e-6)

    def test_result_is_independent_of_wavelength_unit(self):
        bandpass = _calibrated_bandpass()
        flx = np.ones(100) * u.erg / u.s / u.cm**2 / u.nm
        in_nm = synthetic_magnitude(np.linspace(400, 600, 100) * u.nm, flx, bandpass)
        in_angstrom = synthetic_magnitude(
            np.linspace(4000, 6000, 100) * u.angstrom,
            flx.to(u.erg / u.s / u.cm**2 / u.angstrom),
            bandpass,
        )
        assert in_angstrom == pytest.approx(in_nm)

    def test_reduces_leading_axes(self):
        bandpass = _calibrated_bandpass()
        wvl = np.linspace(400, 600, 100) * u.nm
        flx = np.ones((3, 100)) * u.erg / u.s / u.cm**2 / u.nm
        assert synthetic_magnitude(wvl, flx, bandpass).shape == (3,)


class TestPicklesTRDSAtlas1998:
    def test_loads_a_spectral_grid(self):
        atlas = PicklesTRDSAtlas1998()
        assert isinstance(atlas, SpectralGrid)
        assert atlas.points == []

    def test_wavelengths_are_monotonic_and_in_the_optical(self):
        atlas = PicklesTRDSAtlas1998()
        assert np.all(np.diff(atlas.wvl.value) > 0)
        assert atlas.wvl.min() < 400 * u.nm
        assert atlas.wvl.max() > 900 * u.nm

    def test_fluxes_are_finite_and_overwhelmingly_positive(self):
        """The atlas carries a few small negative values from its calibration."""
        atlas = PicklesTRDSAtlas1998()
        assert np.all(np.isfinite(atlas.flx.value))
        assert np.mean(atlas.flx.value >= 0) > 0.99

    def test_holds_many_templates(self):
        assert PicklesTRDSAtlas1998().flx.shape[-1] > 100


class TestCreateColorGrid:
    def test_builds_a_grid_indexed_by_color(self):
        library = SpectralGrid(
            [],
            np.linspace(3000, 10000, 60) * u.angstrom,
            np.ones((60, 4)) * u.erg / u.angstrom / u.s / u.cm**2,
        )
        blue = _calibrated_bandpass(lam_min=400, lam_max=500)
        red = _calibrated_bandpass(lam_min=600, lam_max=700)
        grid = create_color_grid(
            blue, [blue, red], [-1.0, 1.0], library, EBVs=np.linspace(0, 2, 5)
        )
        assert isinstance(grid, SpectralGrid)
        assert len(grid.points) == 1
        assert grid.flx.shape[0] == len(grid.points[0])

    def test_uses_the_default_reddening_grid(self):
        """Omitting EBVs must fall back to DEFAULT_EBVS rather than fail."""
        library = SpectralGrid(
            [],
            np.linspace(3000, 10000, 40) * u.angstrom,
            np.ones((40, 2)) * u.erg / u.angstrom / u.s / u.cm**2,
        )
        blue = _calibrated_bandpass(lam_min=400, lam_max=500)
        red = _calibrated_bandpass(lam_min=600, lam_max=700)
        grid = create_color_grid(blue, [blue, red], [-1.0, 1.0], library)
        assert grid.flx.shape[0] == len(grid.points[0])

    def test_accepts_an_alternative_extinction_model(self):
        """The extmod argument is the documented injection point."""
        library = SpectralGrid(
            [],
            np.linspace(3000, 10000, 40) * u.angstrom,
            np.ones((40, 2)) * u.erg / u.angstrom / u.s / u.cm**2,
        )
        blue = _calibrated_bandpass(lam_min=400, lam_max=500)
        red = _calibrated_bandpass(lam_min=600, lam_max=700)

        class GreyExtinction:
            """A wavelength-independent extinction law.

            Mirrors the dust_extinction interface: ``Ebv`` arrives with a
            trailing singleton axis to broadcast against the wavelengths.
            """

            def extinguish(self, wvl, Ebv):
                return 10 ** (-0.4 * np.asarray(Ebv)) * np.ones(len(wvl))

        grid = create_color_grid(
            blue,
            [blue, red],
            [-1.0, 1.0],
            library,
            EBVs=np.linspace(0, 2, 5),
            extmod=GreyExtinction(),
        )
        assert isinstance(grid, SpectralGrid)


@pytest.mark.remote_data
class TestSolarSpectrumRieke2008:
    def test_returns_wavelength_and_flux(self):
        wvl, flx = SolarSpectrumRieke2008()
        assert wvl.unit.is_equivalent(u.nm)
        assert flx.unit.is_equivalent(u.erg / u.s / u.cm**2 / u.nm)
        assert wvl.shape == flx.shape

    def test_peaks_in_the_visible(self):
        wvl, flx = SolarSpectrumRieke2008()
        optical = (wvl > 300 * u.nm) & (wvl < 1000 * u.nm)
        peak = wvl[optical][np.argmax(flx[optical])]
        assert 400 * u.nm < peak < 600 * u.nm
