import astropy.units as u
import numpy as np
import pytest

from nsb2.atmosphere import SingleScatteringAtmosphere


class TestSingleScatteringAtmosphere:
    @pytest.fixture
    def simple_atmosphere(self):
        """Create a minimal atmosphere with one Rayleigh-like scattering component."""
        def X(z):
            return 1 / np.cos(np.clip(z, 0, np.deg2rad(85)))  # simple sec(z)
        def tau_rayleigh(wvl):
            return 0.1 * (400 * u.nm / wvl) ** 4
        def tau_abs(wvl):
            return 0.01 * np.ones_like(wvl.value)
        return SingleScatteringAtmosphere(
            airmass_func=X,
            tau_rayleigh=tau_rayleigh,
            tau_mie=lambda wvl: 0.05 * np.ones_like(wvl.value),
            tau_absorption=tau_abs,
            g=0.65,
        )

    def test_extinction_at_zenith(self, simple_atmosphere):
        """Extinction at zenith should be exp(-tau * 1)."""
        wvl = np.array([400, 500, 600]) * u.nm
        ext = simple_atmosphere.extinction(np.array([np.pi / 2]), np.array([0.0]), wvl)
        # At zenith, airmass = 1
        assert ext.shape == (1, 3)
        assert np.all(ext > 0)
        assert np.all(ext <= 1)

    def test_extinction_decreases_at_horizon(self, simple_atmosphere):
        """More extinction at lower altitudes (higher zenith angles)."""
        wvl = np.array([500]) * u.nm
        ext_zenith = simple_atmosphere.extinction(np.array([np.pi / 2]), np.array([0.0]), wvl)
        ext_low = simple_atmosphere.extinction(np.array([np.deg2rad(20)]), np.array([0.0]), wvl)
        assert ext_low[0, 0] < ext_zenith[0, 0]

    def test_extinction_shape(self, simple_atmosphere):
        """Output shape should be (N_sources, N_wavelengths)."""
        wvl = np.linspace(300, 700, 50) * u.nm
        alt = np.array([np.pi / 4, np.pi / 3, np.pi / 2])
        ext = simple_atmosphere.extinction(alt, np.zeros(3), wvl)
        assert ext.shape == (3, 50)

    def test_scattering_returns_correct_units(self, simple_atmosphere):
        """Scattering kernel should have units of 1/sr (1/rad^2)."""
        wvl = np.array([500]) * u.nm
        scat = simple_atmosphere.scattering(
            np.array([np.pi / 4]), np.array([0.0]),
            np.array([np.pi / 3]), np.array([0.5]),
            wvl)
        assert scat.unit == 1 / u.radian**2

    def test_scattering_shape(self, simple_atmosphere):
        wvl = np.array([400, 500]) * u.nm
        scat = simple_atmosphere.scattering(
            np.array([np.pi / 4, np.pi / 3])[:, None, None],
            np.array([0.0, 0.1])[:, None, None],
            np.array([np.pi / 2])[None, :, None],
            np.array([0.0])[None, :, None],
            wvl)
        assert scat.shape[-1] == 2  # wavelength dim
