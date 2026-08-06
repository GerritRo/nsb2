"""Tests for :mod:`nsb2.atmosphere.single_scattering`."""

import astropy.units as u
import numpy as np
import pytest
from scipy.integrate import simpson

from nsb2.atmosphere import SingleScatteringAtmosphere
from nsb2.core.atmosphere import haversine


class TestHaversine:
    def test_zero_separation(self):
        assert haversine(0.0, 0.5, 0.5) == pytest.approx(0.0, abs=1e-12)

    def test_quarter_turn_along_the_equator(self):
        assert haversine(np.pi / 2, 0.0, 0.0) == pytest.approx(np.pi / 2)

    def test_antipodal_points(self):
        assert haversine(0.0, np.pi / 2, -np.pi / 2) == pytest.approx(np.pi)

    def test_broadcasts(self):
        result = haversine(np.zeros((3, 1)), np.zeros((3, 1)), np.zeros(4))
        assert result.shape == (3, 4)


class TestSingleScatteringAtmosphere:
    def test_extinction_at_zenith(self, atmosphere):
        wvl = np.array([400, 500, 600]) * u.nm
        ext = atmosphere.extinction(np.array([np.pi / 2]), np.array([0.0]), wvl)
        assert ext.shape == (1, 3)
        assert np.all((ext > 0) & (ext <= 1))

    def test_extinction_matches_beer_lambert_at_zenith(self, atmosphere):
        """At zenith the airmass is one, so transmission is exp(-tau)."""
        wvl = np.array([400.0]) * u.nm
        tau = 0.1 + 0.05 + 0.01
        ext = atmosphere.extinction(np.array([np.pi / 2]), np.array([0.0]), wvl)
        assert float(ext[0, 0]) == pytest.approx(np.exp(-tau))

    def test_extinction_decreases_towards_the_horizon(self, atmosphere):
        wvl = np.array([500]) * u.nm
        zenith = atmosphere.extinction(np.array([np.pi / 2]), np.array([0.0]), wvl)
        low = atmosphere.extinction(np.array([np.deg2rad(20)]), np.array([0.0]), wvl)
        assert low[0, 0] < zenith[0, 0]

    def test_extinction_is_stronger_at_short_wavelengths(self, atmosphere):
        """Rayleigh scattering goes as lambda^-4, so blue light is hit harder."""
        ext = atmosphere.extinction(
            np.array([np.pi / 2]), np.array([0.0]), np.array([350, 700]) * u.nm
        )
        assert ext[0, 0] < ext[0, 1]

    def test_extinction_shape(self, atmosphere):
        wvl = np.linspace(300, 700, 50) * u.nm
        alt = np.array([np.pi / 4, np.pi / 3, np.pi / 2])
        assert atmosphere.extinction(alt, np.zeros(3), wvl).shape == (3, 50)

    def test_scattering_returns_inverse_steradian(self, atmosphere):
        scat = atmosphere.scattering(
            np.array([np.pi / 4]),
            np.array([0.0]),
            np.array([np.pi / 3]),
            np.array([0.5]),
            np.array([500]) * u.nm,
        )
        assert scat.unit == 1 / u.radian**2

    def test_scattering_shape(self, atmosphere):
        scat = atmosphere.scattering(
            np.array([np.pi / 4, np.pi / 3])[:, None, None],
            np.array([0.0, 0.1])[:, None, None],
            np.array([np.pi / 2])[None, :, None],
            np.array([0.0])[None, :, None],
            np.array([400, 500]) * u.nm,
        )
        assert scat.shape[-1] == 2

    def test_scattering_is_finite_at_equal_zenith_angles(self, atmosphere):
        """The gradation term is singular there and must fall back to its limit."""
        angle = np.array([np.pi / 4])
        scat = atmosphere.scattering(
            angle, np.array([0.0]), angle, np.array([0.5]), np.array([500]) * u.nm
        )
        assert np.all(np.isfinite(scat.value))

    def test_scattering_is_stronger_at_small_separations(self, atmosphere):
        """Forward-peaked Mie scattering means light piles up near the source."""
        wvl = np.array([500]) * u.nm
        near = atmosphere.scattering(
            np.array([np.deg2rad(50)]),
            np.array([0.0]),
            np.array([np.deg2rad(55)]),
            np.array([0.0]),
            wvl,
        )
        far = atmosphere.scattering(
            np.array([np.deg2rad(50)]),
            np.array([0.0]),
            np.array([np.deg2rad(55)]),
            np.array([np.pi]),
            wvl,
        )
        assert near[0] > far[0]

    def test_phase_functions_are_normalised(self):
        """Both phase functions must integrate to one over the sphere."""
        theta = np.linspace(0, np.pi, 20001)
        for values in (
            SingleScatteringAtmosphere._rayleigh(theta),
            SingleScatteringAtmosphere._henyey_greenstein(0.65, theta),
        ):
            integral = simpson(values * 2 * np.pi * np.sin(theta), x=theta)
            assert integral == pytest.approx(1.0, rel=1e-3)
