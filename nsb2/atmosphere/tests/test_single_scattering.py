import astropy.units as u
import numpy as np
import pytest
from scipy.integrate import simpson

from nsb2.atmosphere import SingleScatteringAtmosphere
from nsb2.core.atmosphere import haversine


class TestHaversine:
    def test_measures_the_great_circle_separation(self):
        assert haversine(0.0, 0.5, 0.5) == pytest.approx(0.0, abs=1e-12)
        assert haversine(np.pi / 2, 0.0, 0.0) == pytest.approx(np.pi / 2)
        assert haversine(0.0, np.pi / 2, -np.pi / 2) == pytest.approx(np.pi)
        assert haversine(np.zeros((3, 1)), np.zeros((3, 1)), np.zeros(4)).shape == (
            3,
            4,
        )


class TestExtinction:
    def test_follows_beer_lambert_along_the_line_of_sight(self, atmosphere):
        """Transmission is exp(-tau) at zenith, where the airmass is one.

        It falls towards the horizon and, with Rayleigh scattering going as
        lambda^-4, is stronger at short wavelengths.
        """
        zenith = atmosphere.extinction(
            np.array([np.pi / 2]), np.array([0.0]), np.array([400.0]) * u.nm
        )
        assert float(zenith[0, 0]) == pytest.approx(np.exp(-(0.1 + 0.05 + 0.01)))

        wvl = np.array([350, 700]) * u.nm
        overhead = atmosphere.extinction(np.array([np.pi / 2]), np.array([0.0]), wvl)
        assert np.all((overhead > 0) & (overhead <= 1))
        assert overhead[0, 0] < overhead[0, 1], "blue light is hit harder"

        low = atmosphere.extinction(np.array([np.deg2rad(20)]), np.array([0.0]), wvl)
        assert np.all(low[0] < overhead[0])

    def test_broadcasts_over_any_altitude_shape(self, atmosphere):
        """The altitude may arrive as a scalar or with any number of axes.

        The wavelength axis is appended to whatever shape it has.
        """
        wvl = np.linspace(300, 700, 50) * u.nm
        alt = np.linspace(0.1, np.pi / 2, 6)
        flat = atmosphere.extinction(alt, 0.0, wvl)
        assert flat.shape == (6, 50)
        assert atmosphere.extinction(0.7, 0.0, wvl).shape == (50,)

        nested = atmosphere.extinction(alt.reshape(2, 3), 0.0, wvl)
        assert nested.shape == (2, 3, 50)
        np.testing.assert_allclose(np.asarray(nested).reshape(6, 50), np.asarray(flat))


class TestScattering:
    def test_is_forward_peaked_and_finite_where_the_gradation_is_singular(
        self, atmosphere
    ):
        """Forward-peaked Mie scattering piles light up near the source.

        The gradation term must fall back to its limit at equal zenith
        angles, where the general expression is singular.
        """
        broadcast = atmosphere.scattering(
            np.array([np.pi / 4, np.pi / 3])[:, None, None],
            np.array([0.0, 0.1])[:, None, None],
            np.array([np.pi / 2])[None, :, None],
            np.array([0.0])[None, :, None],
            np.array([400, 500]) * u.nm,
        )
        assert broadcast.unit == 1 / u.radian**2
        assert broadcast.shape[-1] == 2

        wvl = np.array([500]) * u.nm
        eval_alt = np.array([np.deg2rad(50)])
        near = atmosphere.scattering(
            eval_alt, np.array([0.0]), np.array([np.deg2rad(55)]), np.array([0.0]), wvl
        )
        far = atmosphere.scattering(
            eval_alt,
            np.array([0.0]),
            np.array([np.deg2rad(55)]),
            np.array([np.pi]),
            wvl,
        )
        assert near[0] > far[0]

        equal = atmosphere.scattering(
            eval_alt, np.array([0.0]), eval_alt, np.array([0.5]), wvl
        )
        assert np.all(np.isfinite(equal.value))

    def test_phase_functions_are_normalised(self):
        """Both phase functions must integrate to one over the sphere."""
        theta = np.linspace(0, np.pi, 20001)
        for values in (
            SingleScatteringAtmosphere._rayleigh(theta),
            SingleScatteringAtmosphere._henyey_greenstein(0.65, theta),
        ):
            integral = simpson(values * 2 * np.pi * np.sin(theta), x=theta)
            assert integral == pytest.approx(1.0, rel=1e-3)
