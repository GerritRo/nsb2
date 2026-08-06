"""Tests for :mod:`nsb2.emitter.airglow`."""

import astropy.units as u
import numpy as np
import pytest

from nsb2.core.sources import LonLatSource
from nsb2.emitter.airglow import from_eso_skycalc, van_rhijn


class TestVanRhijn:
    def test_unity_at_zenith(self):
        assert van_rhijn(90, 0.0) == pytest.approx(1.0)

    def test_increases_towards_the_horizon(self):
        zenith_angles = np.deg2rad([0, 30, 60, 80])
        values = van_rhijn(90, zenith_angles)
        assert np.all(np.diff(values) > 0)

    def test_a_higher_layer_is_less_enhanced(self):
        """A shell further from the surface is crossed at a steeper angle."""
        angle = np.deg2rad(80)
        assert van_rhijn(300, angle) < van_rhijn(90, angle)

    def test_broadcasts_over_zenith_angle(self):
        assert van_rhijn(90, np.zeros((2, 3))).shape == (2, 3)


class TestFromEsoSkycalc:
    def test_builds_a_lonlat_source(self):
        source = from_eso_skycalc(90 * u.km, 130)
        assert isinstance(source, LonLatSource)
        assert source.name == "airglow_eso_skycalc"

    def test_is_defined_in_the_observation_frame(self):
        """Airglow is fixed to the local horizon, so it carries no frame."""
        assert from_eso_skycalc(90 * u.km, 130).frame is None

    def test_spectrum_is_positive(self):
        source = from_eso_skycalc(90 * u.km, 130)
        assert np.all(source.spectral_grid.flx.value >= 0)

    def test_brighter_at_higher_solar_activity(self):
        quiet = from_eso_skycalc(90 * u.km, 70)
        active = from_eso_skycalc(90 * u.km, 250)
        lon, lat = np.array([0.0]), np.array([np.pi / 4])
        assert active.weight_function(lon, lat) > quiet.weight_function(lon, lat)

    def test_height_unit_is_converted(self):
        """Passing the height in metres must give the same answer as in km."""
        lon, lat = np.array([0.0]), np.array([np.pi / 4])
        in_km = from_eso_skycalc(90 * u.km, 130).weight_function(lon, lat)
        in_m = from_eso_skycalc(90000 * u.m, 130).weight_function(lon, lat)
        assert in_m == pytest.approx(in_km)

    def test_rejects_a_height_that_is_not_a_length(self):
        """@quantity_input turns an ambiguous unit into an immediate error."""
        with pytest.raises(u.UnitsError):
            from_eso_skycalc(90 * u.s, 130)

    def test_rejects_a_bare_number(self):
        with pytest.raises(TypeError):
            from_eso_skycalc(90, 130)

    def test_query_scattered_covers_the_visible_sky(self, observation):
        field = from_eso_skycalc(90 * u.km, 130).query_scattered(observation, nside=8)
        assert field.radiance_field is True
        assert len(field.coords) > 0
