import astropy.units as u
import numpy as np
import pytest

from nsb2.core.sources import LonLatSource
from nsb2.emitter.airglow import from_eso_skycalc, van_rhijn

LON, LAT = np.array([0.0]), np.array([np.pi / 4])


class TestVanRhijn:
    def test_brightens_towards_the_horizon_from_unity_at_zenith(self):
        assert van_rhijn(90, 0.0) == pytest.approx(1.0)
        assert np.all(np.diff(van_rhijn(90, np.deg2rad([0, 30, 60, 80]))) > 0)
        assert van_rhijn(300, np.deg2rad(80)) < van_rhijn(90, np.deg2rad(80))
        assert van_rhijn(90, np.zeros((2, 3))).shape == (2, 3)


class TestFromEsoSkycalc:
    def test_builds_a_source_fixed_to_the_local_horizon(self):
        source = from_eso_skycalc(90 * u.km, 130)
        assert isinstance(source, LonLatSource)
        assert source.name == "airglow_eso_skycalc"
        assert source.frame is None
        assert np.all(source.spectral_grid.flx.value >= 0)

    def test_weight_scales_with_solar_activity_and_takes_any_length_unit(self):
        quiet = from_eso_skycalc(90 * u.km, 70).weight_function(LON, LAT)
        active = from_eso_skycalc(90 * u.km, 250).weight_function(LON, LAT)
        assert active > quiet

        in_km = from_eso_skycalc(90 * u.km, 130).weight_function(LON, LAT)
        in_m = from_eso_skycalc(90000 * u.m, 130).weight_function(LON, LAT)
        assert in_m == pytest.approx(in_km)

        # @quantity_input turns an ambiguous unit into an immediate error.
        with pytest.raises(u.UnitsError):
            from_eso_skycalc(90 * u.s, 130)
        with pytest.raises(TypeError):
            from_eso_skycalc(90, 130)

    def test_query_scattered_covers_the_visible_sky(self, observation):
        field = from_eso_skycalc(90 * u.km, 130).query_scattered(observation, nside=8)
        assert field.radiance_field is True
        assert len(field.coords) > 0
