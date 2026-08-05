"""Tests for emitter factory functions (airglow, zodiacal, moon)."""

from unittest.mock import MagicMock, patch

import astropy.units as u
import numpy as np
import pytest
from astropy.coordinates import SkyCoord
from astropy.time import Time

from nsb2.core.sources import EphemerisSource, LonLatSource
from tests.conftest import make_observation


# ---------------------------------------------------------------------------
# Airglow
# ---------------------------------------------------------------------------


class TestAirglowFromEsoSkycalc:
    def test_returns_lonlat_source(self):
        from nsb2.emitter.airglow import from_eso_skycalc

        src = from_eso_skycalc(height=90 * u.km, sfu=130)
        assert isinstance(src, LonLatSource)

    def test_name_contains_airglow(self):
        from nsb2.emitter.airglow import from_eso_skycalc

        src = from_eso_skycalc(height=90 * u.km, sfu=130)
        assert "airglow" in src.name.lower()

    def test_spectral_grid_has_wavelengths(self):
        from nsb2.emitter.airglow import from_eso_skycalc

        src = from_eso_skycalc(height=90 * u.km, sfu=130)
        assert len(src.spectral_grid.wvl) > 0

    def test_query_scattered_returns_field(self):
        from nsb2.emitter.airglow import from_eso_skycalc

        src = from_eso_skycalc(height=90 * u.km, sfu=130)
        obs = make_observation()
        field = src.query_scattered(obs, nside=8)
        assert field.radiance_field is True

    def test_different_sfu_different_weights(self):
        from nsb2.emitter.airglow import from_eso_skycalc

        src_low = from_eso_skycalc(height=90 * u.km, sfu=70)
        src_high = from_eso_skycalc(height=90 * u.km, sfu=200)
        obs = make_observation()
        field_low = src_low.query_scattered(obs, nside=4)
        field_high = src_high.query_scattered(obs, nside=4)
        # Higher SFU → higher brightness
        assert np.sum(field_high.weights.value) > np.sum(field_low.weights.value)

    def test_van_rhijn_increases_toward_horizon(self):
        from nsb2.emitter.airglow import van_rhijn

        zenith = van_rhijn(90, 0.0)  # at zenith (z=0)
        horizon = van_rhijn(90, np.deg2rad(80))  # near horizon
        assert horizon > zenith


# ---------------------------------------------------------------------------
# Zodiacal Light
# ---------------------------------------------------------------------------


def _make_fake_solar_spectrum():
    """Return a (wvl, spectrum) that covers zodiacal + ROLO range."""
    wvl = np.linspace(1000, 30000, 300) * u.angstrom
    spectrum = np.ones(300) * 1.8e-8 * u.erg / u.s / u.cm**2 / u.angstrom
    return wvl, spectrum


class TestZodiacalFromLeinert1998:
    def test_returns_lonlat_source(self):
        from nsb2.emitter.zodiacal import from_leinert1998

        with patch("nsb2.emitter.zodiacal.SolarSpectrumRieke2008") as mock_solar:
            mock_solar.return_value = _make_fake_solar_spectrum()
            src = from_leinert1998()
        assert isinstance(src, LonLatSource)

    def test_name_contains_zodiacal(self):
        from nsb2.emitter.zodiacal import from_leinert1998

        with patch("nsb2.emitter.zodiacal.SolarSpectrumRieke2008") as mock_solar:
            mock_solar.return_value = _make_fake_solar_spectrum()
            src = from_leinert1998()
        assert "zodiacal" in src.name.lower() or "Zodiacal" in src.name

    def test_spectral_grid_has_wavelengths(self):
        from nsb2.emitter.zodiacal import from_leinert1998

        with patch("nsb2.emitter.zodiacal.SolarSpectrumRieke2008") as mock_solar:
            mock_solar.return_value = _make_fake_solar_spectrum()
            src = from_leinert1998()
        assert len(src.spectral_grid.wvl) > 0

    def test_query_scattered_returns_field(self):
        from nsb2.emitter.zodiacal import from_leinert1998

        with patch("nsb2.emitter.zodiacal.SolarSpectrumRieke2008") as mock_solar:
            mock_solar.return_value = _make_fake_solar_spectrum()
            src = from_leinert1998()
        obs = make_observation()
        field = src.query_scattered(obs, nside=4)
        assert field.radiance_field is True


# ---------------------------------------------------------------------------
# Moon
# ---------------------------------------------------------------------------


class TestMoonFromNoll2013:
    def test_returns_ephemeris_source(self):
        from nsb2.emitter.moon import from_noll2013

        with patch("nsb2.emitter.moon.SolarSpectrumRieke2008") as mock_solar:
            mock_solar.return_value = _make_fake_solar_spectrum()
            src = from_noll2013()
        assert isinstance(src, EphemerisSource)

    def test_body_is_moon(self):
        from nsb2.emitter.moon import from_noll2013

        with patch("nsb2.emitter.moon.SolarSpectrumRieke2008") as mock_solar:
            mock_solar.return_value = _make_fake_solar_spectrum()
            src = from_noll2013()
        assert src.body == "moon"

    def test_spectral_grid_shape(self):
        from nsb2.emitter.moon import from_noll2013

        with patch("nsb2.emitter.moon.SolarSpectrumRieke2008") as mock_solar:
            mock_solar.return_value = _make_fake_solar_spectrum()
            src = from_noll2013()
        # SpectralGrid has 1D parameter (phase angle) and 3 components
        assert src.spectral_grid.flx.ndim == 3
        assert src.spectral_grid.flx.shape[-1] == 3

    def test_query_direct_nighttime(self):
        """Moon query at nighttime observation (may be above or below horizon)."""
        from nsb2.emitter.moon import from_noll2013

        with patch("nsb2.emitter.moon.SolarSpectrumRieke2008") as mock_solar:
            mock_solar.return_value = _make_fake_solar_spectrum()
            src = from_noll2013()
        obs = make_observation()
        from nsb2.instrument.HESS import CT1

        inst = CT1()
        pix_coords = inst.pixel_coords(obs)
        field, refs = src.query_direct(obs, pix_coords, inst.pixel_radii())
        # Should return a SourceField (possibly empty if moon below horizon)
        from nsb2.core.dtypes import SourceField

        assert isinstance(field, SourceField)

    def test_query_scattered_nighttime(self):
        from nsb2.emitter.moon import from_noll2013

        with patch("nsb2.emitter.moon.SolarSpectrumRieke2008") as mock_solar:
            mock_solar.return_value = _make_fake_solar_spectrum()
            src = from_noll2013()
        obs = make_observation()
        from nsb2.core.dtypes import SourceField

        field = src.query_scattered(obs)
        assert isinstance(field, SourceField)
