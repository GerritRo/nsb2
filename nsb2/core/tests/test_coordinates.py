"""Tests for :mod:`nsb2.core.coordinates`."""

import astropy.units as u
import pytest
from astropy.coordinates import GeocentricTrueEcliptic, SkyCoord, get_body
from astropy.time import Time

from nsb2.core.coordinates import SunRelativeEclipticFrame

OBSTIME = Time("2024-06-15T22:00:00")


class TestSunRelativeEclipticFrame:
    def test_roundtrip_preserves_position(self):
        original = SkyCoord(120 * u.deg, 15 * u.deg, frame="icrs")
        relative = original.transform_to(SunRelativeEclipticFrame(obstime=OBSTIME))
        back = SkyCoord(relative).transform_to("icrs")
        assert back.separation(original).arcsec == pytest.approx(0.0, abs=1e-6)

    def test_longitude_is_measured_from_the_sun(self):
        """The Sun itself must sit at zero relative longitude."""
        sun = get_body("sun", OBSTIME)
        relative = sun.transform_to(SunRelativeEclipticFrame(obstime=OBSTIME))
        assert relative.alpha.deg == pytest.approx(0.0, abs=0.5)

    def test_latitude_matches_ecliptic_latitude(self):
        coord = SkyCoord(200 * u.deg, -30 * u.deg, frame="icrs")
        ecliptic = coord.transform_to(GeocentricTrueEcliptic(obstime=OBSTIME))
        relative = coord.transform_to(SunRelativeEclipticFrame(obstime=OBSTIME))
        assert relative.beta.deg == pytest.approx(ecliptic.lat.deg, abs=1e-6)

    def test_longitude_is_normalised_like_any_astropy_frame(self):
        coord = SkyCoord(10 * u.deg, 0 * u.deg, frame="icrs")
        relative = coord.transform_to(SunRelativeEclipticFrame(obstime=OBSTIME))
        assert 0 <= relative.alpha.deg < 360

    def test_reverse_transform_yields_a_frame_not_a_skycoord(self):
        """A FunctionTransform must return an instance of the target frame."""
        relative = SkyCoord(
            120 * u.deg, 15 * u.deg, frame=SunRelativeEclipticFrame(obstime=OBSTIME)
        )
        back = relative.transform_to(GeocentricTrueEcliptic(obstime=OBSTIME))
        assert back.frame.name == "geocentrictrueecliptic"

    def test_reverse_transform_inverts_the_forward_one(self):
        ecliptic = SkyCoord(
            200 * u.deg, -30 * u.deg, frame=GeocentricTrueEcliptic(obstime=OBSTIME)
        )
        relative = ecliptic.transform_to(SunRelativeEclipticFrame(obstime=OBSTIME))
        back = SkyCoord(relative).transform_to(GeocentricTrueEcliptic(obstime=OBSTIME))
        assert back.separation(ecliptic).arcsec == pytest.approx(0.0, abs=1e-6)

    def test_forward_transform_without_obstime_raises(self):
        """The Sun cannot be located without a time, in either direction.

        Called directly rather than through ``transform_to``: astropy
        substitutes J2000 for a ``None`` obstime on
        `~astropy.coordinates.GeocentricTrueEcliptic`, so the guard is
        defensive and cannot be reached through the frame API.
        """
        from nsb2.core.coordinates import gte_to_sunrel

        class WithoutObstime:
            obstime = None

        with pytest.raises(ValueError, match="obstime"):
            gte_to_sunrel(WithoutObstime(), SunRelativeEclipticFrame(obstime=OBSTIME))

    def test_reverse_transform_without_obstime_raises(self):
        coord = SkyCoord(
            10 * u.deg, 0 * u.deg, frame=SunRelativeEclipticFrame(obstime=None)
        )
        with pytest.raises(ValueError, match="obstime"):
            coord.transform_to(GeocentricTrueEcliptic(obstime=OBSTIME))
