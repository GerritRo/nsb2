import astropy.units as u
import pytest
from astropy.coordinates import GeocentricTrueEcliptic, SkyCoord, get_body
from astropy.time import Time

from nsb2.core.coordinates import SunRelativeEclipticFrame

OBSTIME = Time("2024-06-15T22:00:00")


class TestSunRelativeEclipticFrame:
    def test_roundtrips_through_both_neighbouring_frames(self):
        original = SkyCoord(120 * u.deg, 15 * u.deg, frame="icrs")
        back = SkyCoord(
            original.transform_to(SunRelativeEclipticFrame(obstime=OBSTIME))
        ).transform_to("icrs")
        assert back.separation(original).arcsec == pytest.approx(0.0, abs=1e-6)

        ecliptic = SkyCoord(
            200 * u.deg, -30 * u.deg, frame=GeocentricTrueEcliptic(obstime=OBSTIME)
        )
        relative = ecliptic.transform_to(SunRelativeEclipticFrame(obstime=OBSTIME))
        back = SkyCoord(relative).transform_to(GeocentricTrueEcliptic(obstime=OBSTIME))
        assert back.separation(ecliptic).arcsec == pytest.approx(0.0, abs=1e-6)
        # A FunctionTransform must return an instance of the target frame.
        assert back.frame.name == "geocentrictrueecliptic"

    def test_longitude_is_measured_from_the_sun(self):
        """The Sun sits at zero relative longitude.

        The latitude is untouched, and the longitude is normalised to
        [0, 360) like any astropy frame.
        """
        sun = get_body("sun", OBSTIME)
        assert sun.transform_to(
            SunRelativeEclipticFrame(obstime=OBSTIME)
        ).alpha.deg == pytest.approx(0.0, abs=0.5)

        coord = SkyCoord(200 * u.deg, -30 * u.deg, frame="icrs")
        ecliptic = coord.transform_to(GeocentricTrueEcliptic(obstime=OBSTIME))
        relative = coord.transform_to(SunRelativeEclipticFrame(obstime=OBSTIME))
        assert relative.beta.deg == pytest.approx(ecliptic.lat.deg, abs=1e-6)
        assert 0 <= relative.alpha.deg < 360
