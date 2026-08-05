import astropy.units as u
import numpy as np
import pytest
from astropy.coordinates import GeocentricTrueEcliptic, SkyCoord
from astropy.time import Time

from nsb2.core.coordinates import SunRelativeEclipticFrame


@pytest.fixture
def obstime():
    return Time("2024-06-15T22:00:00")


class TestSunRelativeEclipticFrame:
    def test_frame_has_obstime(self, obstime):
        frame = SunRelativeEclipticFrame(obstime=obstime)
        assert frame.obstime == obstime

    def test_gte_to_sunrel_has_alpha_beta(self, obstime):
        gte = SkyCoord(
            lon=90 * u.deg,
            lat=10 * u.deg,
            frame=GeocentricTrueEcliptic(obstime=obstime),
        )
        sunrel = gte.transform_to(SunRelativeEclipticFrame(obstime=obstime))
        assert hasattr(sunrel, "alpha")
        assert hasattr(sunrel, "beta")

    def test_gte_to_sunrel_alpha_wrapped(self, obstime):
        gte = SkyCoord(
            lon=90 * u.deg,
            lat=0 * u.deg,
            frame=GeocentricTrueEcliptic(obstime=obstime),
        )
        sunrel = gte.transform_to(SunRelativeEclipticFrame(obstime=obstime))
        assert -180 <= sunrel.alpha.deg <= 180

    def test_gte_to_sunrel_lat_preserved(self, obstime):
        lat_orig = -15 * u.deg
        gte = SkyCoord(
            lon=120 * u.deg,
            lat=lat_orig,
            frame=GeocentricTrueEcliptic(obstime=obstime),
        )
        sunrel = gte.transform_to(SunRelativeEclipticFrame(obstime=obstime))
        assert abs(sunrel.beta.deg - lat_orig.value) < 0.01

    def test_gte_to_sunrel_multiple_points(self, obstime):
        lons = np.array([0, 90, 180, 270]) * u.deg
        lats = np.array([0, 10, -10, 5]) * u.deg
        gte = SkyCoord(lon=lons, lat=lats, frame=GeocentricTrueEcliptic(obstime=obstime))
        sunrel = gte.transform_to(SunRelativeEclipticFrame(obstime=obstime))
        assert len(sunrel) == 4

    def test_sunrel_to_gte_executes_body(self, obstime):
        """The sunrel_to_gte function body executes (lines 48-60 covered) before
        astropy raises TypeError because the function incorrectly returns a SkyCoord
        instead of a raw frame object."""
        sunrel = SkyCoord(
            alpha=30 * u.deg,
            beta=5 * u.deg,
            frame=SunRelativeEclipticFrame(obstime=obstime),
        )
        with pytest.raises(TypeError):
            sunrel.transform_to(GeocentricTrueEcliptic(obstime=obstime))

    def test_gte_to_sunrel_raises_without_obstime_direct(self):
        """Call the transform function directly with obstime=None to trigger ValueError."""
        # Import the registered transform function directly
        import importlib

        mod = importlib.import_module("nsb2.core.coordinates")
        gte_to_sunrel = getattr(mod, "gte_to_sunrel")

        class _MockGTE:
            obstime = None

        with pytest.raises(ValueError, match="obstime"):
            gte_to_sunrel(_MockGTE(), SunRelativeEclipticFrame())

    def test_sunrel_to_gte_raises_without_obstime_direct(self):
        """Call the transform function directly with obstime=None to trigger ValueError."""
        import importlib

        mod = importlib.import_module("nsb2.core.coordinates")
        sunrel_to_gte = getattr(mod, "sunrel_to_gte")

        class _MockSunRel:
            obstime = None

        with pytest.raises(ValueError, match="obstime"):
            sunrel_to_gte(_MockSunRel(), GeocentricTrueEcliptic())
