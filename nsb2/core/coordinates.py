from typing import ClassVar

import astropy.units as u
from astropy.coordinates import (
    BaseCoordinateFrame,
    FunctionTransform,
    GeocentricTrueEcliptic,
    RepresentationMapping,
    SphericalRepresentation,
    TimeAttribute,
    frame_transform_graph,
    get_body,
)

__all__ = [
    "SunRelativeEclipticFrame",
    "gte_to_sunrel",
    "sunrel_to_gte",
]


class SunRelativeEclipticFrame(BaseCoordinateFrame):
    """Ecliptic coordinates with longitude measured from the Sun.

    Longitude ``alpha`` is the ecliptic longitude minus the Sun's;
    latitude ``beta`` is the ecliptic latitude.

    Parameters
    ----------
    obstime : astropy.time.Time
        Time of observation, needed to locate the Sun.  Required for any
        transformation into or out of this frame.

    Notes
    -----
    Like every astropy spherical frame, ``alpha`` is stored normalised to
    ``[0, 360)`` degrees.  Solar elongation is therefore the *wrapped*
    absolute value of ``alpha``, not ``alpha`` itself: a direction ten
    degrees west of the Sun is stored as 350 degrees, not as -10.

    Examples
    --------
    >>> import astropy.units as u
    >>> from astropy.coordinates import SkyCoord
    >>> from astropy.time import Time
    >>> t = Time("2024-06-15T22:00:00")
    >>> sc = SkyCoord(10 * u.deg, 20 * u.deg, frame="icrs")
    >>> rel = sc.transform_to(SunRelativeEclipticFrame(obstime=t))
    >>> bool(0 * u.deg <= rel.alpha < 360 * u.deg)
    True
    """

    default_representation = SphericalRepresentation

    obstime = TimeAttribute(default=None)

    frame_specific_representation_info: ClassVar = {
        SphericalRepresentation: [
            RepresentationMapping("lon", "alpha"),
            RepresentationMapping("lat", "beta"),
            RepresentationMapping("distance", "distance"),
        ]
    }


@frame_transform_graph.transform(
    FunctionTransform, GeocentricTrueEcliptic, SunRelativeEclipticFrame
)
def gte_to_sunrel(gte_coords, sunrel_frame):
    """Transform geocentric true ecliptic to Sun-relative ecliptic coordinates.

    Parameters
    ----------
    gte_coords : astropy.coordinates.GeocentricTrueEcliptic
        Coordinates to transform.  Must carry an ``obstime``.
    sunrel_frame : SunRelativeEclipticFrame
        Target frame.

    Returns
    -------
    SunRelativeEclipticFrame
        The transformed coordinates.

    Raises
    ------
    ValueError
        If ``gte_coords`` has no ``obstime``, so the Sun cannot be located.
    """
    obstime = gte_coords.obstime
    if obstime is None:
        raise ValueError("GeocentricTrueEcliptic coords must have obstime")

    sun = get_body("sun", obstime)
    sun_ecl = sun.transform_to(GeocentricTrueEcliptic(obstime=obstime))

    alpha = (gte_coords.lon - sun_ecl.lon).wrap_at(180 * u.deg)
    beta = gte_coords.lat
    distance = gte_coords.distance if gte_coords.distance.unit != u.one else None

    return SunRelativeEclipticFrame(
        alpha=alpha, beta=beta, distance=distance, obstime=obstime
    )


@frame_transform_graph.transform(
    FunctionTransform, SunRelativeEclipticFrame, GeocentricTrueEcliptic
)
def sunrel_to_gte(sunrel_coords, gte_frame):
    """Transform Sun-relative ecliptic to geocentric true ecliptic coordinates.

    Parameters
    ----------
    sunrel_coords : SunRelativeEclipticFrame
        Coordinates to transform.  Must carry an ``obstime``.
    gte_frame : astropy.coordinates.GeocentricTrueEcliptic
        Target frame.

    Returns
    -------
    astropy.coordinates.GeocentricTrueEcliptic
        The transformed coordinates.

    Raises
    ------
    ValueError
        If ``sunrel_coords`` has no ``obstime``, so the Sun cannot be located.
    """
    obstime = sunrel_coords.obstime
    if obstime is None:
        raise ValueError("SunRelativeEclipticFrame must have obstime")

    sun_ecl = get_body("sun", obstime).transform_to(
        GeocentricTrueEcliptic(obstime=obstime)
    )

    lon = (sun_ecl.lon + sunrel_coords.alpha).wrap_at(360 * u.deg)
    lat = sunrel_coords.beta
    distance = sunrel_coords.distance if sunrel_coords.distance.unit != u.one else None

    return GeocentricTrueEcliptic(
        lon=lon, lat=lat, distance=distance, obstime=obstime, equinox=gte_frame.equinox
    )
