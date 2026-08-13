import astropy.units as u
import numpy as np

from nsb2.core.sources import LonLatSource
from nsb2.core.spectral import SpectralGrid

from .. import ASSETS_PATH

__all__ = [
    "from_eso_skycalc",
    "van_rhijn",
]


EARTH_RADIUS = 6738


def van_rhijn(height_km, zenith_angle):
    """Line-of-sight enhancement of emission from a thin spherical shell.

    Airglow originates in a thin layer of the upper atmosphere.  Looking
    towards the horizon, the line of sight crosses that layer at a shallower
    angle and so traverses more of it, which brightens the emission.  The van
    Rhijn function [vanRhijn1921]_ gives that enhancement relative to zenith.

    Parameters
    ----------
    height_km : float
        Height of the emitting layer above the surface, in kilometres.
    zenith_angle : array_like
        Zenith angle of the line of sight, in radians.

    Returns
    -------
    numpy.ndarray
        Enhancement factor, equal to one at zenith and rising towards the
        horizon.

    Examples
    --------
    >>> float(van_rhijn(90, 0.0))
    1.0
    """
    r_rh = EARTH_RADIUS / (EARTH_RADIUS + height_km)
    return 1 / (1 - r_rh**2 * np.sin(zenith_angle) ** 2) ** 0.5


@u.quantity_input(height=u.m)
def from_eso_skycalc(height: u.Quantity, sfu: float) -> LonLatSource:
    """Build an airglow source from the ESO SkyCalc reference spectrum.

    The tabulated spectrum ships with ``nsb2`` and was generated with the ESO
    SkyCalc sky model [Noll2012]_ at a solar radio flux of 130 sfu.  It is
    rescaled to the requested activity level using a linear fit to [Noll2012]_
    Figure 14, and brightened towards the horizon by :func:`van_rhijn`.

    Parameters
    ----------
    height : astropy.units.Quantity
        Height of the emitting layer above the surface, in any length unit.
        Around 90 km for the dominant hydroxyl and oxygen emission.
    sfu : float
        Solar radio flux at 10.7 cm in solar flux units, as a proxy for solar
        activity.

    Returns
    -------
    nsb2.core.sources.LonLatSource
        Airglow, defined in the observation's own horizon frame.

    Raises
    ------
    astropy.units.UnitsError
        If ``height`` is not a length.
    """
    ag_array = np.genfromtxt(ASSETS_PATH / "eso_skycalc_airglow_130sfu.dat")
    spectral = SpectralGrid(
        [],
        ag_array[:, 0] * u.nm,
        np.atleast_2d(ag_array[:, 1]).T / u.s / u.m**2 / u.micron / u.arcsec**2,
    )

    def weight_function(lon, lat):
        """Scale the reference spectrum for solar activity and zenith angle."""
        return (
            van_rhijn(height.to_value(u.km), np.pi / 2 - lat)
            * (0.2 + 0.00614 * sfu)
            * u.dimensionless_unscaled
        )

    def data_function(lon, lat):
        """Return no spectral coordinates; the spectrum does not vary."""
        return np.empty((len(lat), 0))

    return LonLatSource(
        None, weight_function, data_function, spectral, name="airglow_eso_skycalc"
    )
