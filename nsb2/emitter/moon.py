import astropy.units as u
import numpy as np
from astropy.coordinates import get_body, get_body_barycentric
from scipy.interpolate import UnivariateSpline

from nsb2.core.photometry import SolarSpectrumRieke2008
from nsb2.core.sources import EphemerisSource
from nsb2.core.spectral import SpectralGrid

from .. import ASSETS_PATH


def from_noll2013():
    # Load ROLO albedo coefficients
    rolo = np.genfromtxt(ASSETS_PATH / "noll2013_lunar_rolo.dat", delimiter=",")

    # Load solar spectrum
    wvl, spectrum = SolarSpectrumRieke2008()

    # Mask to ROLO range
    mask = (wvl >= rolo[0,0]*u.nm) & (wvl <= rolo[-1,0]*u.nm)
    wvl = wvl[mask]
    spectrum = spectrum[mask]

    def lnA(p, g, s_sel):
        p_1, p_2, p_3, p_4 = 4.06054, 12.8802, np.deg2rad(-30.5858), np.deg2rad(16.7498)

        sum_a = p[0] + p[1] * g + p[2] * g**2 + p[3] * g**3
        sum_b = p[4] * s_sel + p[5] * s_sel**3 + p[6] * s_sel**5
        sum_c = (
            p[7] * np.exp(-g / p_1)
            + p[8] * np.exp(-g / p_2)
            + p[9] * np.cos((g - p_3) / p_4)
        )

        return sum_a + sum_b + sum_c

    def weight_function(obstime):
        moon = get_body("moon", obstime)
        sun = get_body("sun", obstime)

        d_earth_moon = moon.distance.to(u.km)
        d_sun_moon = sun.distance.to(u.AU)

        omega_moon_std = 6.4236e-5  # sr at 384400 km

        # Distance corrections
        sun_factor = (1.0*u.AU / d_sun_moon) ** 2
        obs_factor = (384400*u.km / d_earth_moon) ** 2

        return np.atleast_1d(1/np.pi * omega_moon_std * obs_factor * sun_factor) * u.dimensionless_unscaled

    def data_function(obstime):
        pos = {b: get_body_barycentric(b, obstime).xyz.value for b in ("sun", "moon", "earth")}

        v_sun = pos["sun"] - pos["moon"]
        v_earth = pos["earth"] - pos["moon"]

        cos_g = np.dot(v_sun, v_earth) / (np.linalg.norm(v_sun) * np.linalg.norm(v_earth))

        return np.atleast_1d(np.arccos(np.clip(cos_g, -1, 1)))

    def calc_albedo(lam, g, s_sel):
        res = []
        for j in range(25):
            res.append(np.exp(lnA(rolo[j][1:], g, s_sel))*0.87) # 13% reduction as in Noll2013

        s = UnivariateSpline(rolo[:, 0], np.asarray(res), k=1, s=0)
        return s(lam)

    # Calculate for different min/med/max libration:
    g_array = np.deg2rad(np.linspace(0,180,50))
    spec = np.zeros((len(g_array), len(wvl), 3))
    for i, g in enumerate(g_array):
        spec[i,:,0] = calc_albedo(wvl.to(u.nm), g, np.pi-(g-np.deg2rad(8)))*spectrum
        spec[i,:,1] = calc_albedo(wvl.to(u.nm), g, np.pi-(g))*spectrum
        spec[i,:,2] = calc_albedo(wvl.to(u.nm), g, np.pi-(g+np.deg2rad(8)))*spectrum

    spec = spec * spectrum.unit
    photon_spec = spec.to(u.photon / (u.nm * u.s * u.cm**2),
                          equivalencies=u.spectral_density(wvl[np.newaxis,:,np.newaxis]))/u.photon

    spectral = SpectralGrid([g_array], wvl, photon_spec)

    return EphemerisSource('moon', weight_function, data_function, spectral)
