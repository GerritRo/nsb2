import astropy.units as u
import numpy as np
import pytest
from astropy.coordinates import SkyCoord

from nsb2.core.dtypes import PixelRefs, SourceField
from nsb2.core.instrument import EffectiveApertureInstrument
from tests.conftest import make_bandpass, make_observation, make_spectral_grid


def _make_mock_response(n_pix=4, grid_size=5):
    """Create a mock instrument response suitable for EffectiveApertureInstrument."""
    x_arr, y_arr, v_arr = [], [], []
    for i in range(n_pix):
        cx = np.deg2rad(i * 0.5 - 0.75)
        cy = 0.0
        x = np.linspace(cx - 0.005, cx + 0.005, grid_size)
        y = np.linspace(cy - 0.005, cy + 0.005, grid_size)
        v = np.ones((grid_size, grid_size)) * 10.0
        x_arr.append(x)
        y_arr.append(y)
        v_arr.append(v)
    return {
        'x': np.array(x_arr),
        'y': np.array(y_arr),
        'values': np.array(v_arr),
    }


class TestEffectiveApertureInstrument:
    @pytest.fixture
    def instrument(self):
        resp = _make_mock_response()
        bp = make_bandpass()
        return EffectiveApertureInstrument(resp, bp)

    def test_n_pixels(self, instrument):
        assert instrument.n_pixels == 4

    def test_pixel_coords_returns_skycoord(self, instrument):
        obs = make_observation()
        pc = instrument.pixel_coords(obs)
        assert isinstance(pc, SkyCoord)
        assert len(pc) == 4

    def test_pixel_radii(self, instrument):
        radii = instrument.pixel_radii()
        assert len(radii) == 4
        assert np.all(radii > 0)

    def test_fov_range(self, instrument):
        lon_range, lat_range = instrument.fov_range()
        assert lon_range[0] < lon_range[1]
        assert lat_range[0] < lat_range[1]

    def test_compute_pixel_weights_point_source(self, instrument):
        sg = make_spectral_grid()
        obs = make_observation()
        # Place sources at small offsets within the FOV
        coords = SkyCoord([0.0, 0.001], [0.0, 0.0], unit='rad', frame=obs)
        field = SourceField(
            coords=coords,
            weights=np.ones((2, 1)) * u.dimensionless_unscaled,
            spectral_data=np.empty((2, 0)),
            spectral_grid=sg,
        )
        refs = PixelRefs(
            indices=[np.array([0, 1]), np.array([0]), np.array([]), np.array([])],
            weights=None,
        )
        refs = instrument.compute_pixel_weights(field, refs, obs)
        assert refs.weights is not None
        assert len(refs.weights) == 4
        assert refs.weights[0].unit == u.m**2

    def test_compute_pixel_weights_diffuse_source(self, instrument):
        sg = make_spectral_grid()
        obs = make_observation()
        coords = SkyCoord([0.0], [0.0], unit='rad', frame=obs)
        field = SourceField(
            coords=coords,
            weights=np.ones((1, 1)) * u.dimensionless_unscaled,
            spectral_data=np.empty((1, 0)),
            spectral_grid=sg,
            radiance_field=True,
        )
        refs = PixelRefs(
            indices=[np.array([0]), np.array([0]), np.array([0]), np.array([0])],
            weights=None,
        )
        refs = instrument.compute_pixel_weights(field, refs, obs)
        assert refs.weights is not None
        for w in refs.weights:
            assert u.m**2 * u.radian**2 == w.unit

    def test_project_discrete(self, instrument):
        rates = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]) / u.s
        refs = PixelRefs(
            indices=[np.array([0, 1]), np.array([0]), np.array([]), np.array([1])],
            weights=[np.array([0.5, 0.3]) * u.m**2, np.array([1.0]) * u.m**2,
                     np.array([]) * u.m**2, np.array([0.2]) * u.m**2],
        )
        result = instrument.project_discrete(rates, refs)
        assert result.shape == (4, 3)
        # Pixel 0: w[0]*min(rates[0]), w[0]*med(rates[0]) + w[1]*min(rates[1]) etc.
        assert not np.all(np.isnan(result.value[:2]))  # first two pixels have data
