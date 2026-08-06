"""Tests for :mod:`nsb2.core.instrument`."""

import astropy.units as u
import numpy as np
import pytest
from astropy.coordinates import SkyCoord

from nsb2.conftest import make_spectral_grid
from nsb2.core.dtypes import PixelRefs, SourceField
from nsb2.core.instrument import _min_med_max


class TestMinMedMax:
    def test_collapses_component_axis_to_three(self):
        assert _min_med_max(np.ones((5, 4))).shape == (5, 3)

    def test_orders_min_median_max(self):
        result = _min_med_max(np.array([[1.0, 5.0, 3.0]]))
        np.testing.assert_allclose(result[0], [1.0, 3.0, 5.0])

    def test_ignores_nan(self):
        result = _min_med_max(np.array([[1.0, np.nan, 3.0]]))
        np.testing.assert_allclose(result[0], [1.0, 2.0, 3.0])


class TestEffectiveApertureInstrument:
    def test_n_pixels(self, instrument):
        assert instrument.n_pixels == 4

    def test_pixel_coords_returns_skycoord(self, instrument, observation):
        coords = instrument.pixel_coords(observation)
        assert isinstance(coords, SkyCoord)
        assert len(coords) == 4

    def test_pixel_radii(self, instrument):
        radii = instrument.pixel_radii()
        assert len(radii) == 4
        assert np.all(radii > 0)

    def test_fov_range(self, instrument):
        lon_range, lat_range = instrument.fov_range()
        assert lon_range[0] < lon_range[1]
        assert lat_range[0] < lat_range[1]

    def test_eval_grid_shape(self, instrument, observation):
        assert instrument.eval_grid(observation, n=3).shape == (3, 3)

    def test_eval_grid_spans_fov(self, instrument, observation):
        grid = instrument.eval_grid(observation, n=2)
        assert len(grid.flatten()) == 4

    def test_compute_pixel_weights_point_source(self, instrument, observation):
        field = SourceField(
            coords=SkyCoord([0.0, 0.001], [0.0, 0.0], unit="rad", frame=observation),
            weights=np.ones((2, 1)) * u.dimensionless_unscaled,
            spectral_data=np.empty((2, 0)),
            spectral_grid=make_spectral_grid(),
        )
        refs = PixelRefs(
            indices=[np.array([0, 1]), np.array([0]), np.array([]), np.array([])],
            weights=None,
        )
        filled = instrument.compute_pixel_weights(field, refs, observation)
        assert filled.weights is not None
        assert len(filled.weights) == 4
        assert filled.weights[0].unit == u.m**2

    def test_compute_pixel_weights_does_not_modify_input(self, instrument, observation):
        field = SourceField(
            coords=SkyCoord([0.0], [0.0], unit="rad", frame=observation),
            weights=np.ones((1, 1)) * u.dimensionless_unscaled,
            spectral_data=np.empty((1, 0)),
            spectral_grid=make_spectral_grid(),
            radiance_field=True,
        )
        refs = PixelRefs(indices=[np.array([0])] * 4, weights=None)
        instrument.compute_pixel_weights(field, refs, observation)
        assert refs.weights is None

    def test_compute_pixel_weights_diffuse_source(self, instrument, observation):
        field = SourceField(
            coords=SkyCoord([0.0], [0.0], unit="rad", frame=observation),
            weights=np.ones((1, 1)) * u.dimensionless_unscaled,
            spectral_data=np.empty((1, 0)),
            spectral_grid=make_spectral_grid(),
            radiance_field=True,
        )
        refs = PixelRefs(indices=[np.array([0])] * 4, weights=None)
        filled = instrument.compute_pixel_weights(field, refs, observation)
        assert filled.weights is not None
        for weight in filled.weights:
            assert weight.unit == u.m**2 * u.radian**2

    def test_project_discrete(self, instrument):
        rates = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]) / u.s
        refs = PixelRefs(
            indices=[np.array([0, 1]), np.array([0]), np.array([]), np.array([1])],
            weights=[
                np.array([0.5, 0.3]) * u.m**2,
                np.array([1.0]) * u.m**2,
                np.array([]) * u.m**2,
                np.array([0.2]) * u.m**2,
            ],
        )
        result = instrument.project_discrete(rates, refs)
        assert result.shape == (4, 3)
        # Pixel 0 sums both sources; pixel 2 has none and must be exactly zero.
        np.testing.assert_allclose(
            result[0].value, [0.5 * 1 + 0.3 * 4, 0.5 * 2 + 0.3 * 5, 0.5 * 3 + 0.3 * 6]
        )
        np.testing.assert_array_equal(result[2].value, np.zeros(3))

    def test_project_discrete_requires_weights(self, instrument):
        refs = PixelRefs(indices=[np.array([0])] * 4, weights=None)
        with pytest.raises(ValueError, match="weights"):
            instrument.project_discrete(np.ones((1, 3)) / u.s, refs)

    def test_project_continuous(self, instrument, observation):
        eval_coords = instrument.eval_grid(observation, n=2)
        rates = np.ones((2, 2, 1, 3)) / u.s
        result = instrument.project_continuous(rates, eval_coords)
        assert result.shape == (4, 3)
        assert result.unit.is_equivalent(u.m**2 * u.sr / u.s)
