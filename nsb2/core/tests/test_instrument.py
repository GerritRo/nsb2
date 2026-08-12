import astropy.units as u
import numpy as np
import pytest
from astropy.coordinates import SkyCoord

from nsb2.conftest import make_bandpass, make_spectral_grid
from nsb2.core.dtypes import PixelRefs, SourceField
from nsb2.core.instrument import EffectiveApertureInstrument, _min_med_max

LON_HALF_WIDTH = 0.04
LAT_HALF_WIDTH = 0.01
RESPONSE_VALUE = 3.0


def _asymmetric_response(n_pix=3, grid_size=7):
    """A flat response on a non-square offset grid."""
    centres = np.linspace(-LON_HALF_WIDTH, LON_HALF_WIDTH, n_pix)
    return {
        "x": np.array(
            [
                np.linspace(c - LON_HALF_WIDTH, c + LON_HALF_WIDTH, grid_size)
                for c in centres
            ]
        ),
        "y": np.array(
            [np.linspace(-LAT_HALF_WIDTH, LAT_HALF_WIDTH, grid_size) for _ in centres]
        ),
        "values": np.full((n_pix, grid_size, grid_size), RESPONSE_VALUE),
    }


@pytest.fixture
def asymmetric_instrument():
    return EffectiveApertureInstrument(_asymmetric_response(), make_bandpass())


class TestMinMedMax:
    def test_collapses_the_component_axis_to_min_median_max(self):
        assert _min_med_max(np.ones((5, 4))).shape == (5, 3)
        np.testing.assert_allclose(
            _min_med_max(np.array([[1.0, 5.0, 3.0]]))[0], [1.0, 3.0, 5.0]
        )
        # nan values are ignored rather than poisoning the whole pixel.
        np.testing.assert_allclose(
            _min_med_max(np.array([[1.0, np.nan, 3.0]]))[0], [1.0, 2.0, 3.0]
        )


class TestEffectiveApertureInstrument:
    def test_geometry_follows_the_response_grid(
        self, instrument, asymmetric_instrument, observation
    ):
        """``x`` is the longitude offset and ``y`` the latitude offset.

        :meth:`fov_range` reports them in that order.
        """
        coords = instrument.pixel_coords(observation)
        assert instrument.n_pixels == 4
        assert isinstance(coords, SkyCoord)
        assert len(coords) == instrument.n_pixels
        radii = instrument.pixel_radii()
        assert len(radii) == instrument.n_pixels
        assert np.all(radii > 0)

        lon_range, lat_range = asymmetric_instrument.fov_range()
        assert lon_range == pytest.approx((-2 * LON_HALF_WIDTH, 2 * LON_HALF_WIDTH))
        assert lat_range == pytest.approx((-LAT_HALF_WIDTH, LAT_HALF_WIDTH))

    def test_eval_grid_covers_every_pixel(self, asymmetric_instrument, observation):
        """Pixels outside the grid would have to be extrapolated onto."""
        grid = asymmetric_instrument.eval_grid(observation, n=3)
        assert grid.shape == (3, 3)

        grid = asymmetric_instrument.eval_grid(observation, n=2).transform_to(
            observation
        )
        pixels = asymmetric_instrument.pixel_coords(observation)
        assert grid.lon.rad.min() <= pixels.lon.rad.min()
        assert grid.lon.rad.max() >= pixels.lon.rad.max()
        assert grid.lat.rad.min() <= pixels.lat.rad.min()
        assert grid.lat.rad.max() >= pixels.lat.rad.max()

    def test_pixel_solid_angle_integrates_over_both_offset_axes(
        self, asymmetric_instrument
    ):
        """A flat response integrates to value times the area of the grid."""
        expected = RESPONSE_VALUE * (2 * LON_HALF_WIDTH) * (2 * LAT_HALF_WIDTH)
        np.testing.assert_allclose(asymmetric_instrument._pix_area_sr, expected)

    def test_compute_pixel_weights_for_point_and_diffuse_sources(
        self, instrument, observation
    ):
        """A point source is looked up in the response map and weighted in m2.

        A radiance fills the pixel, so it is weighted by m2 sr instead.
        """
        point = SourceField(
            coords=SkyCoord([0.0, 0.001], [0.0, 0.0], unit="rad", frame=observation),
            weights=np.ones((2, 1)) * u.dimensionless_unscaled,
            spectral_data=np.empty((2, 0)),
            spectral_grid=make_spectral_grid(),
        )
        refs = PixelRefs(
            indices=[np.array([0, 1]), np.array([0]), np.array([]), np.array([])],
            weights=None,
        )
        filled = instrument.compute_pixel_weights(point, refs, observation)
        assert filled.weights is not None
        assert len(filled.weights) == instrument.n_pixels
        assert filled.weights[0].unit == u.m**2
        assert refs.weights is None, "the input assignment must not be modified"

        diffuse = SourceField(
            coords=SkyCoord([0.0], [0.0], unit="rad", frame=observation),
            weights=np.ones((1, 1)) * u.dimensionless_unscaled,
            spectral_data=np.empty((1, 0)),
            spectral_grid=make_spectral_grid(),
            radiance_field=True,
        )
        filled = instrument.compute_pixel_weights(
            diffuse, PixelRefs(indices=[np.array([0])] * 4), observation
        )
        for weight in filled.weights:
            assert weight.unit == u.m**2 * u.radian**2

    def test_project_discrete_sums_the_sources_of_each_pixel(self, instrument):
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
        np.testing.assert_allclose(
            result[0].value, [0.5 * 1 + 0.3 * 4, 0.5 * 2 + 0.3 * 5, 0.5 * 3 + 0.3 * 6]
        )
        # A pixel with no sources must be exactly zero, not nan.
        np.testing.assert_array_equal(result[2].value, np.zeros(3))

        with pytest.raises(ValueError, match="weights"):
            instrument.project_discrete(
                np.ones((1, 3)) / u.s, PixelRefs(indices=[np.array([0])] * 4)
            )

    def test_project_continuous_interpolates_onto_the_pixels(
        self, instrument, observation
    ):
        eval_coords = instrument.eval_grid(observation, n=2)
        result = instrument.project_continuous(np.ones((2, 2, 1, 3)) / u.s, eval_coords)
        assert result.shape == (4, 3)
        assert result.unit.is_equivalent(u.m**2 * u.sr / u.s)
