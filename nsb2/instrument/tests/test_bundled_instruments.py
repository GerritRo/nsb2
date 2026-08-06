"""Tests for the bundled instrument models in :mod:`nsb2.instrument`."""

import astropy.units as u
import numpy as np
import pytest

from nsb2.instrument import BANDPASS_PATH, RESPONSE_PATH
from nsb2.instrument.CTAO import LST1North, MSTNorth
from nsb2.instrument.HESS import CT1

FACTORIES = [LST1North, MSTNorth, CT1]


class TestDataFiles:
    def test_bundled_directories_exist(self):
        assert BANDPASS_PATH.is_dir()
        assert RESPONSE_PATH.is_dir()

    def test_every_response_has_a_bandpass(self):
        assert list(RESPONSE_PATH.glob("*.npz"))
        assert list(BANDPASS_PATH.glob("*.dat"))


@pytest.mark.parametrize("factory", FACTORIES, ids=lambda f: f.__name__)
class TestBundledInstruments:
    def test_loads(self, factory):
        assert factory().n_pixels > 0

    def test_pixel_radii_are_positive(self, factory):
        assert np.all(factory().pixel_radii() > 0)

    def test_fov_is_ordered_and_of_plausible_size(self, factory):
        lon_range, lat_range = factory().fov_range()
        assert lon_range[0] < lon_range[1]
        assert lat_range[0] < lat_range[1]
        # Every IACT camera covers between one and twenty degrees.
        assert 1 < np.rad2deg(lon_range[1] - lon_range[0]) < 20

    def test_pixel_solid_angles_are_positive(self, factory):
        assert np.all(factory()._pix_area_sr > 0)

    def test_bandpass_covers_the_cherenkov_band(self, factory):
        """Cherenkov light peaks in the near UV, so the band must reach it."""
        bandpass = factory().bandpass
        assert bandpass.min <= 350 * u.nm
        assert bandpass.max >= 500 * u.nm

    def test_pixel_coords_match_pixel_count(self, factory, observation):
        instrument = factory()
        assert len(instrument.pixel_coords(observation)) == instrument.n_pixels
