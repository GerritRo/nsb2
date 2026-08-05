"""Tests for CTAO and HESS instrument factory functions."""

import astropy.units as u
import numpy as np
import pytest
from astropy.coordinates import SkyCoord

from nsb2.core.instrument import EffectiveApertureInstrument
from nsb2.core.spectral import Bandpass
from nsb2.instrument import BANDPASS_PATH, RESPONSE_PATH
from nsb2.instrument.CTAO import LST1North, MSTNorth
from nsb2.instrument.HESS import CT1
from tests.conftest import make_observation


class TestCTAOInstruments:
    def test_lst1north_returns_instrument(self):
        inst = LST1North()
        assert isinstance(inst, EffectiveApertureInstrument)

    def test_lst1north_has_pixels(self):
        inst = LST1North()
        assert inst.n_pixels > 0

    def test_lst1north_bandpass(self):
        inst = LST1North()
        assert isinstance(inst.bandpass, Bandpass)

    def test_lst1north_pixel_coords(self):
        inst = LST1North()
        obs = make_observation()
        coords = inst.pixel_coords(obs)
        assert isinstance(coords, SkyCoord)
        assert len(coords) == inst.n_pixels

    def test_lst1north_pixel_radii(self):
        inst = LST1North()
        radii = inst.pixel_radii()
        assert len(radii) == inst.n_pixels
        assert np.all(radii > 0)

    def test_lst1north_fov_range(self):
        inst = LST1North()
        lon_range, lat_range = inst.fov_range()
        assert lon_range[0] < lon_range[1]
        assert lat_range[0] < lat_range[1]

    def test_mstnorth_returns_instrument(self):
        inst = MSTNorth()
        assert isinstance(inst, EffectiveApertureInstrument)

    def test_mstnorth_has_pixels(self):
        inst = MSTNorth()
        assert inst.n_pixels > 0

    def test_mstnorth_bandpass(self):
        inst = MSTNorth()
        assert isinstance(inst.bandpass, Bandpass)


class TestHESSInstruments:
    def test_ct1_returns_instrument(self):
        inst = CT1()
        assert isinstance(inst, EffectiveApertureInstrument)

    def test_ct1_has_pixels(self):
        inst = CT1()
        assert inst.n_pixels > 0

    def test_ct1_bandpass(self):
        inst = CT1()
        assert isinstance(inst.bandpass, Bandpass)

    def test_ct1_pixel_coords(self):
        inst = CT1()
        obs = make_observation()
        coords = inst.pixel_coords(obs)
        assert isinstance(coords, SkyCoord)
        assert len(coords) == inst.n_pixels

    def test_ct1_pixel_radii(self):
        inst = CT1()
        radii = inst.pixel_radii()
        assert len(radii) == inst.n_pixels
        assert np.all(radii > 0)


class TestBandpassFromCsv:
    def test_from_csv_lst_like(self):
        bp = Bandpass.from_csv(BANDPASS_PATH / "LST_like.dat")
        assert isinstance(bp, Bandpass)
        assert bp.lam.unit == u.nm
        assert len(bp.lam) > 0
        assert np.all(bp.trx >= 0)

    def test_from_csv_mst_like(self):
        bp = Bandpass.from_csv(BANDPASS_PATH / "MST_like.dat")
        assert isinstance(bp, Bandpass)
        assert bp.min < bp.max

    def test_from_csv_hess_ct1(self):
        bp = Bandpass.from_csv(BANDPASS_PATH / "hess1u_ct1.dat")
        assert isinstance(bp, Bandpass)


class TestInstrumentEvalGrid:
    @pytest.fixture
    def instrument(self):
        return LST1North()

    def test_eval_grid_shape(self, instrument):
        obs = make_observation()
        grid = instrument.eval_grid(obs, n=3)
        assert grid.shape == (3, 3)

    def test_eval_grid_is_skycoord(self, instrument):
        obs = make_observation()
        grid = instrument.eval_grid(obs, n=2)
        assert isinstance(grid, SkyCoord)

    def test_eval_grid_n1(self, instrument):
        obs = make_observation()
        grid = instrument.eval_grid(obs, n=1)
        assert grid.shape == (1, 1)
