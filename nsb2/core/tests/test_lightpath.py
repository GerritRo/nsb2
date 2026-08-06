"""Tests for :mod:`nsb2.core.lightpath`."""

import astropy.units as u
import numpy as np
import pytest

from nsb2.conftest import make_spectral_grid
from nsb2.core.lightpath import DirectPath, LightPath, ScatteredPath
from nsb2.core.solver import (
    ExplicitDirectSolver,
    ExplicitScatteredSolver,
    LUTDirectSolver,
)
from nsb2.core.sources import EphemerisSource


class TestDirectPath:
    def test_defaults_to_explicit_solver(self):
        assert isinstance(DirectPath().solver, ExplicitDirectSolver)

    def test_name_defaults_to_class_name(self):
        assert DirectPath().name == "DirectPath"

    def test_custom_name_is_kept(self):
        assert DirectPath(name="stars").name == "stars"

    def test_rejects_scattered_solver(self):
        with pytest.raises(TypeError, match="DirectSolver"):
            DirectPath(solver=ExplicitScatteredSolver())

    def test_compile_delegates_to_solver(self, diffuse_source, instrument, atmosphere):
        solver = LUTDirectSolver()
        DirectPath(solver=solver).compile(diffuse_source, instrument, atmosphere)
        assert diffuse_source in solver._luts

    def test_compile_is_zero_for_explicit_solver(
        self, diffuse_source, instrument, atmosphere
    ):
        assert DirectPath().compile(diffuse_source, instrument, atmosphere) == 0

    def test_zero_rates_when_no_source_in_view(
        self, instrument, atmosphere, observation
    ):
        """A body below the horizon must give zero rates rather than an error."""
        sun = EphemerisSource(
            "sun",
            lambda t: np.array([1.0]) * u.dimensionless_unscaled,
            lambda t: np.array([0.5]),
            make_spectral_grid(),
        )
        prediction = DirectPath().compute(sun, instrument, atmosphere, observation)
        assert prediction.rates.shape == (instrument.n_pixels, 3)
        np.testing.assert_array_equal(prediction.rates.value, 0.0)


class TestScatteredPath:
    def test_defaults_to_explicit_solver(self):
        assert isinstance(ScatteredPath().solver, ExplicitScatteredSolver)

    def test_rejects_direct_solver(self):
        with pytest.raises(TypeError, match="ScatteredSolver"):
            ScatteredPath(solver=ExplicitDirectSolver())

    def test_grid_options_are_stored(self):
        path = ScatteredPath(nside=16, eval_grid_n=3)
        assert path.nside == 16
        assert path.eval_grid_n == 3

    def test_eval_grid_n_changes_projection_input(
        self, instrument, atmosphere, diffuse_source, observation
    ):
        coarse = ScatteredPath(nside=8, eval_grid_n=2).compute(
            diffuse_source, instrument, atmosphere, observation
        )
        fine = ScatteredPath(nside=8, eval_grid_n=3).compute(
            diffuse_source, instrument, atmosphere, observation
        )
        assert coarse.rates.shape == fine.rates.shape

    def test_zero_rates_when_nothing_above_horizon(
        self, instrument, atmosphere, observation
    ):
        sun = EphemerisSource(
            "sun",
            lambda t: np.array([1.0]) * u.dimensionless_unscaled,
            lambda t: np.array([0.5]),
            make_spectral_grid(),
        )
        prediction = ScatteredPath().compute(sun, instrument, atmosphere, observation)
        np.testing.assert_array_equal(prediction.rates.value, 0.0)
        assert prediction.indirect is True


class TestLightPathBase:
    def test_compile_defaults_to_no_cost(self):
        class NullPath(LightPath):
            def compute(self, source, instrument, atmosphere, observation):
                raise NotImplementedError

        assert NullPath().compile(None, None, None) == 0
