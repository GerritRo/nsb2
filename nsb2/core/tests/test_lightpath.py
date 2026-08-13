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


def _sun():
    """A source that is below the horizon."""
    return EphemerisSource(
        "sun",
        lambda t: np.array([1.0]) * u.dimensionless_unscaled,
        lambda t: np.array([0.5]),
        make_spectral_grid(),
    )


class TestDirectPath:
    def test_defaults_to_an_explicit_solver_and_rejects_the_wrong_kind(self):
        assert isinstance(DirectPath().solver, ExplicitDirectSolver)
        assert DirectPath().name == "DirectPath"
        assert DirectPath(name="stars").name == "stars"
        with pytest.raises(TypeError, match="DirectSolver"):
            DirectPath(solver=ExplicitScatteredSolver())

    def test_compile_delegates_to_the_solver(
        self, diffuse_source, instrument, atmosphere
    ):
        solver = LUTDirectSolver()
        DirectPath(solver=solver).compile(diffuse_source, instrument, atmosphere)
        assert diffuse_source in solver._luts
        # An explicit solver has nothing to compile and costs nothing.
        assert DirectPath().compile(diffuse_source, instrument, atmosphere) == 0

    def test_zero_rates_when_no_source_is_in_view(
        self, instrument, atmosphere, observation
    ):
        """A body below the horizon gives zero rates rather than an error."""
        for path, indirect in ((DirectPath(), False), (ScatteredPath(), True)):
            prediction = path.compute(_sun(), instrument, atmosphere, observation)
            assert prediction.rates.shape == (instrument.n_pixels, 3)
            np.testing.assert_array_equal(prediction.rates.value, 0.0)
            assert prediction.indirect is indirect


class TestScatteredPath:
    def test_defaults_to_an_explicit_solver_and_stores_the_grid_options(self):
        assert isinstance(ScatteredPath().solver, ExplicitScatteredSolver)
        path = ScatteredPath(nside=16, eval_grid_n=3)
        assert path.nside == 16
        assert path.eval_grid_n == 3
        with pytest.raises(TypeError, match="ScatteredSolver"):
            ScatteredPath(solver=ExplicitDirectSolver())

    def test_projection_is_independent_of_the_evaluation_grid_size(
        self, instrument, atmosphere, diffuse_source, observation
    ):
        """A uniform sky gives the same per-pixel rate at any grid size."""
        coarse, fine = (
            ScatteredPath(nside=8, eval_grid_n=n).compute(
                diffuse_source, instrument, atmosphere, observation
            )
            for n in (2, 3)
        )
        assert coarse.rates.shape == fine.rates.shape
        assert coarse.rates.value == pytest.approx(fine.rates.value, rel=1e-3)


class TestLightPathBase:
    def test_compile_defaults_to_no_cost(self):
        class NullPath(LightPath):
            def compute(self, source, instrument, atmosphere, observation):
                raise NotImplementedError

        assert NullPath().compile(None, None, None) == 0
