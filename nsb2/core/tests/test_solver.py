"""Tests for :mod:`nsb2.core.solver`."""

import astropy.units as u
import numpy as np
import pytest

from nsb2.core.solver import (
    ExplicitDirectSolver,
    ExplicitScatteredSolver,
    LUTDirectSolver,
    LUTScatteredSolver,
    Solver,
    _trapz_einsum,
)


class TestTrapzEinsum:
    def test_matches_numpy_trapezoid_for_a_flat_integrand(self):
        wvl = np.linspace(400, 500, 21) * u.nm
        a = np.ones((3, 21))
        b = np.ones((21, 2))
        result = _trapz_einsum(a, b, wvl, "zN,Nc,N->zc")
        assert result.shape == (3, 2)
        np.testing.assert_allclose(result.to_value(u.nm), 100.0)


class TestSolverBase:
    def test_compile_defaults_to_no_cost(self):
        assert Solver().compile(None, None, None) == 0


class TestExplicitDirectSolver:
    def test_rates_are_attenuated_by_extinction(
        self, diffuse_source, atmosphere, bandpass, observation
    ):
        field, _ = diffuse_source.query_direct(
            observation,
            diffuse_source.query_scattered(observation, nside=4).coords[:4],
            np.full(4, 0.1),
        )
        rates = ExplicitDirectSolver().compute_rates(
            diffuse_source, field, atmosphere, bandpass
        )
        assert np.all(rates.value >= 0)


class TestExplicitScatteredSolver:
    def test_rates_have_grid_shape(
        self, diffuse_source, atmosphere, bandpass, instrument, observation
    ):
        field = diffuse_source.query_scattered(observation, nside=4)
        eval_coords = instrument.eval_grid(observation, n=2)
        rates = ExplicitScatteredSolver().compute_rates(
            diffuse_source, field, atmosphere, bandpass, eval_coords
        )
        assert rates.shape[:2] == (2, 2)
        assert rates.shape[2] == len(field.coords)


class TestLUTDirectSolver:
    def test_compute_before_compile_raises(
        self, diffuse_source, atmosphere, bandpass, observation, instrument
    ):
        field, _ = diffuse_source.query_direct(
            observation, instrument.pixel_coords(observation), instrument.pixel_radii()
        )
        with pytest.raises(RuntimeError, match="compile"):
            LUTDirectSolver().compute_rates(diffuse_source, field, atmosphere, bandpass)

    def test_compile_returns_zero_cost(self, diffuse_source, instrument, atmosphere):
        solver = LUTDirectSolver()
        assert solver.compile(diffuse_source, instrument, atmosphere) == 0

    def test_compile_accepts_unknown_options(
        self, diffuse_source, instrument, atmosphere
    ):
        """Unknown options belong to other solvers and must be ignored."""
        solver = LUTDirectSolver()
        solver.compile(diffuse_source, instrument, atmosphere, scattering_theta_bins=3)
        assert diffuse_source in solver._luts


class TestLUTScatteredSolver:
    def test_compute_before_compile_raises(
        self, diffuse_source, atmosphere, bandpass, observation, instrument
    ):
        field = diffuse_source.query_scattered(observation, nside=4)
        eval_coords = instrument.eval_grid(observation, n=2)
        with pytest.raises(RuntimeError, match="compile"):
            LUTScatteredSolver().compute_rates(
                diffuse_source, field, atmosphere, bandpass, eval_coords
            )

    def test_compile_stores_a_table(self, diffuse_source, instrument, atmosphere):
        solver = LUTScatteredSolver()
        solver.compile(
            diffuse_source,
            instrument,
            atmosphere,
            scattering_z_bins=4,
            scattering_theta_bins=4,
        )
        assert diffuse_source in solver._luts
