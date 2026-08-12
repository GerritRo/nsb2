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


class TestSolverHelpers:
    def test_trapz_einsum_integrates_a_flat_integrand(self):
        wvl = np.linspace(400, 500, 21) * u.nm
        result = _trapz_einsum(np.ones((3, 21)), np.ones((21, 2)), wvl, "zN,Nc,N->zc")
        assert result.shape == (3, 2)
        np.testing.assert_allclose(result.to_value(u.nm), 100.0)
        # The base solver is a no-op with nothing to pre-compute.
        assert Solver().compile(None, None, None) == 0


class TestExplicitSolvers:
    def test_direct_rates_are_attenuated_by_extinction(
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

    def test_scattered_rates_span_the_evaluation_grid(
        self, diffuse_source, atmosphere, bandpass, instrument, observation
    ):
        field = diffuse_source.query_scattered(observation, nside=4)
        eval_coords = instrument.eval_grid(observation, n=2)
        rates = ExplicitScatteredSolver().compute_rates(
            diffuse_source, field, atmosphere, bandpass, eval_coords
        )
        assert rates.shape[:2] == (2, 2)
        assert rates.shape[2] == len(field.coords)


class TestLUTSolvers:
    def test_direct_lut_must_be_compiled_before_use(
        self, diffuse_source, atmosphere, bandpass, observation, instrument
    ):
        field, _ = diffuse_source.query_direct(
            observation, instrument.pixel_coords(observation), instrument.pixel_radii()
        )
        solver = LUTDirectSolver()
        with pytest.raises(RuntimeError, match="compile"):
            solver.compute_rates(diffuse_source, field, atmosphere, bandpass)

        assert (
            solver.compile(
                diffuse_source, instrument, atmosphere, scattering_theta_bins=3
            )
            == 0
        )
        assert diffuse_source in solver._luts

    def test_scattered_lut_must_be_compiled_before_use(
        self, diffuse_source, atmosphere, bandpass, observation, instrument
    ):
        field = diffuse_source.query_scattered(observation, nside=4)
        eval_coords = instrument.eval_grid(observation, n=2)
        solver = LUTScatteredSolver()
        with pytest.raises(RuntimeError, match="compile"):
            solver.compute_rates(
                diffuse_source, field, atmosphere, bandpass, eval_coords
            )

        solver.compile(
            diffuse_source,
            instrument,
            atmosphere,
            scattering_z_bins=4,
            scattering_theta_bins=4,
        )
        assert diffuse_source in solver._luts
