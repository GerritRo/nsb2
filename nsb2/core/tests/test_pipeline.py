"""Tests for :mod:`nsb2.core.pipeline`."""

import pytest

from nsb2.core.dtypes import Prediction
from nsb2.core.lightpath import DirectPath, ScatteredPath
from nsb2.core.pipeline import CompositePipeline, Pipeline, PipelineLike
from nsb2.core.solver import LUTDirectSolver, LUTScatteredSolver


class TestPipeline:
    def test_init_wraps_single_source_in_list(
        self, instrument, atmosphere, diffuse_source
    ):
        pipe = Pipeline(instrument, atmosphere, diffuse_source, DirectPath())
        assert pipe.sources == [diffuse_source]
        assert len(pipe.paths) == 1

    def test_satisfies_pipeline_protocol(self, instrument, atmosphere, diffuse_source):
        pipe = Pipeline(instrument, atmosphere, diffuse_source, [DirectPath()])
        assert isinstance(pipe, PipelineLike)

    def test_compile_is_noop_for_explicit_solver(
        self, instrument, atmosphere, diffuse_source
    ):
        pipe = Pipeline(instrument, atmosphere, diffuse_source, [DirectPath()])
        assert pipe.compile() == 0

    def test_predict_direct_path(
        self, instrument, atmosphere, diffuse_source, observation
    ):
        pipe = Pipeline(instrument, atmosphere, diffuse_source, [DirectPath()])
        results = pipe.predict(observation)
        assert len(results) == 1
        assert isinstance(results[0], Prediction)
        assert results[0].indirect is False
        assert results[0].rates.shape == (instrument.n_pixels, 3)

    def test_predict_tags_source_and_path(
        self, instrument, atmosphere, diffuse_source, observation
    ):
        pipe = Pipeline(instrument, atmosphere, diffuse_source, [DirectPath()])
        result = pipe.predict(observation)[0]
        assert result.source_name == diffuse_source.name
        assert result.path_name == "DirectPath"

    def test_predict_runs_every_source_path_combination(
        self, instrument, atmosphere, diffuse_source, observation
    ):
        pipe = Pipeline(
            instrument,
            atmosphere,
            [diffuse_source, diffuse_source],
            [DirectPath(), ScatteredPath(nside=8)],
        )
        assert len(pipe.predict(observation)) == 4

    def test_predict_scattered_path_is_marked_indirect(
        self, instrument, atmosphere, diffuse_source, observation
    ):
        pipe = Pipeline(
            instrument, atmosphere, diffuse_source, [ScatteredPath(nside=8)]
        )
        assert pipe.predict(observation)[0].indirect is True

    def test_rates_are_positive(
        self, instrument, atmosphere, diffuse_source, observation
    ):
        pipe = Pipeline(instrument, atmosphere, diffuse_source, [DirectPath()])
        assert (pipe.predict(observation)[0].rates.value > 0).all()

    def test_add_creates_composite(self, instrument, atmosphere, diffuse_source):
        p1 = Pipeline(instrument, atmosphere, diffuse_source, [DirectPath()])
        p2 = Pipeline(instrument, atmosphere, diffuse_source, [DirectPath()])
        assert isinstance(p1 + p2, CompositePipeline)

    def test_add_composite_prepends(self, instrument, atmosphere, diffuse_source):
        """Pipeline + CompositePipeline flattens into a single composite."""
        pipe = Pipeline(instrument, atmosphere, diffuse_source, [DirectPath()])
        composite = pipe + pipe
        combined = pipe + composite
        assert isinstance(combined, CompositePipeline)
        assert len(combined._pipelines) == 3
        assert combined._pipelines[0] is pipe

    def test_add_unsupported_type_returns_not_implemented(
        self, instrument, atmosphere, diffuse_source
    ):
        pipe = Pipeline(instrument, atmosphere, diffuse_source, [DirectPath()])
        with pytest.raises(TypeError):
            pipe + 1


class TestCompositePipeline:
    @pytest.fixture
    def composite(self, instrument, atmosphere, diffuse_source):
        p1 = Pipeline(instrument, atmosphere, diffuse_source, [DirectPath()])
        p2 = Pipeline(instrument, atmosphere, diffuse_source, [DirectPath()])
        return p1 + p2

    def test_predict_combines_results(self, composite, observation):
        assert len(composite.predict(observation)) == 2

    def test_compile_sums_costs(self, composite):
        assert composite.compile() == 0

    def test_add_pipeline_extends(
        self, composite, instrument, atmosphere, diffuse_source
    ):
        p3 = Pipeline(instrument, atmosphere, diffuse_source, [DirectPath()])
        assert len((composite + p3)._pipelines) == 3

    def test_add_composite_concatenates(self, composite):
        assert len((composite + composite)._pipelines) == 4

    def test_add_unsupported_type_returns_not_implemented(self, composite):
        with pytest.raises(TypeError):
            composite + "not a pipeline"


class TestLUTPipeline:
    def test_compile_and_predict_via_lut(
        self, instrument, atmosphere, diffuse_source, observation
    ):
        pipe = Pipeline(
            instrument,
            atmosphere,
            diffuse_source,
            [DirectPath(solver=LUTDirectSolver())],
        )
        pipe.compile(extinction_z_bins=10)
        results = pipe.predict(observation)
        assert len(results) == 1
        assert results[0].indirect is False

    def test_lut_agrees_with_explicit_solver(
        self, instrument, atmosphere, diffuse_source, observation
    ):
        """The lookup table is an approximation, but must track the exact result."""
        explicit = Pipeline(
            instrument, atmosphere, diffuse_source, [DirectPath()]
        ).predict(observation)[0]

        lut_pipe = Pipeline(
            instrument, atmosphere, diffuse_source, [DirectPath(LUTDirectSolver())]
        )
        lut_pipe.compile(extinction_z_bins=90)
        approximate = lut_pipe.predict(observation)[0]

        assert approximate.rates.value == pytest.approx(explicit.rates.value, rel=0.01)

    def test_scattered_lut_compiles_and_predicts(
        self, instrument, atmosphere, diffuse_source, observation
    ):
        pipe = Pipeline(
            instrument,
            atmosphere,
            diffuse_source,
            [ScatteredPath(LUTScatteredSolver(), nside=8)],
        )
        pipe.compile(scattering_z_bins=5, scattering_theta_bins=5)
        assert pipe.predict(observation)[0].indirect is True
