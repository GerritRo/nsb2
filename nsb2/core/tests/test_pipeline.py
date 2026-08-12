import pytest

from nsb2.core.dtypes import Prediction
from nsb2.core.lightpath import DirectPath, ScatteredPath
from nsb2.core.pipeline import CompositePipeline, Pipeline, PipelineLike
from nsb2.core.solver import LUTDirectSolver, LUTScatteredSolver


class TestPipeline:
    def test_predicts_one_tagged_result_per_source_and_path(
        self, instrument, atmosphere, diffuse_source, observation
    ):
        pipe = Pipeline(instrument, atmosphere, diffuse_source, DirectPath())
        assert isinstance(pipe, PipelineLike)
        assert pipe.sources == [diffuse_source], "a single source is wrapped in a list"
        assert len(pipe.paths) == 1

        results = pipe.predict(observation)
        assert len(results) == 1
        assert isinstance(results[0], Prediction)
        assert results[0].indirect is False
        assert results[0].rates.shape == (instrument.n_pixels, 3)
        assert (results[0].rates.value > 0).all()
        assert results[0].source_name == diffuse_source.name
        assert results[0].path_name == "DirectPath"

        every_combination = Pipeline(
            instrument,
            atmosphere,
            [diffuse_source, diffuse_source],
            [DirectPath(), ScatteredPath(nside=8)],
        )
        predictions = every_combination.predict(observation)
        assert len(predictions) == 4
        assert [p.indirect for p in predictions] == [False, True, False, True]

    def test_compile_is_a_noop_for_explicit_solvers(
        self, instrument, atmosphere, diffuse_source
    ):
        assert (
            Pipeline(instrument, atmosphere, diffuse_source, [DirectPath()]).compile()
            == 0
        )

    def test_addition_flattens_into_a_composite(
        self, instrument, atmosphere, diffuse_source
    ):
        pipe = Pipeline(instrument, atmosphere, diffuse_source, [DirectPath()])
        assert isinstance(pipe + pipe, CompositePipeline)

        combined = pipe + (pipe + pipe)
        assert isinstance(combined, CompositePipeline)
        assert len(combined._pipelines) == 3
        assert combined._pipelines[0] is pipe

        with pytest.raises(TypeError):
            pipe + 1


class TestCompositePipeline:
    @pytest.fixture
    def composite(self, instrument, atmosphere, diffuse_source):
        pipe = Pipeline(instrument, atmosphere, diffuse_source, [DirectPath()])
        return pipe + Pipeline(instrument, atmosphere, diffuse_source, [DirectPath()])

    def test_runs_every_constituent_pipeline(self, composite, observation):
        assert len(composite.predict(observation)) == 2
        assert composite.compile() == 0

    def test_addition_keeps_the_composite_flat(
        self, composite, instrument, atmosphere, diffuse_source
    ):
        extra = Pipeline(instrument, atmosphere, diffuse_source, [DirectPath()])
        assert len((composite + extra)._pipelines) == 3
        assert len((composite + composite)._pipelines) == 4

        with pytest.raises(TypeError):
            composite + "not a pipeline"


class TestLUTPipeline:
    def test_direct_lut_agrees_with_the_explicit_solver(
        self, instrument, atmosphere, diffuse_source, observation
    ):
        explicit = Pipeline(
            instrument, atmosphere, diffuse_source, [DirectPath()]
        ).predict(observation)[0]

        lut_pipe = Pipeline(
            instrument, atmosphere, diffuse_source, [DirectPath(LUTDirectSolver())]
        )
        assert lut_pipe.compile(extinction_z_bins=90) == 0
        approximate = lut_pipe.predict(observation)[0]

        assert approximate.indirect is False
        assert approximate.rates.value == pytest.approx(explicit.rates.value, rel=0.01)

    def test_scattered_lut_agrees_with_the_explicit_solver(
        self, instrument, atmosphere, diffuse_source, observation
    ):
        explicit = Pipeline(
            instrument, atmosphere, diffuse_source, [ScatteredPath(nside=8)]
        ).predict(observation)[0]

        lut_pipe = Pipeline(
            instrument,
            atmosphere,
            diffuse_source,
            [ScatteredPath(LUTScatteredSolver(), nside=8)],
        )
        lut_pipe.compile(scattering_z_bins=20, scattering_theta_bins=20)
        approximate = lut_pipe.predict(observation)[0]

        assert approximate.indirect is True
        assert approximate.rates.value == pytest.approx(explicit.rates.value, rel=0.01)
