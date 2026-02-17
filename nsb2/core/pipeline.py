from __future__ import annotations

from typing import Protocol, runtime_checkable

from nsb2.core.atmosphere import Atmosphere
from nsb2.core.dtypes import Prediction
from nsb2.core.instrument import Instrument
from nsb2.core.lightpath import LightPath
from nsb2.core.sources import Source


@runtime_checkable
class PipelineLike(Protocol):
    """Protocol for pipeline objects (Pipeline and CompositePipeline)."""
    def compile(self, **kwargs) -> float: ...
    def predict(self, observation) -> list[Prediction]: ...


class Pipeline:
    def __init__(
        self,
        instrument: Instrument,
        atmosphere: Atmosphere,
        sources: Source | list[Source],
        paths: LightPath | list[LightPath],
    ) -> None:
        self.instrument = instrument
        self.atmosphere = atmosphere
        self.sources = sources if isinstance(sources, list) else [sources]
        self.paths = paths if isinstance(paths, list) else [paths]

    def compile(self, **kwargs) -> float:
        cost = 0.0
        for path in self.paths:
            for source in self.sources:
                cost += path.compile(source, self.instrument,
                                     self.atmosphere, **kwargs)
        return cost

    def predict(self, observation) -> list[Prediction]:
        """Run the simulation for the given observation.

        Returns
        -------
        predictions : list of Prediction
        """
        results = []
        for source in self.sources:
            for path in self.paths:
                pred = path.compute(
                    source, self.instrument, self.atmosphere, observation)
                pred.source_name = source.name
                pred.path_name = path.name
                results.append(pred)
        return results

    def __add__(self, other) -> CompositePipeline:
        if isinstance(other, Pipeline):
            return CompositePipeline([self, other])
        if isinstance(other, CompositePipeline):
            return CompositePipeline([self] + other._pipelines)
        return NotImplemented


class CompositePipeline:
    def __init__(self, pipelines: list[Pipeline]) -> None:
        self._pipelines = list(pipelines)

    def compile(self, **kwargs) -> float:
        return sum(p.compile(**kwargs) for p in self._pipelines)

    def predict(self, observation) -> list[Prediction]:
        results = []
        for p in self._pipelines:
            results.extend(p.predict(observation))
        return results

    def __add__(self, other) -> CompositePipeline:
        if isinstance(other, CompositePipeline):
            return CompositePipeline(self._pipelines + other._pipelines)
        if isinstance(other, Pipeline):
            return CompositePipeline(self._pipelines + [other])
        return NotImplemented
