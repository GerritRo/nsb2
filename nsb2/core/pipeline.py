"""Composition of sources, light paths, atmosphere and instrument.

A pipeline is the top-level object of a simulation: it holds one instrument
and atmosphere together with the sources and light paths to trace, and runs
every combination of the two for a given observation.  Pipelines can be added
together, which is how contributions computed with different settings -- a
lookup table for a star catalogue, explicit integration for the Moon -- are
combined into one prediction.
"""

from __future__ import annotations

import logging
from typing import Protocol, runtime_checkable

from nsb2.core.atmosphere import Atmosphere
from nsb2.core.dtypes import Prediction
from nsb2.core.instrument import Instrument
from nsb2.core.lightpath import LightPath
from nsb2.core.sources import Source

__all__ = [
    "CompositePipeline",
    "Pipeline",
    "PipelineLike",
]


logger = logging.getLogger(__name__)


@runtime_checkable
class PipelineLike(Protocol):
    """Structural type shared by :class:`Pipeline` and :class:`CompositePipeline`."""

    def compile(self, **kwargs) -> float:
        """Pre-compute whatever the pipeline's solvers need."""
        ...

    def predict(self, observation) -> list[Prediction]:
        """Run the simulation for one observation."""
        ...


class Pipeline:
    """One instrument and atmosphere, traced over a set of sources and paths.

    Parameters
    ----------
    instrument : nsb2.core.instrument.Instrument
        Instrument collecting the light.
    atmosphere : nsb2.core.atmosphere.Atmosphere
        Atmosphere the light travels through.
    sources : nsb2.core.sources.Source or list of nsb2.core.sources.Source
        Sources to trace.  A single source is wrapped in a list.
    paths : nsb2.core.lightpath.LightPath or list of nsb2.core.lightpath.LightPath
        Light paths to trace each source along.  A single path is wrapped in
        a list.

    Examples
    --------
    >>> pipeline = Pipeline(instrument, atmosphere, moon,
    ...                     [DirectPath(), ScatteredPath()])  # doctest: +SKIP
    >>> predictions = pipeline.predict(observation)  # doctest: +SKIP
    """

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
        """Pre-compute the lookup tables of every source-path combination.

        Must be called before :meth:`predict` when any path uses a lookup
        table solver.  Paths that integrate explicitly ignore the call.

        Parameters
        ----------
        **kwargs
            Solver-specific compilation options, forwarded to every path.

        Returns
        -------
        float
            Summed compilation cost metric.
        """
        cost = 0.0
        for path in self.paths:
            for source in self.sources:
                cost += path.compile(source, self.instrument, self.atmosphere, **kwargs)
        return cost

    def predict(self, observation) -> list[Prediction]:
        """Run the simulation for one observation.

        Parameters
        ----------
        observation : astropy.coordinates.BaseCoordinateFrame
            The telescope pointing frame, carrying the observation time and
            location on its ``origin``.

        Returns
        -------
        list of nsb2.core.dtypes.Prediction
            One prediction per source and light path, each tagged with the
            names of the source and path that produced it.
        """
        results = []
        for source in self.sources:
            for path in self.paths:
                logger.debug("predicting %s along %s", source.name, path.name)
                pred = path.compute(
                    source, self.instrument, self.atmosphere, observation
                )
                pred.source_name = source.name
                pred.path_name = path.name
                results.append(pred)
        return results

    def __add__(self, other) -> CompositePipeline:
        """Combine this pipeline with another into a :class:`CompositePipeline`."""
        if isinstance(other, Pipeline):
            return CompositePipeline([self, other])
        if isinstance(other, CompositePipeline):
            return CompositePipeline([self, *other._pipelines])
        return NotImplemented


class CompositePipeline:
    """Several pipelines run together as one.

    Parameters
    ----------
    pipelines : list of Pipeline
        The pipelines to run.

    See Also
    --------
    Pipeline.__add__ : The usual way to build one.
    """

    def __init__(self, pipelines: list[Pipeline]) -> None:
        self._pipelines = list(pipelines)

    def compile(self, **kwargs) -> float:
        """Compile every constituent pipeline.

        Parameters
        ----------
        **kwargs
            Solver-specific compilation options, forwarded to every pipeline.

        Returns
        -------
        float
            Summed compilation cost metric.
        """
        return sum(p.compile(**kwargs) for p in self._pipelines)

    def predict(self, observation) -> list[Prediction]:
        """Run every constituent pipeline for one observation.

        Parameters
        ----------
        observation : astropy.coordinates.BaseCoordinateFrame
            The telescope pointing frame.

        Returns
        -------
        list of nsb2.core.dtypes.Prediction
            The concatenated predictions of every constituent pipeline.
        """
        results = []
        for p in self._pipelines:
            results.extend(p.predict(observation))
        return results

    def __add__(self, other) -> CompositePipeline:
        """Combine this pipeline with another into a new :class:`CompositePipeline`."""
        if isinstance(other, CompositePipeline):
            return CompositePipeline(self._pipelines + other._pipelines)
        if isinstance(other, Pipeline):
            return CompositePipeline([*self._pipelines, other])
        return NotImplemented
