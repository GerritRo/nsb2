from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import astropy.units as u
import numpy as np

from nsb2.core.dtypes import Prediction
from nsb2.core.solver import (
    DirectSolver,
    ExplicitDirectSolver,
    ExplicitScatteredSolver,
    ScatteredSolver,
)
from nsb2.core.sources import DEFAULT_NSIDE

__all__ = [
    "DirectPath",
    "LightPath",
    "ScatteredPath",
]


if TYPE_CHECKING:
    from nsb2.core.atmosphere import Atmosphere
    from nsb2.core.instrument import Instrument
    from nsb2.core.sources import Source

logger = logging.getLogger(__name__)

DEFAULT_EVAL_GRID_N = 2


class LightPath(ABC):
    """Base class for physical light paths from a source to a pixel.

    Attributes
    ----------
    name : str
        Human-readable identifier, carried through to
        :class:`nsb2.core.dtypes.Prediction`.
    """

    name: str = ""

    @abstractmethod
    def compute(
        self,
        source: Source,
        instrument: Instrument,
        atmosphere: Atmosphere,
        observation,
    ) -> Prediction:
        """Compute the pixel rates from one source along this path.

        Parameters
        ----------
        source : nsb2.core.sources.Source
            Source to trace.
        instrument : nsb2.core.instrument.Instrument
            Instrument collecting the light.
        atmosphere : nsb2.core.atmosphere.Atmosphere
            Atmosphere the light travels through.
        observation : astropy.coordinates.BaseCoordinateFrame
            The telescope pointing frame.

        Returns
        -------
        nsb2.core.dtypes.Prediction
            Per-pixel rates for this source and path.
        """
        ...

    def compile(
        self,
        source: Source,
        instrument: Instrument,
        atmosphere: Atmosphere,
        **kwargs,
    ) -> float:
        """Pre-compute whatever the path's solver needs.

        Parameters
        ----------
        source : nsb2.core.sources.Source
            Source the pre-computation applies to.
        instrument : nsb2.core.instrument.Instrument
            Instrument whose passband the spectra are restricted to.
        atmosphere : nsb2.core.atmosphere.Atmosphere
            Atmosphere model to tabulate.
        **kwargs
            Solver-specific compilation options.

        Returns
        -------
        float
            Compilation cost metric; zero when nothing was compiled.
        """
        return 0


class DirectPath(LightPath):
    """Direct light: source, atmospheric extinction, then the pixel it lands in.

    Parameters
    ----------
    solver : nsb2.core.solver.DirectSolver, optional
        Rate computation strategy.  Default is
        :class:`~nsb2.core.solver.ExplicitDirectSolver`.
    name : str, optional
        Identifier for the path.  Defaults to the class name.

    Raises
    ------
    TypeError
        If ``solver`` is not a `~nsb2.core.solver.DirectSolver`.
    """

    def __init__(self, solver: DirectSolver | None = None, name: str = "") -> None:
        self.name = name or type(self).__name__
        self.solver = solver or ExplicitDirectSolver()
        if not isinstance(self.solver, DirectSolver):
            raise TypeError(
                f"DirectPath requires a DirectSolver, got {type(self.solver).__name__}"
            )

    def compile(
        self,
        source: Source,
        instrument: Instrument,
        atmosphere: Atmosphere,
        **kwargs,
    ) -> float:
        """Pre-compute the solver's lookup tables.

        See :meth:`LightPath.compile` for the parameters and returns.
        """
        return self.solver.compile(source, instrument, atmosphere, **kwargs)

    def compute(
        self,
        source: Source,
        instrument: Instrument,
        atmosphere: Atmosphere,
        observation,
    ) -> Prediction:
        """Compute the directly transmitted pixel rates.

        See :meth:`LightPath.compute` for the parameters and returns.  If no
        source falls into the field of view, the returned rates are zero.
        """
        pix_coords = instrument.pixel_coords(observation)
        field, pixel_refs = source.query_direct(
            observation, pix_coords, instrument.pixel_radii()
        )

        if field.spectral_data.shape[0] == 0:
            logger.debug("no %s sources in the field of view", source.name)
            return Prediction(
                rates=np.zeros((len(pix_coords), 3)) * u.Hz, indirect=False
            )

        pixel_refs = instrument.compute_pixel_weights(field, pixel_refs, observation)
        rates = self.solver.compute_rates(
            source, field, atmosphere, instrument.bandpass
        )

        return Prediction(
            rates=instrument.project_discrete(rates, pixel_refs), indirect=False
        )


class ScatteredPath(LightPath):
    """Scattered light: the whole visible sky, scattered into the field of view.

    The hemisphere is sampled on a HEALPix grid [Gorski2005]_ through healpy
    [Zonca2019]_, so every scattered prediction depends on them; see
    :ref:`acknowledgements`.

    Parameters
    ----------
    solver : nsb2.core.solver.ScatteredSolver, optional
        Rate computation strategy.  Default is
        :class:`~nsb2.core.solver.ExplicitScatteredSolver`.
    name : str, optional
        Identifier for the path.  Defaults to the class name.
    nside : int, optional
        HEALPix resolution for the hemisphere query.  Default is
        :data:`~nsb2.core.sources.DEFAULT_NSIDE`.
    eval_grid_n : int, optional
        Number of evaluation grid points per axis, giving an
        ``eval_grid_n`` by ``eval_grid_n`` grid across the field of view.
        Default is :data:`DEFAULT_EVAL_GRID_N`.

    Raises
    ------
    TypeError
        If ``solver`` is not a `~nsb2.core.solver.ScatteredSolver`.
    """

    def __init__(
        self,
        solver: ScatteredSolver | None = None,
        name: str = "",
        *,
        nside: int = DEFAULT_NSIDE,
        eval_grid_n: int = DEFAULT_EVAL_GRID_N,
    ) -> None:
        self.name = name or type(self).__name__
        self.solver = solver or ExplicitScatteredSolver()
        self.nside = nside
        self.eval_grid_n = eval_grid_n
        if not isinstance(self.solver, ScatteredSolver):
            raise TypeError(
                f"ScatteredPath requires a ScatteredSolver, "
                f"got {type(self.solver).__name__}"
            )

    def compile(
        self,
        source: Source,
        instrument: Instrument,
        atmosphere: Atmosphere,
        **kwargs,
    ) -> float:
        """Pre-compute the solver's lookup tables.

        See :meth:`LightPath.compile` for the parameters and returns.
        """
        return self.solver.compile(source, instrument, atmosphere, **kwargs)

    def compute(
        self,
        source: Source,
        instrument: Instrument,
        atmosphere: Atmosphere,
        observation,
    ) -> Prediction:
        """Compute the in-scattered pixel rates.

        See :meth:`LightPath.compute` for the parameters and returns.  If no
        source is above the horizon, the returned rates are zero.
        """
        field = source.query_scattered(observation, nside=self.nside)

        if field.spectral_data.shape[0] == 0:
            logger.debug("no %s sources above the horizon", source.name)
            n_pix = len(instrument.pixel_coords(observation))
            return Prediction(rates=np.zeros((n_pix, 3)) * u.Hz, indirect=True)

        eval_coords = instrument.eval_grid(observation, n=self.eval_grid_n)
        rates = self.solver.compute_rates(
            source, field, atmosphere, instrument.bandpass, eval_coords
        )

        return Prediction(
            rates=instrument.project_continuous(rates, eval_coords), indirect=True
        )
