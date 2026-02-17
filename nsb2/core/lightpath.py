from __future__ import annotations

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

if TYPE_CHECKING:
    from nsb2.core.atmosphere import Atmosphere
    from nsb2.core.instrument import Instrument
    from nsb2.core.sources import Source


class LightPath(ABC):
    """A physical light path from source to instrument pixel.
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
        """Compute pixel rates for one source via this light path.

        Parameters
        ----------
        source : Source
        instrument : Instrument
        atmosphere : Atmosphere
        observation : coordinate frame

        Returns
        -------
        Prediction
        """
        ...

    def compile(
        self,
        source: Source,
        instrument: Instrument,
        atmosphere: Atmosphere,
        **kwargs,
    ) -> float:
        """Optional pre-computation step (e.g., LUT generation).

        Returns 0 by default (no cost).
        """
        return 0


class DirectPath(LightPath):
    """Direct light: source -> atmospheric extinction -> discrete pixel projection.

    Parameters
    ----------
    solver : DirectSolver, optional
        Rate computation strategy. Defaults to ExplicitDirectSolver.
    """

    def __init__(self, solver: DirectSolver | None = None, name: str = "") -> None:
        self.name = name or type(self).__name__
        self.solver = solver or ExplicitDirectSolver()
        if not isinstance(self.solver, DirectSolver):
            raise TypeError(
                f"DirectPath requires a DirectSolver, got {type(self.solver).__name__}")

    def compile(self, source: Source, instrument: Instrument, atmosphere: Atmosphere, **kwargs) -> float:
        return self.solver.compile(source, instrument, atmosphere, **kwargs)

    def compute(self, source: Source, instrument: Instrument, atmosphere: Atmosphere, observation) -> Prediction:
        pix_coords = instrument.pixel_coords(observation)
        field, pixel_refs = source.query_direct(
            observation, pix_coords, instrument.pixel_radii())

        if field.spectral_data.shape[0] == 0:
            return Prediction(rates=np.zeros((len(pix_coords), 3))*u.Hz, indirect=False)

        pixel_refs = instrument.compute_pixel_weights(field, pixel_refs, observation)

        rates = self.solver.compute_rates(
            source, field, atmosphere, instrument.bandpass)

        return Prediction(
            rates=instrument.project_discrete(rates, pixel_refs),
            indirect=False)


class ScatteredPath(LightPath):
    """Scattered light: hemisphere -> scattering kernel -> eval grid.

    Parameters
    ----------
    solver : ScatteredSolver, optional
        Rate computation strategy. Defaults to ExplicitScatteredSolver.
    nside : int
        HEALPix resolution (NSIDE) for the hemisphere source query.
    eval_grid_n : int
        Number of grid points per axis for the FoV evaluation grid
        (produces an eval_grid_n x eval_grid_n grid).
    """

    def __init__(
        self,
        solver: ScatteredSolver | None = None,
        name: str = "",
        *,
        nside: int = 64,
        eval_grid_n: int = 2,
    ) -> None:
        self.name = name or type(self).__name__
        self.solver = solver or ExplicitScatteredSolver()
        self.nside = nside
        self.eval_grid_n = eval_grid_n
        if not isinstance(self.solver, ScatteredSolver):
            raise TypeError(
                f"ScatteredPath requires a ScatteredSolver, "
                f"got {type(self.solver).__name__}")

    def compile(self, source: Source, instrument: Instrument, atmosphere: Atmosphere, **kwargs) -> float:
        return self.solver.compile(source, instrument, atmosphere, **kwargs)

    def compute(self, source: Source, instrument: Instrument, atmosphere: Atmosphere, observation) -> Prediction:
        field = source.query_scattered(observation, nside=self.nside)

        if field.spectral_data.shape[0] == 0:
            n_pix = len(instrument.pixel_coords(observation))
            return Prediction(rates=np.zeros((n_pix, 3))*u.Hz, indirect=True)

        eval_coords = instrument.eval_grid(observation, n=self.eval_grid_n)

        rates = self.solver.compute_rates(
            source, field, atmosphere, instrument.bandpass, eval_coords)

        return Prediction(
            rates=instrument.project_continuous(rates, eval_coords),
            indirect=True)
