"""Simulating night sky background for imaging air Cherenkov telescopes.

``nsb2`` predicts the rate of night sky background photons in each pixel of
an IACT camera.  A simulation is assembled from four independent pieces --
one or more :mod:`sources <nsb2.core.sources>`, an
:mod:`atmosphere <nsb2.core.atmosphere>`, an
:mod:`instrument <nsb2.core.instrument>` and the
:mod:`light paths <nsb2.core.lightpath>` to trace -- which a
:class:`~nsb2.core.pipeline.Pipeline` runs for a given observation.
"""

from pathlib import Path

__version__ = "0.1.0"

#: Directory holding the reference data files bundled with the package.
ASSETS_PATH = Path(__file__).parent / "data"

__all__ = ["ASSETS_PATH", "__version__"]
