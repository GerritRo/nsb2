"""Ready-made instrument models for CTAO telescopes.

The bundled responses are representative of the telescope types rather than
measurements of any individual built telescope, and are named ``*_like``
accordingly.
"""

import numpy as np

from nsb2.core.instrument import EffectiveApertureInstrument
from nsb2.core.spectral import Bandpass

from . import BANDPASS_PATH, RESPONSE_PATH

__all__ = [
    "LST1North",
    "MSTNorth",
]


def LST1North() -> EffectiveApertureInstrument:
    """Build a model of the first Large-Sized Telescope on La Palma.

    Returns
    -------
    nsb2.core.instrument.EffectiveApertureInstrument
        The telescope, with its bundled effective aperture map and passband.
    """
    response = np.load(RESPONSE_PATH / "LST_North_1_like.npz")
    bandpass = Bandpass.from_csv(BANDPASS_PATH / "LST_like.dat")
    return EffectiveApertureInstrument(response, bandpass)


def MSTNorth() -> EffectiveApertureInstrument:
    """Build a model of a northern Medium-Sized Telescope.

    Returns
    -------
    nsb2.core.instrument.EffectiveApertureInstrument
        The telescope, with its bundled effective aperture map and passband.
    """
    response = np.load(RESPONSE_PATH / "MST_North_like.npz")
    bandpass = Bandpass.from_csv(BANDPASS_PATH / "MST_like.dat")
    return EffectiveApertureInstrument(response, bandpass)

def SST():
    response = np.load(RESPONSE_PATH / "SST_like.npz")
    bandpass = Bandpass.from_csv(BANDPASS_PATH / "SST_like.dat")
    return EffectiveApertureInstrument(response, bandpass)

