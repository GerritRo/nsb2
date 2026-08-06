"""Ready-made instrument models for H.E.S.S. telescopes."""

import numpy as np

from nsb2.core.instrument import EffectiveApertureInstrument
from nsb2.core.spectral import Bandpass

from . import BANDPASS_PATH, RESPONSE_PATH

__all__ = [
    "CT1",
]


def CT1() -> EffectiveApertureInstrument:
    """Build a model of H.E.S.S. telescope CT1 after the camera upgrade.

    Returns
    -------
    nsb2.core.instrument.EffectiveApertureInstrument
        The telescope, with its bundled effective aperture map and passband.

    Notes
    -----
    The effective aperture map is a best guess rather than the output of an
    official ray-tracing campaign.
    """
    response = np.load(RESPONSE_PATH / "HESSI_best_guess.npz")
    bandpass = Bandpass.from_csv(BANDPASS_PATH / "hess1u_ct1.dat")
    return EffectiveApertureInstrument(response, bandpass)
