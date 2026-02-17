import numpy as np

from nsb2.core.instrument import EffectiveApertureInstrument
from nsb2.core.spectral import Bandpass

from . import BANDPASS_PATH, RESPONSE_PATH


def LST1North():
    response = np.load(RESPONSE_PATH / "LST_North_1_like.npz")
    bandpass = Bandpass.from_csv(BANDPASS_PATH / "LST_like.dat")
    return EffectiveApertureInstrument(response, bandpass)

def MSTNorth():
    response = np.load(RESPONSE_PATH / "MST_North_like.npz")
    bandpass = Bandpass.from_csv(BANDPASS_PATH / "MST_like.dat")
    return EffectiveApertureInstrument(response, bandpass)
