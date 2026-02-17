import numpy as np

from nsb2.core.instrument import EffectiveApertureInstrument
from nsb2.core.spectral import Bandpass

from . import BANDPASS_PATH, RESPONSE_PATH


def CT1():
    response = np.load(RESPONSE_PATH / "HESSI_best_guess.npz")
    bandpass = Bandpass.from_csv(BANDPASS_PATH / "hess1u_ct1.dat")
    return EffectiveApertureInstrument(response, bandpass)
