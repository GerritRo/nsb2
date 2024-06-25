from .. import ASSETS_PATH

import numpy as np
import histlite as hl

def hl_spectra(file):
    spec = np.genfromtxt(file, delimiter=" ")
    h = hl.Hist(spec[:,0], spec[:,1][:-1])
    return h

def eso_airglow():
    return hl_spectra(ASSETS_PATH+'airglow_noll2012.dat')