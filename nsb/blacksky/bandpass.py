# A collection of various bandpasses. Most are taken from the SVO Filter
# Profile Service http://svo2.cab.inta-csic.es/theory/fps/ . They take
# the wavelength in nm as an argument.
import numpy as np
from scipy.interpolate import UnivariateSpline

from . import ASSETS_PATH

class SVOFilter():
    def __init__(self, file):
        arr = np.loadtxt(file, delimiter=",")
        self.lam = arr[:, 0]/10
        self.trx = arr[:, 1]

        self.min = np.min(self.lam)
        self.max = np.max(self.lam)

        self.spline = UnivariateSpline(self.lam, self.trx, s=0, ext=1)

    def __call__(self, lam):
        return self.spline(lam)

def GaiaDR3_G():
    return SVOFilter(ASSETS_PATH + 'gaiadr3_G.dat')

def GaiaDR3_RP():
    return SVOFilter(ASSETS_PATH + 'gaiadr3_RP.dat')

def GaiaDR3_BP():
    return SVOFilter(ASSETS_PATH + 'gaiadr3_BP.dat')

def Tycho_V():
    return SVOFilter(ASSETS_PATH + 'tycho_V.dat')

def Tycho_B():
    return SVOFilter(ASSETS_PATH + 'tycho_B.dat')

def Hipp_M():
    return SVOFilter(ASSETS_PATH + 'hipp_M.dat')

def OSN_V():
    return SVOFilter(ASSETS_PATH + 'OSN_V.dat')

def OSN_B():
    return SVOFilter(ASSETS_PATH + 'OSN_B.dat')

def OSN_U():
    return SVOFilter(ASSETS_PATH + 'OSN_U.dat')