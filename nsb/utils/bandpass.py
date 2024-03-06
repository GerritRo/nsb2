import numpy as np
from scipy.interpolate import UnivariateSpline

def p_spline(file):
    arr = np.loadtxt(file, delimiter=",")
    x = arr[:, 0]
    y = arr[:, 1]
    
    s = UnivariateSpline(x, y, s=0, ext=1)
    
    return s

def HESS1U():
    return p_spline('/home/gerritr/ECAP/nsb_simulation/nsb2/nsb/utils/assets/hess1u.csv')

def GaiaDR3():
    return p_spline('/home/gerritr/ECAP/nsb_simulation/nsb2/nsb/utils/assets/g_passband.csv')

def JohnsonV():
    return p_spline('/home/gerritr/ECAP/nsb_simulation/nsb2/nsb/utils/assets/johnson_v.csv')