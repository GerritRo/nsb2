import numpy as np
from .constants import h, c, k_B

# Physical Formulas used in blacksky
## Planck Blackbody Radiation
def blackbody(lam, T):
    f1 = 1e-9*2*h*c**2
    f2 = h*c/k_B
    return (f1/lam**5 * 1/(np.exp(f2/(T*lam))-1))