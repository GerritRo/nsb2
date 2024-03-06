import numpy as np
import histlite as hl

def hl_spectra(file):
    spec = np.genfromtxt(file, delimiter=" ")
    h = hl.Hist(spec[:,0], spec[:,1][:-1])
    return h

def eso_airglow():
    return hl_spectra('/home/gerritr/ECAP/nsb_simulation/nsb2/nsb/utils/assets/airglow_130sfu.csv')