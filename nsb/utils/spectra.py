from .. import ASSETS_PATH

import numpy as np
import scipy.interpolate as si
from astropy.utils.data import download_file
from astropy.io import fits

def SolarSpectrumRieke2008():
    f_down = download_file('https://archive.stsci.edu/hlsps/reference-atlases/cdbs/grid/solsys/solar_spec.fits', cache=True)
    hdul = fits.open(f_down)
    return si.UnivariateSpline(hdul[1].data['WAVELENGTH']/10, hdul[1].data['FLUX']*150, ext=2, s=0)

def AirglowNoll2012():
    ag_array = np.genfromtxt(ASSETS_PATH+'airglow_noll2012.dat')
    return si.UnivariateSpline(ag_array[:,0], ag_array[:,1], s=0)