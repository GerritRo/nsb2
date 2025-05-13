from multiprocessing.pool import ThreadPool
import time

import numpy as np
from scipy.interpolate import UnivariateSpline
from astropy.io import fits

from . import ASSETS_PATH

def Vega_STIS003():
    file = ASSETS_PATH + 'Vega_STIS003.dat'
    arr = np.loadtxt(file, delimiter=",")
    lam = arr[:, 0]*1e3
    trx = arr[:, 1]*1e-7*(100**2)*10

    return UnivariateSpline(lam, trx, s=0, ext=1)

def Vega_STIS008():
    file = ASSETS_PATH + 'Vega_STIS008.dat'
    arr = np.loadtxt(file, delimiter=",")
    lam = arr[:, 0]/10
    trx = arr[:, 1]*1e-7*(100**2)*10

    return UnivariateSpline(lam, trx, s=0, ext=1)

def download_coelho2014():
    def open_file(f, wvl=False):
        hdul = fits.open(f, cache=False)
        wvls = 10**(hdul[0].header['CRVAL1'] + np.arange(2250)*hdul[0].header['CDELT1'])
        data = hdul[0].data
        hdul.close()
        if wvl == False:
            return data
        else:
            return wvls

    file_list = [line.strip() for line in open(ASSETS_PATH+'coelho2014_urls.dat', 'r')]
    p = ThreadPool()
    rs = p.map_async(open_file, file_list)
    p.close() # No more work
    while True:
        if rs.ready(): break
        remaining = rs._number_left
        print("Waiting for", remaining, "tasks to complete...", end='\r')
        time.sleep(1)

    wvl = open_file(file_list[0], wvl=True)
    specs = np.asarray(rs.get())
    
    np.save(ASSETS_PATH+'coelho2014.npy', np.vstack([wvl, specs]))