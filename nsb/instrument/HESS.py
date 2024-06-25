from . import BANDPASS_PATH, RESPONSE_PATH

from nsb.core.instrument import Instrument, Camera, Bandpass
# HESS Optic Descriptions for each Telescope
class CT1(Instrument):
    def __init__(self, N):
        ct1_camera = Camera.from_response(RESPONSE_PATH+'hess1u_ct1.pkl')
        ct1_bandpass = Bandpass.from_csv(BANDPASS_PATH+'hess1u_ct1.dat')
        super().__init__({'camera':ct1_camera, 'bandpass':ct1_bandpass}, N)