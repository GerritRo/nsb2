import numpy as np
import pytest

from nsb2.instrument import BANDPASS_PATH, RESPONSE_PATH
from nsb2.instrument.CTAO import SST, LST1North, MSTNorth
from nsb2.instrument.HESS import CT1

FACTORIES = [LST1North, MSTNorth, SST, CT1]


class TestDataFiles:
    def test_every_response_has_a_bandpass(self):
        assert BANDPASS_PATH.is_dir() and RESPONSE_PATH.is_dir()
        assert list(RESPONSE_PATH.glob("*.npz"))
        assert list(BANDPASS_PATH.glob("*.dat"))


@pytest.mark.parametrize("factory", FACTORIES, ids=lambda f: f.__name__)
def test_loads_a_plausible_camera(factory, observation):
    """Each bundled model must describe a real IACT camera."""
    instrument = factory()

    assert instrument.n_pixels > 0
    assert len(instrument.pixel_coords(observation)) == instrument.n_pixels
    assert np.all(instrument.pixel_radii() > 0)
    assert np.all(instrument._pix_area_sr > 0)

    lon_range, lat_range = instrument.fov_range()
    assert lon_range[0] < lon_range[1]
    assert lat_range[0] < lat_range[1]
