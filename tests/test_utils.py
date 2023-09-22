import numpy as np
import pytest

# import micropolarray as ml
from micropolarray.processing.demodulation import Malus
from micropolarray.processing.demosaic import merge_polarizations


@pytest.fixture(autouse=True)
def dummy_data():
    """Dummy data factory"""

    def _make_dummy_data(dimension):
        dummydata = np.zeros(shape=(dimension, dimension))
        dummydata[0::2, 0::2] = 1
        dummydata[0::2, 1::2] = 2
        dummydata[1::2, 0::2] = 3
        dummydata[1::2, 1::2] = 4
        return dummydata

    return _make_dummy_data


def generate_polarized_data(
    shape, S, angle_rad=0, t=1, eff=1, angles_list=[0, 45, -45, 90]
):
    single_pol_shape = (int(shape[0] / 2), int(shape[0] / 2))
    ones = np.ones(shape=single_pol_shape)
    angles = np.array([np.deg2rad(angle) for angle in angles_list])
    subimages = np.array(
        [ones * S * Malus(angle_rad, t, eff, angle) for angle in angles]
    )
    return merge_polarizations(subimages)
