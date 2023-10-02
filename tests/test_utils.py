import glob
import os

import numpy as np
import pytest

import micropolarray as ml

# import micropolarray as ml
from micropolarray.processing.demodulation import Malus


@pytest.fixture(autouse=True)
def generate_dummy_data():
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
    return ml.merge_polarizations(subimages)


def generate_polarimetric_measurements(tmp_path, input_signal, t, eff, shape):
    shape = (shape, shape)
    # try demodulation
    angles = np.array([np.deg2rad(angle) for angle in [0, 45, -45, 90]])
    output_dir = tmp_path / "computed_matrix"
    output_str = str(output_dir)

    polarizations = np.arange(-45, 91, 15)
    pols_rad = np.deg2rad(polarizations)

    ones = np.ones(shape=shape)
    for pol, pol_rad in zip(polarizations, pols_rad):
        result_image = ml.MicropolImage(
            generate_polarized_data(
                shape=shape,
                S=input_signal,
                angle_rad=pol_rad,
                t=t,
                eff=eff,
            )
        )
        result_image.save_as_fits(tmp_path / f"pol_{int(pol)}.fits")
    if False:  # check that fit will be ok
        for angle in angles:
            pars, pcov = curve_fit(
                Malus,
                pols_rad,
                np.array([Malus(pol, t, eff, angle) for pol in pols_rad]),
            )
            print(f"t = {pars[0]}")
            print(f"eff = {pars[1] }")
            print(f"phi = {np.rad2deg(pars[2]) }")

    # read the files
    filenames = sorted(
        glob.glob(str(tmp_path / "pol*.fits")),
        key=lambda x: int(x.split(os.path.sep)[-1][4:].strip(".fits")),
    )
