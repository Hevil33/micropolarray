import sys
import time

import matplotlib.pyplot as plt
import micropolarray as ml
import numpy as np
import numpy.linalg
from micropolarray.processing.nrgf import remove_outliers_simple
from scipy.ndimage.filters import gaussian_filter


def main():
    t = 1
    e = 1
    angles = np.array([0, 45, -45, 90])
    angles = np.deg2rad(angles)
    ml.set_default_angles({0: 0, 45: 1, -45: 2, 90: 3})

    theo = (
        0.5
        * t
        * np.array(
            [
                [1, e * np.cos(2.0 * angles[0]), e * np.sin(2.0 * angles[0])],
                [1, e * np.cos(2.0 * angles[1]), e * np.sin(2.0 * angles[1])],
                [1, e * np.cos(2.0 * angles[2]), e * np.sin(2.0 * angles[2])],
                [1, e * np.cos(2.0 * angles[3]), e * np.sin(2.0 * angles[3])],
            ]
        )
    )
    print(theo.shape)
    theo_1 = np.array(
        [
            [0.5, 0.5, 0],
            [0.5, 0, 0.5],
            [0.5, 0, -0.5],
            [0.5, -0.5, 0],
        ]
    )

    X = numpy.linalg.pinv(theo)
    X_1 = numpy.linalg.pinv(theo_1)

    folder = "./test_data/dummy_16x16"
    for i in range(3):
        for j in range(4):
            mij = np.full(shape=(16, 16), fill_value=X[i, j])
            image = ml.Image(mij)
            image.save_as_fits(folder + f"/m{i}{j}.fits")

    demodulator = ml.Demodulator(folder)

    data = np.zeros((16, 16))
    data[0::2, 0::2] = 1
    data[0::2, 1::2] = 0.5
    data[1::2, 0::2] = 0.5
    data[1::2, 1::2] = 0

    image = ml.MicroPolarizerArrayImage(
        "./test_data/image.fits", demosaic_mode="adjacent"
    )
    # image = ml.MicroPolarizerArrayImage(data)
    image.show_with_pol_params()

    image = image.demodulate(demodulator)
    image.show_with_pol_params()
    plt.show()

    # image.save_as_raw("./test_data/image.raw")


if __name__ == "__main__":
    main()
