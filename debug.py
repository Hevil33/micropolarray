import sys
import time
import tracemalloc
from glob import glob
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg

import micropolarray as ml
from micropolarray.processing.demodulation_errors import *
from micropolarray.processing.nrgf import remove_outliers_simple


def main():
    demodulator = ml.Demodulator(
        "/home/herve/dottorato/cormag/2023_flight/post_flight_calibration/polarimetria/demo_matrices_computation/demo_matrices/notilt/4"
    )
    image = ml.MicropolImage(
        "/home/herve/dottorato/cormag/2023_flight/post_flight_calibration/polarimetria/demo_matrices_computation/input_data/notilt/p0_sum.fits"
    ).rebin(4)

    image = image.demodulate(demodulator)

    image_error = MicropolImageError(
        image, image_error=(np.sqrt(image.data / 2.75)), demodulator=demodulator
    )

    print(f"{image_error.sigma_S / image.Stokes_vec.data = }")
    print()
    print(f"{image_error.sigma_AoLP / image.AoLP.data = }")
    print()
    print(f"{image_error.sigma_DoLP / image.DoLP.data = }")
    print()
    print(f"{image_error.sigma_pB / image.pB.data = }")
    print()


if __name__ == "__main__":
    main()
