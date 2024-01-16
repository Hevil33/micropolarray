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
    error_S = get_error_on_demodulation(np.sqrt(image.data / 2.75), demodulator)

    fig, ax = image.show_histogram(split_pols=True, bins=1000)
    fig.set_dpi(200)

    fig, ax = plt.subplots(dpi=200)
    hist = np.histogram(error_S[0] / image.I.data, bins=1000)
    ax.stairs(*hist)

    plt.show()


if __name__ == "__main__":
    main()
