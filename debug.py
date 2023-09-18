import sys
import time
import tracemalloc
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg

import micropolarray as ml
from micropolarray.processing.nrgf import remove_outliers_simple


def main():
    filename = glob(
        "/media/herve/TOSHIBA EXT/dottorato/cormag/2023_flight/data/images/volo fase 1/seq. 10/*[!no]_tilt*.fits"
    )[0]

    image = ml.MicropolImage(filename)
    # image.mask_occulter(1471, 2172, 1181)
    image.mask_occulter(1471, 2172, 500)
    image.show_with_pol_params()

    plt.show()

    # image2 = ml.PolarcamImage(data)


if __name__ == "__main__":
    main()
