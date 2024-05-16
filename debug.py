import sys
import time
import tracemalloc
from glob import glob
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg
from scipy.ndimage import median_filter
from skimage.filters.thresholding import threshold_multiotsu, threshold_otsu

import micropolarray as ml
from micropolarray.processing.demodulation_errors import *
from micropolarray.processing.image_cleaning import remove_outliers_simple
from micropolarray.processing.linear_roi import DDA, linear_roi, linear_roi_from_polar


def main():
    image = ml.MicropolImage(
        "/home/herve/dottorato/antarticor/herve/campagna_2022/results/2021_12_11/corona_0/corona.fits"
    )

    nrfg = ml.nrgf(image.data, 1500, 2078, rho_min=300)

    fig, ax = plt.subplots(dpi=200)
    ax.imshow(nrfg)
    plt.show()


if __name__ == "__main__":
    main()
