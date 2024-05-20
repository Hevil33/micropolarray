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

    fig, ax, fig2, ax2 = image.show_with_pol_params()
    plt.show()


if __name__ == "__main__":
    main()
