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
from micropolarray.processing.nrgf import remove_outliers_simple


def main():
    image = ml.MicroPolarizerArrayImage(
        "/home/herve/dottorato/cormag/2023_flight/L1/volo fase 3 (post reboot)/seq.  1/sum_notilted.fits"
    )

    x, y, r = ml.find_occulter_hough(image.data)

    fig, ax = image.show()
    ax.add_artist(plt.Circle((x, y), r, alpha=0.5))
    plt.show()


if __name__ == "__main__":
    main()
