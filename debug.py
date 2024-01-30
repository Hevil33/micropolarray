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
from micropolarray.processing.linear_roi import DDA, linear_roi, linear_roi_from_polar
from micropolarray.processing.nrgf import remove_outliers_simple


def main():
    image = ml.MicropolImage(
        "/home/herve/dottorato/antarticor/herve/campagna_2022/results/2021_12_11/corona_0/corona.fits"
    )
    # image = ml.MicropolImage(
    #    "/home/herve/dottorato/cormag/2023_flight/L1/volo fase 1/seq. 10/sum_tilted.fits"
    # )
    fig, ax = image.show_pol_param("pB")
    # plt.show()

    y, x, r = ml.PolarCam().occulter_pos_last
    print(y, x, r)

    x = x / 2
    y = y / 2
    r = r / 2
    ax.add_artist(plt.Circle((x, y), r, fill=False, alpha=0.5))

    fig2, ax2 = plt.subplots(dpi=200)
    if True:
        # ax.axvline(0)
        # ax.axvline(image.pB.data.shape[0])
        # ax.axhline(0)
        # ax.axhline(image.pB.data.shape[1])

        for i, angle in enumerate(np.arange(-180, 180, 45)):
            result, ys, xs, ratio = linear_roi_from_polar(
                image.pB.data,
                (y, x),
                angle,
            )

            ax.plot(xs, ys, label=f"{angle:3.2f}")
            ax2.plot(range(len(result)) * ratio, result, label=f"{i:3.2f}")

        ax.legend()
        ax.add_artist(plt.Circle((x, y), 30, fill=True, alpha=1))
        plt.show()

    pixels = linear_roi(image.pB.data, (y, x), (233, 970))

    function = DDA

    ax.plot(*function((y, x), (930, 934)))
    ax.plot(*function((y, x), (930, 300)))
    ax.plot(*function((y, x), (650, 27)))
    ax.plot(*function((y, x), (271, 22)))
    ax.plot(*function((y, x), (11, 296)))
    ax.plot(*function((y, x), (11, 739)))

    plt.show()


if __name__ == "__main__":
    main()
