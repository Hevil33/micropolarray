import sys
import time

import matplotlib.pyplot as plt
import micropolarray as ml
import numpy as np
from micropolarray.processing.nrgf import remove_outliers_simple
from scipy.ndimage.filters import gaussian_filter


def main():
    image = ml.MicroPolarizerArrayImage("./test_data/antarticor_flat.fits")

    image.show(
        vmin=ml.median_minus_std(image.data),
        vmax=ml.median_plus_std(image.data),
    )

    blurred = gaussian_filter(image.data, sigma=17)
    # occulter_pos = ml.find_occulter_position(blurred)
    occulter_pos = ml.find_occulter_position(image.data, method="algo")
    print(occulter_pos)

    fig, ax = image.show()
    circle = plt.Circle(
        (occulter_pos[0], occulter_pos[1]),
        occulter_pos[2],
        fill=False,
        color="red",
    )
    circle2 = plt.Circle(
        (
            ml.PolarCam().occulter_pos_last[0],
            ml.PolarCam().occulter_pos_last[1],
        ),
        ml.PolarCam().occulter_pos_last[2],
        fill=False,
        color="green",
    )
    ax.add_artist(circle)
    ax.add_artist(circle2)

    plt.show()


if __name__ == "__main__":
    main()
