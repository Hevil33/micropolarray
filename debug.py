import sys
import time

import matplotlib.pyplot as plt
import micropolarray as ml
import numpy as np
from micropolarray.processing.nrgf import remove_outliers_simple
from scipy.ndimage.filters import gaussian_filter


def main():
    image = ml.MicroPolarizerArrayImage("./.fits")

    image.show(
        vmin=ml.median_minus_std(image.data),
        vmax=ml.median_plus_std(image.data),
    )

    blurred = gaussian_filter(image.data, sigma=17)

    occulter_pos = ml.find_occulter_position(blurred)

    print(occulter_pos)


if __name__ == "__main__":
    main()
