import sys
import time

import matplotlib.pyplot as plt
import micropolarray as ml
import numpy as np
from micropolarray.processing.nrgf import remove_outliers_simple
from scipy.ndimage.filters import gaussian_filter


def main():
    image = ml.MicroPolarizerArrayImage("./test_data/image.fits")
    image.save_as_raw("./test_data/image.raw")


if __name__ == "__main__":
    main()
