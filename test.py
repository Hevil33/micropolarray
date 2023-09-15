import glob
import os
import sys
from itertools import product
from logging import debug, error, info, warning

import matplotlib.pyplot as plt
import numpy as np
import pytest
from astropy.io import fits
from micropolarray.image import Image
from micropolarray.micropol_image import MicropolImage
from micropolarray.processing.demodulation import Demodulator
from micropolarray.processing.demosaic import demosaic

PIXELS = 16  # test data resolution pixels x pixels
TEST_FILES_PATH = "test_data/"


def write_generic_image():
    data = np.zeros(shape=(PIXELS, PIXELS))
    for i, j in product(range(PIXELS), range(PIXELS)):
        data[i, j] = i * j

    hul = fits.PrimaryHDU(
        data=data,
        do_not_scale_image_data=True,
        uint=False,
    )
    hul.writeto(TEST_FILES_PATH + "sample_image.fits", overwrite=True)


if __name__ == "__main__":
    data = np.ones(shape=(20, 20))
    write_generic_image()

    # -----------------------generic image class---------------------
    info("Testing: image.py")
    image = Image(TEST_FILES_PATH + "sample_image.fits")
    image.show()
    info("Done!")

    # -----------------------micropol image class---------------------
    print("-" * 30)
    info("Testing: micropol_image.py, Stokes_params.py")
    image = MicropolImage(data)
    image = MicropolImage(TEST_FILES_PATH + "sample_image.fits")
    image = MicropolImage(glob.glob(TEST_FILES_PATH + "sample_image.fits"))

    image.show_with_pol_params()
    image.save_as_fits(TEST_FILES_PATH + "/image.fits")
    info("Done!")

    print("-" * 30)
    info("Testing: process/demosaic.py (no saving files)")
    dummydata = np.array(
        [
            [1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2],
            [3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4],
            [1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2],
            [3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4],
            [1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2],
            [3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4],
            [1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2],
            [3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4],
            [1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2],
            [3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4],
            [1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2],
            [3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4],
            [1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2],
            [3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4],
            [1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2],
            [3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4],
        ]
    )
    image = MicropolImage(dummydata, demosaic_mode="mean")
    image.demosaic()
    for idx, demo_image in enumerate(image.demosaiced_images):
        if not np.array_equal(demo_image, np.full((16, 16), (idx + 1) / 4.0)):
            error(f"Demo image number: {idx}")
            print(demo_image)
            error("Should be:")
            print(np.full((16, 16), (idx + 1) / 4.0))
            sys.exit(1)
    image.demosaiced_images = demosaic(image.data, option="adjacent")
    for idx, demo_image in enumerate(image.demosaiced_images):
        if not np.array_equal(demo_image, np.full((16, 16), (idx + 1))):
            error(f"Demo image number: {idx}")
            print(demo_image)
            error("Should be:")
            print(np.full((16, 16), (idx + 1)))
            sys.exit(1)
    image.save_demosaiced_images_as_fits(TEST_FILES_PATH + "/image.fits")
    info("Done!")

    # ----------------------------------------------------------------
    print("-" * 30)
    info("Testing image.py rebinning")
    dummydata = np.array(
        [
            [1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2],
            [3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4],
            [1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2],
            [3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4],
            [1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2],
            [3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4],
            [1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2],
            [3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4],
            [1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2],
            [3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4],
            [1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2],
            [3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4],
            [1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2],
            [3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4],
            [1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2],
            [3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4],
        ]
    )
    image = MicropolImage(dummydata)
    rebinned_theo2 = np.array(
        [
            [4, 8, 4, 8, 4, 8, 4, 8],
            [12, 16, 12, 16, 12, 16, 12, 16],
            [4, 8, 4, 8, 4, 8, 4, 8],
            [12, 16, 12, 16, 12, 16, 12, 16],
            [4, 8, 4, 8, 4, 8, 4, 8],
            [12, 16, 12, 16, 12, 16, 12, 16],
            [4, 8, 4, 8, 4, 8, 4, 8],
            [12, 16, 12, 16, 12, 16, 12, 16],
        ]
    )
    binned_image_2 = image.rebin(2)
    image2 = MicropolImage(binned_image_2.data)
    if not np.array_equal(image2.data, rebinned_theo2):
        error("rebinning 2x2 is incorrect")
        print(f"binned: {image2.data}")
        print(f"theo: {rebinned_theo2}")
        sys.exit(1)
    rebinned_theo4 = np.array(
        [
            [16, 32, 16, 32],
            [48, 64, 48, 64],
            [16, 32, 16, 32],
            [48, 64, 48, 64],
        ]
    )
    binned_image_4 = image.rebin(4)
    image4 = MicropolImage(binned_image_4.data)
    if not np.array_equal(image4.data, rebinned_theo4):
        error("rebinning 4x4 is incorrect")
        print(f"binned: {image4.data}")
        print(f"theo: {rebinned_theo4}")
        sys.exit(1)

    info("Done!")

    print("-" * 30)
    info("All tests performed successfully!")
    print("-" * 30)
