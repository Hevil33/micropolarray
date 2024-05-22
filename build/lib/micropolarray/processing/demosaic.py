import sys
import warnings
from logging import info

import numpy as np
from numba import njit
from numba.typed import List

from micropolarray.processing.congrid import congrid

warnings.filterwarnings(
    "ignore", category=UserWarning, append=True
)  # Warnings of fits package became annoying.


@njit  # MAKES OPERATIONS REALLY FAST, checked
def demosaicmean(data):
    """Loops over right polarization pixel location, takes 1/4 of that,
    stores it in the 2x2 superpixel.
    demo_images[0] = data[y=0, x=0]
    demo_images[1] = data[y=0, x=1]
    demo_images[2] = data[y=1, x=0]
    demo_images[3] = data[y=1, x=1]
    """
    data = 1.0 * data
    temp_data = data.copy()
    demo_images = [
        np.zeros_like(temp_data),
        np.zeros_like(temp_data),
        np.zeros_like(temp_data),
        np.zeros_like(temp_data),
    ]
    counter = 0
    for j in range(2):
        for i in range(2):
            for y_super in range(0, data.shape[0], 2):
                for x_super in range(0, data.shape[1], 2):
                    mean = data[y_super + j, x_super + i] * 0.25
                    temp_data[y_super : y_super + 2, x_super : x_super + 2] = mean
            demo_images[counter] = temp_data.copy()
            counter += 1

    return demo_images


@njit
def demosaicadjacent(data):
    data = 1.0 * data
    temp_data = data.copy()
    demo_images = [
        np.zeros_like(temp_data),
        np.zeros_like(temp_data),
        np.zeros_like(temp_data),
        np.zeros_like(temp_data),
    ]
    counter = 0
    for y_pix_family in range(2):
        for x_pix_family in range(2):
            for y in range(0, data.shape[0] - 2, 2):
                for x in range(0, data.shape[1] - 2, 2):
                    temp_data[y, x] = data[y + y_pix_family, x + x_pix_family]
                    temp_data[y + 1, x] = 0.5 * (
                        data[y + y_pix_family, x + x_pix_family]
                        + data[y + y_pix_family + 2, x + x_pix_family]
                    )
                    temp_data[y, x + 1] = 0.5 * (
                        data[y + y_pix_family, x + x_pix_family]
                        + data[y + y_pix_family, x + x_pix_family + 2]
                    )
                    temp_data[y + 1, x + 1] = (
                        data[y + y_pix_family, x + x_pix_family]
                        + data[y + y_pix_family + 2, x + x_pix_family]
                        + data[y + y_pix_family, x + x_pix_family + 2]
                        + data[
                            y + y_pix_family + 2,
                            x + x_pix_family + 2,
                        ]
                    ) * 0.25
            demo_images[counter] = temp_data.copy()
            counter += 1

    return demo_images


def split_polarizations(data: np.ndarray):
    if (data.shape[0] % 2) or (data.shape[1] % 2):
        raise ValueError("Odd number of pixels, can't split polarizations.")

    single_pol_images = np.array(
        [data[j::2, i::2] for j in range(2) for i in range(2)],
        # [
        #    self.data[0::2, 0::2], # x= 0, y = 0
        #    self.data[0::2, 1::2], # x= 1, y = 0
        #    self.data[1::2, 0::2], # x= 0, y = 1
        #    self.data[1::2, 1::2], # x= 1, y = 1
        # ],
        dtype=float,
    )

    return single_pol_images


@njit
def merge_polarizations(single_pol_images: np.ndarray):
    data = np.zeros(
        shape=(
            single_pol_images[0].shape[0] * 2,
            single_pol_images[0].shape[1] * 2,
        )
    )
    data[0::2, 0::2] = single_pol_images[0]
    data[0::2, 1::2] = single_pol_images[1]
    data[1::2, 0::2] = single_pol_images[2]
    data[1::2, 1::2] = single_pol_images[3]
    return data


def demosaic(image_data, option="adjacent"):
    """
    Returns a [4,n,m] array of polarized images, starting from a
    micropolarizer image array [n, m].
    """
    data = image_data
    temp_data = data.copy()
    demo_images = np.empty([4, data.shape[0], data.shape[1]], dtype="d")

    if option == "mean":
        info("Demosaicing (mean method)... ")
        demo_images = demosaicmean(
            np.array(image_data, dtype=float)
        )  # casting needed by numba

    elif option == "adjacent":
        """
        longer: loops over superpixel, tries to get value of right polarized pixel, except if at image boundary. Then takes the mean of adjacent cells, stores it temporarily in the 2x2 pixel, then rotates it to match the right
        """
        info("Demosaicing (adjacent method)...")
        # Adding two new columns/rows to avoid segmentationfault-like error
        stacked_data = np.vstack([image_data, image_data[-2:, :]])
        stacked_data = np.hstack([stacked_data, stacked_data[:, -2:]])
        demo_images = demosaicadjacent(stacked_data)
        demo_images = np.array(demo_images)[:, :-2, :-2]

    elif option == "spline":
        single_pol_subimages = np.array(
            [image_data[j::2, i::2] for j in range(2) for i in range(2)],
            dtype=np.float,
        )
        demo_images = [
            congrid(
                single_pol_subimages[i],
                (image_data.shape[0], image_data.shape[1]),
            )
            for i in range(4)
        ]

    else:
        raise ValueError('"option" should be one of ["mean", "adjacent"]')

    return np.array(demo_images)
