import numpy as np
from numba import njit

from micropolarray.processing.demosaic import (
    merge_polarizations,
    split_polarizations,
)


@njit
def shift(data: np.ndarray, y: int, x: int):
    newdata = np.zeros_like(data)
    for j in range((y > 0) * y, newdata.shape[0] - np.abs(y) * (y < 0)):
        for i in range((x > 0) * x, newdata.shape[1] - np.abs(x) * (x < 0)):
            newdata[j, i] = data[j - y, i - x]

    return newdata


def shift_micropol(data: np.ndarray, y: int, x: int):
    """Splits the image into single polarizations, shifts each of them by y,x and then merges them back.

    Args:
        data (np.ndarray): array to shift
        y (int): vertical shift (positive inside the image)
        x (int): horizontal shift (positive inside the image)

    Returns:
        np.ndarray: shifted array
    """
    single_pol_subimages = split_polarizations(data)
    new_single_pol_subimages = np.zeros_like(single_pol_subimages)

    for i, image in enumerate(single_pol_subimages):
        new_single_pol_subimages[i, :, :] = shift(image, y, x)

    return merge_polarizations(new_single_pol_subimages)
