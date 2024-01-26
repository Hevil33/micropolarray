from logging import info

import numpy as np
from numba import jit, njit


def print_trimming_info(height, width, new_height, new_width):
    # print(
    #    f"Data trimmed to fit rebinning: ({height}, {width}) -> ({new_height}, {new_width})"
    # )
    print(
        "Data trimmed to fit rebinning: (",
        height,
        ",",
        width,
        ") -> (",
        new_height,
        ",",
        new_width,
        ")",
    )


def micropolarray_rebin(data: np.ndarray, binning=2):
    """Wrapper for the faster rebinning donw with numba. First deletes last row/column until binning is possible, then calls binning on the result shape.

    Args:
        data (np.ndarray): data to rebin
        height (int): lenght of first axis
        width (int): lenght of second axis
        binning (int, optional): Binning to be performed. Defaults to 2.

    Returns:
        ndarray: binned data, trimmed if necessary
    """
    height, width = data.shape
    new_height, new_width = trim_to_match_2xbinning(height, width, binning)
    new_height = int(new_height / binning)
    new_width = int(new_width / binning)
    new_data = micropolarray_jitrebin(data, new_height, new_width, binning)

    return new_data


@njit
def micropolarray_jitrebin_old(data, height, width, binning=2):
    """Fast rebinning function for the micropolarray image."""
    # Skip last row/columns until they are divisible by binning,
    # allows any binning
    new_height, new_width = trim_to_match_2xbinning(height, width, binning)
    new_height = int(new_height / binning)
    new_width = int(new_width / binning)
    new_data = np.zeros(shape=(new_height, new_width), dtype=float)
    for new_y in range(new_height):
        for new_x in range(new_width):
            for y_scaler in range((new_y % 2), (new_y % 2) + 2 * binning, 2):
                for x_scaler in range((new_x % 2), (new_x % 2) + 2 * binning, 2):
                    i = (new_y - new_y % 2) * binning + y_scaler
                    j = (new_x - new_x % 2) * binning + x_scaler
                    new_data[new_y, new_x] += data[i, j]

    return new_data


@njit
def micropolarray_jitrebin(data, new_height, new_width, binning=2):
    """Fast rebinning function for the micropolarray image. Needs to be wrapped to print info."""
    new_data = np.zeros(shape=(new_height, new_width), dtype=float)
    for new_y in range(new_height):
        for new_x in range(new_width):
            for y_scaler in range((new_y % 2), (new_y % 2) + 2 * binning, 2):
                for x_scaler in range((new_x % 2), (new_x % 2) + 2 * binning, 2):
                    i = (new_y - new_y % 2) * binning + y_scaler
                    j = (new_x - new_x % 2) * binning + x_scaler
                    new_data[new_y, new_x] += data[i, j]

    return new_data


def standard_rebin(data, binning: int) -> np.array:
    """Rebins the data, binned each binningxbinning.

    Args:
        image (np.array): data to be binned
        binning (int): binning to be applied. A value of 2 will result in a 2x2 binning (1 pixel is a sum of 4 neighbour pixels)

    Raises:
        KeyError: cannot divide image height/width by the binning value

    Returns:
        np.array: binned data
    """
    height, width = data.shape
    new_height, new_width = trim_to_match_binning(height, width, binning)
    new_height = int(new_height / binning)
    new_width = int(new_width / binning)

    rebinned_data = standard_jitrebin(data, new_height, new_width, binning)
    return rebinned_data


@njit
def standard_jitrebin(data, new_height, new_width, binning=2):
    new_data = np.zeros(shape=(new_height, new_width))
    for new_y in range(new_height):
        for new_x in range(new_width):
            for y_scaler in range(binning):
                for x_scaler in range(binning):
                    new_data[new_y, new_x] += data[
                        binning * new_y + y_scaler,
                        binning * new_x + x_scaler,
                    ]

    return new_data


def trim_to_match_2xbinning(height: int, width: int, binning: int):
    """Deletes the last image pixels until superpixel binning is compatible with new dimensions

    Args:
        height (int): image height
        width (int): image width
        binning (int): image binning

    Returns:
        int, int: image new height and width
    """
    trimmed = False
    new_width = width
    new_height = height
    while new_width % (2 * binning):
        new_width -= 2
        trimmed = True
    while new_height % (2 * binning):
        new_height -= 2
        trimmed = True
    if trimmed:
        print_trimming_info(height, width, new_height, new_width)

    return new_height, new_width


def trim_to_match_binning(height, width, binning, verbose=True):
    """Deletes the last image pixels until simple binning is compatible with new dimensions

    Args:
        height (int): image height
        width (int): image width
        binning (int): image binning
        verbose (bool, optional): warns user of trimming. Defaults to True.

    Returns:
        int, int: image new height and width
    """
    trimmed = False
    new_height = height
    new_width = width
    while new_width % (binning):
        new_width -= 1
        trimmed = True
    while new_height % (binning):
        new_height -= 1
        trimmed = True
    if trimmed and verbose:
        print_trimming_info(height, width, new_height, new_width)
    return new_height, new_width
