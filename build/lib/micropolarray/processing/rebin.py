from numba import njit
import numpy as np
from logging import info


@njit
def micropolarray_jitrebin(data, height, width, binning=2):
    """Fast rebinning function for the micropolarray image."""
    # Skip last row/columns until they are divisible by binning,
    # allows any binning
    trimmed = False
    while width % (2 * binning):
        width -= 2
        trimmed = True
    while height % (2 * binning):
        height -= 2
        trimmed = True
    if trimmed:
        print(
            "Data trimmed to fit rebinning. Starting from new shape: (",
            width,
            ", ",
            height,
            ")",
        )
    new_width = int(width / binning)
    new_height = int(height / binning)
    new_data = np.zeros(shape=(new_height, new_width))
    for new_y in range(new_height):
        for new_x in range(new_width):
            for y_scaler in range((new_y % 2), (new_y % 2) + 2 * binning, 2):
                for x_scaler in range(
                    (new_x % 2), (new_x % 2) + 2 * binning, 2
                ):
                    i = (new_y - new_y % 2) * binning + y_scaler
                    j = (new_x - new_x % 2) * binning + x_scaler
                    new_data[new_y, new_x] += data[i, j]
    return new_data


def standard_rebin(image, binning: int) -> np.array:
    """Rebins the image, binned each binningxbinning. Sum bins by
    default, unless mean = True is specified.

    Args:
        image (MicroPolarizerArrayImage): image to be binned
        binning (int): binning to be applied. A value of 2 will result in a 2x2 binning (1 pixel is a sum of 4 neighbour pixels)

    Raises:
        KeyError: cannot divide image height/width by the binning value

    Returns:
        np.array: binned image
    """
    if (image.width % binning) or (image.height % binning):
        raise KeyError(
            "Invalid binning, must be divisor of both image height and width."
        )
    rebinned_data = standard_jitrebin(
        image.data, image.width, image.height, binning
    )
    return rebinned_data


@njit
def standard_jitrebin(data, width, height, binning=2):
    trimmed = False
    while width % (2 * binning):
        width -= 2
        trimmed = True
    while height % (2 * binning):
        height -= 2
        trimmed = True
    if trimmed:
        print(
            "Data trimmed to fit rebinning. Starting from new shape: (",
            width,
            ", ",
            height,
            ")",
        )
    new_height = int(height / binning)
    new_width = int(width / binning)
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


def trim_to_match_2xbinning(
    height: int, width: int, binning: int, verbose=True
):
    """Deletes the last image pixels until superpixel binning is compatible with new dimensions

    Args:
        height (int): image height_
        width (int): image width
        binning (int): image binning
        verbose (bool, optional): warns user of trimming. Defaults to True.

    Returns:
        int, int: image new height and width
    """
    trimmed = False
    while width % (2 * binning):
        width -= 2
        trimmed = True
    while height % (2 * binning):
        height -= 2
        trimmed = True
    if trimmed and verbose:
        info(
            "Data trimmed to fit rebinning. Starting from new shape: (",
            width,
            ", ",
            height,
            ")",
        )
    return height, width


def trim_to_match_binning(height, width, binning, verbose=True):
    """Deletes the last image pixels until simple binning is compatible with new dimensions

    Args:
        height (int): image height_
        width (int): image width
        binning (int): image binning
        verbose (bool, optional): warns user of trimming. Defaults to True.

    Returns:
        int, int: image new height and width
    """
    trimmed = False
    while width % (binning):
        width -= 1
        trimmed = True
    while height % (binning):
        height -= 1
        trimmed = True
    if trimmed and verbose:
        info(
            "Data trimmed to fit rebinning. Starting from new shape: (",
            width,
            ", ",
            height,
            ")",
        )
    return height, width
