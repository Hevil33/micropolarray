import sys
from itertools import product

import numpy as np
from numba import njit

from ..micropol_image import MicropolImage
from ..utils import timer
from .demodulation import Demodulator
from .demosaic import split_polarizations


def get_error_on_demodulation(
    image_error: np.ndarray, demodulator: Demodulator
) -> np.ndarray:
    """Returns the error on the image, propagated through the demodulation matrix. If M[i, j] is the demodulation matrix, sigma_I[k] are the four pixel values in a superpixel, and S[i, j] is the Stokes vector, returns
    sqrt(M @ (sqrt(I /gain)))

    Args:
        image_error (np.ndarray): array containing the pixel by pixel error to propagate.
        demodulator (Demodulator): demodulator containing the demodulation matrix.

    Returns:
        np.ndarray: errors of the computed Stokes vector as a [3, y, x] array.
    """
    mij_square = np.multiply(demodulator.mij, demodulator.mij)

    single_pol_subimages = split_polarizations(image_error)
    pixel_poisson_variance = np.expand_dims(
        np.multiply(single_pol_subimages, single_pol_subimages), axis=0
    )

    # S_variance = mij * sigma_image
    S_variance = np.matmul(
        mij_square,
        pixel_poisson_variance,
        axes=[(-4, -3), (-3, -4), (-4, -3)],
    )[:, 0]

    return np.sqrt(S_variance)
