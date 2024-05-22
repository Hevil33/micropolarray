import sys
from itertools import product

import numpy as np
from numba import njit

from ..micropol_image import MicropolImage
from ..utils import timer
from .demodulation import Demodulator
from .demosaic import split_polarizations


class MicropolImageError:
    def __init__(
        self, image: MicropolImage, image_error: np.ndarray, demodulator: Demodulator
    ) -> None:
        self.sigma_S = get_error_on_Stokes(
            image_error=image_error, demodulator=demodulator
        )
        self.sigma_pB = get_error_on_pB(image.Stokes_vec, self.sigma_S)
        self.sigma_DoLP = get_error_on_DoLP(image.Stokes_vec, self.sigma_S)
        self.sigma_AoLP = get_error_on_AoLP(image.Stokes_vec, self.sigma_S)


def get_error_on_Stokes(
    image_error: np.ndarray, demodulator: Demodulator
) -> np.ndarray:
    """Returns the error on the image, propagated through the demodulation matrix. If M[i, j] is the demodulation matrix, sigma_I[k] are the four pixel values in a superpixel, and S[i, j] is the Stokes vector, returns the matrix product
    sqrt(M^2 @ I^2)

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


def get_error_on_pB(S: np.ndarray, sigma_S: np.ndarray) -> np.ndarray:
    I, Q, U = S
    sigma_I, sigma_Q, sigma_U = sigma_S

    pb_var = (Q * Q * sigma_Q * sigma_Q + U * U * sigma_U * sigma_U) / (Q * Q + U * U)

    return np.sqrt(pb_var)


def get_error_on_DoLP(S: np.ndarray, sigma_S: np.ndarray) -> np.ndarray:
    I, Q, U = S
    sigma_I, sigma_Q, sigma_U = sigma_S

    pB = np.sqrt(Q * Q + U * U)

    dolp_var = (Q * Q * sigma_Q * sigma_Q + U * U * sigma_U * sigma_U) / (
        (I * pB) * (I * pB)
    ) + (pB / (I * I)) * (pB / (I * I)) * sigma_I * sigma_I

    return np.sqrt(dolp_var)


def get_error_on_AoLP(S: np.ndarray, sigma_S: np.ndarray) -> np.ndarray:
    I, Q, U = S
    sigma_I, sigma_Q, sigma_U = sigma_S

    aolp_var = (sigma_U * sigma_U + sigma_Q * sigma_Q * U * U / (Q * Q)) / (
        4 * Q * Q * (1 + (U * U / (Q * Q))) * (1 + (U * U / (Q * Q)))
    )

    return np.sqrt(aolp_var)
