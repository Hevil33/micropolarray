import numpy as np


"""----------------------------------------------------------------------"""
""" 
    Functions to compute Angle of Linear Polarization (AoLP), Degree of
    Linear Polarization (DoLP) and Polarized Brightness (pB), returned as 
    numpy arrays.
    
    Args:
        Stokes_vec_components (numpy.array[3, number of y pixels, number of x pixels]): array containing elements of the Stokes vector of an np.array[y, x] image, in the form [S0, S1, S2].
"""


def AoLP(Stokes_vec_components):
    """Angle of linear polarization in [rad]"""
    I, Q, U = Stokes_vec_components
    angle = np.where(
        Q != 0.0,
        0.5 * np.arctan(1.0 * U / (1.0 * Q), dtype=float),
        np.deg2rad(90.0),
    )  # set it to 90 deg when denominator explodes
    return angle
    return 0.5 * np.arctan2(U, Q)


def DoLP(Stokes_vec_components):
    """Degree of linear polarization in [%]"""
    I, Q, U = Stokes_vec_components
    return np.divide(
        np.sqrt((Q * Q) + (U * U), dtype=float), I, where=(I != 0)
    )  # avoids 0/0 error


def pB(Stokes_vec_components):
    """Polarized brighness in [%]"""
    I, Q, U = Stokes_vec_components
    return np.sqrt((Q * Q) + (U * U), dtype=float)


"""----------------------------------------------------------------------"""


def normalize2pi(angles_list):
    if type(angles_list) is not list:
        angles_list = [
            angles_list,
        ]
    for i, angle in enumerate(angles_list):
        while angle > 90:
            angle -= 180
        while angle <= -90:
            angle += 180
        angles_list[i] = angle

    return angles_list
