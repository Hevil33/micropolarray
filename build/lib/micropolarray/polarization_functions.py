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
    return 0.5 * np.arctan2(U, Q)


def DoLP(Stokes_vec_components):
    """Degree of linear polarization in [%]"""
    I, Q, U = Stokes_vec_components
    return np.divide(
        np.sqrt((Q * Q) + (U * U)), I, where=(I != 0)
    )  # avoids 0/0 error


def pB(Stokes_vec_components):
    """Polarized brighness in [%]"""
    I, Q, U = Stokes_vec_components
    return np.sqrt((Q * Q) + (U * U))


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
