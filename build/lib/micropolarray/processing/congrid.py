import numpy as np
import scipy.interpolate
import scipy.ndimage
from numba import njit
from scipy.interpolate import griddata


# not working
@njit
def micropolarray_jitcongrid(data, width, height, scale):
    new_width = int(width * scale)
    new_height = int(height * scale)
    new_data = np.zeros(shape=(new_height, new_width), dtype=float)
    meshgrid = np.zeros(shape=(new_height, new_width, 2))
    centers = [
        [np.sqrt((0.5 + i) ** 2 + (0.5 + j) ** 2) for i in range(new_height)]
        for j in range(new_width)
    ]

    # Fill grid with nearest points
    for new_i in range(new_width):
        for new_j in range(new_height):
            pass
    return centers


# def interpolate(x, x_0, y_0, x_1, y_1):
#    return (x - x_0) * ((y_1 - y_0) / (x_1 - x_0)) + y_0


# Copied from IDL congrid, arbitrary reshape
def congrid(a, newdims, kind="linear") -> np.ndarray:
    """Reshapes the data into any new lenght and width

    Args:
        a (np.array): data to be reshaped
        newdims (tuple | list): new lenght and width
        kind (str, optional): interpolation type. Defaults to "linear".

    Returns:
        ndarray: numpy array of congridded image
    """
    a = np.cast[float](a)

    m1 = np.cast[int](True)
    old = np.array(a.shape)
    ndims = len(a.shape)
    newdims = np.asarray(newdims, dtype=float)
    dimlist = []

    # Linear method, default
    for i in range(ndims):
        base = np.arange(newdims[i])
        dimlist.append((old[i] - m1) / (newdims[i] - m1) * base)

    olddims = [np.arange(i, dtype=float) for i in list(a.shape)]
    mint = scipy.interpolate.interp1d(
        olddims[-1], a, kind=kind, fill_value="extrapolate"
    )
    newa = mint(dimlist[-1])

    trorder = [ndims - 1] + list(range(ndims - 1))
    for i in range(ndims - 2, -1, -1):
        newa = newa.transpose(trorder)
        mint = scipy.interpolate.interp1d(
            olddims[i], newa, kind=kind, fill_value="extrapolate"
        )
        newa = mint(dimlist[i])
    if ndims > 1:
        newa = newa.transpose(trorder)

    return newa
