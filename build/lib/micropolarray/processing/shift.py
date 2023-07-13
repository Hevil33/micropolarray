import numpy as np
from numba import njit


@njit
def shift(data: np.ndarray, y: int, x: int):
    newdata = np.zeros_like(data)
    for j in range((y > 0) * y, newdata.shape[0] - np.abs(y) * (y < 0)):
        for i in range((x > 0) * x, newdata.shape[1] - np.abs(x) * (x < 0)):
            newdata[j, i] = data[j - y, i - x]

    return newdata
