import sys

import matplotlib.pyplot as plt
import numpy as np

from micropolarray.utils import normalize2pi


def linear_roi_from_polar(
    data: np.ndarray,
    center: list,
    theta: float,
    r: list = None,
) -> list:
    """Returns a linear roi starting from the center and extending
    to r or to the edge of the input data array.
    Angles start vertically and rotate anti-clockwise (0deg
    corresponds to fixed x and increasing y).

    Args:
        data (np.ndarray): input array from which to select a roi
        center (list): center coordinates
        theta (float): angle of the linear roi
        r (list, optional): Maximum radius for the roi. Defaults to
        None.

    Returns:
        np.ndarray: 1-dimensional array containing the selected values
        from data
        np.ndarray: roi indexes along the first (y) dimension of data
        np.ndarray: roi indexes along the second (x) dimension of data
        float: ratio between the pixel lenght and the lenght of the
        returned roi (see linear_roi.DDA)
    """
    y1, x1 = center
    y2, x2 = float(y1), float(x1)

    if r is None:
        r = 1.0e18

    while (
        (y2 < data.shape[0] - 1)
        and (y2 > 1)
        and (x2 < data.shape[1] - 1)
        and (x2 > 1)
        and (np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2) < r - 1)
    ):
        y2 = y2 - np.sin(theta)
        x2 = x2 - np.cos(theta)

    ys, xs, points_density = DDA((y1, x1), (y2, x2))
    while (
        (ys[-1] >= data.shape[0])
        or (xs[-1] >= data.shape[1])
        or (ys[-1] < 0)
        or (xs[-1] < 0)
    ):
        ys = ys[:-2]
        xs = xs[:-2]

    result = np.array([data[y, x] for y, x in zip(ys, xs)])

    return [result, ys, xs, points_density]


def linear_roi(data: np.ndarray, start: list, end: list) -> np.ndarray:
    """Get values

    Args:
        data (np.ndarray): _description_
        start (list): _description_
        end (list): _description_

    Returns:
        np.ndarray: _description_
    """
    ys, xs = DDA(start, end)

    vals = np.array([data[y, x] for y, x in zip(ys, xs)])

    return vals


def DDA(start: list, end: list) -> np.ndarray:
    """Digital_differential_analyzer algorithm for line rasterizing.
    Unlike bresenham, works in every quadrant.
    NOTE: even if the distance between start and end coordinates is
    the same, a different number of points is selected depending on
    the line slope, so the ratio between distance and number of
    points is also returned.

    Args:
        start (list): starting point coordinates
        end (list): ending point coordinates

    Returns:
        np.ndarray: interpolated points locations
        float: ratio between the distance from start to end point and
        the number of returned locations
    """
    y1, x1 = [int(i) for i in start]
    y2, x2 = [int(i) for i in end]

    dx = x2 - x1
    dy = y2 - y1
    if np.abs(dx) >= np.abs(dy):
        step = np.abs(dx)
    else:
        step = np.abs(dy)
    dx = dx / step
    dy = dy / step
    x = x1
    y = y1
    xs = [int(x)]
    ys = [int(y)]

    i = 0
    while i <= step:
        x = x + dx
        y = y + dy
        i = i + 1
        # if (int(y) != ys[-1]) and (int(x) != xs[-1]):
        ys.append(int(y))
        xs.append(int(x))

    points_density = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2) / len(xs)

    return np.array(ys), np.array(xs), points_density


def bresenham(start: list, end: list) -> np.ndarray:
    """Bresenham algorithm for generating integers on a line.
    Efficient BUT works ONLY in the first octant

    Args:
        start (list): starting point coordinates
        end (list): ending point coordinates

    Returns:
        np.ndarray: coordinates of the points under the line from
        start to end
    """
    y1, x1 = [int(i) for i in start]
    y2, x2 = [int(i) for i in end]

    x, y = x1, y1
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    gradient = dy / float(dx)

    if gradient > 1:
        dx, dy = dy, dx
        x, y = y, x
        x1, y1 = y1, x1
        x2, y2 = y2, x2

    p = 2 * dy - dx
    # Initialize the plotting points
    xcoordinates = [x]
    ycoordinates = [y]

    for k in range(2, dx + 2):
        if p > 0:
            y = y + 1 if y < y2 else y - 1
            p = p + 2 * (dy - dx)
        else:
            p = p + 2 * dy

        x = x + 1 if x < x2 else x - 1

        xcoordinates.append(x)
        ycoordinates.append(y)

    return np.array(ycoordinates), np.array(xcoordinates)
