import sys

import matplotlib.pyplot as plt
import numpy as np

from micropolarray import MicropolImage
from micropolarray.cameras import PolarCam
from micropolarray.utils import normalize2pi


def linear_roi_from_polar(
    data: np.ndarray,
    center: list,
    theta: float,
    r: list = None,
) -> list:
    """Performs a linear roi starting from the center and extending
    to r or to the edge of the input data array.
    Angles start horizontally and rotate anti-clockwise (0deg
    corresponds to fixed y and increasing x).

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
        float: ratio between the lenght of the roi in pixels and its lenght in elements (see linear_roi.DDA). In other words, pixels/elements_n. Its inverse is number of elements per pixel.
    """
    y1, x1 = center
    y2, x2 = float(y1), float(x1)

    if r is None:
        r = [0, 1.0e18]

    theta_rad = np.deg2rad(theta)

    # keep running along theta from center until minimum radius to get start of line
    while np.sqrt((center[1] - x1) ** 2 + (center[0] - y1) ** 2) < r[0]:
        y1 = y1 - np.sin(theta_rad)
        x1 = x1 + np.cos(theta_rad)

    # keep running along theta from center until end of image is reached to get end of line
    while (
        # (y2 < data.shape[0] - 1)
        (y2 < data.shape[0])
        and (y2 > 1)
        # and (x2 < data.shape[1] - 1)
        and (x2 < data.shape[1])
        and (x2 > 1)
        and (np.sqrt((x2 - center[1]) ** 2 + (y2 - center[0]) ** 2) < r[1] - 1)
    ):
        y2 = y2 - np.sin(theta_rad)
        x2 = x2 + np.cos(theta_rad)

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


def linear_roi(data: np.ndarray, start: list, end: list) -> list:
    """Get values

    Args:
        data (np.ndarray): data on which to perform the roi
        start (list): (y, x) starting point in pixel
        end (list): (y, x) ending point in pixel

    Returns:
        list: 1d array of y coordinates, 1d array of x coordinates, values of data calculated at rois
    """
    ys, xs, density = DDA(start, end)
    # print(f"{xs, ys = }")
    # print(f"{start, end = }")

    vals = np.array([data[y, x] for y, x in zip(ys, xs)])

    return ys, xs, vals


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


if __name__ == "__main__":
    image = MicropolImage(
        "/home/herve/dottorato/antarticor/herve/campagna_2022/results/2021_12_11/corona_0/corona.fits"
    )
    # image = ml.MicropolImage(
    #    "/home/herve/dottorato/cormag/2023_flight/L1/volo fase 1/seq. 10/sum_tilted.fits"
    # )
    fig, ax = image.show_pol_param("pB")
    # plt.show()

    y, x, r = PolarCam().occulter_pos_last
    print(y, x, r)

    x = x / 2
    y = y / 2
    r = r / 2
    ax.add_artist(plt.Circle((x, y), r, fill=False, alpha=0.5))

    fig2, ax2 = plt.subplots(dpi=200)
    if True:
        # ax.axvline(0)
        # ax.axvline(image.pB.data.shape[0])
        # ax.axhline(0)
        # ax.axhline(image.pB.data.shape[1])

        for i, angle in enumerate(np.arange(-180, 180, 45)):
            result, ys, xs, ratio = linear_roi_from_polar(
                image.pB.data,
                (y, x),
                angle,
            )

            ax.plot(xs, ys, label=f"{angle:3.2f}")
            ax2.plot(range(len(result)) * ratio, result, label=f"{i:3.2f}")

        ax.legend()
        ax.add_artist(plt.Circle((x, y), 30, fill=True, alpha=1))
        plt.show()

    pixels = linear_roi(image.pB.data, (y, x), (233, 970))

    function = DDA

    ax.plot(*function((y, x), (930, 934)))
    ax.plot(*function((y, x), (930, 300)))
    ax.plot(*function((y, x), (650, 27)))
    ax.plot(*function((y, x), (271, 22)))
    ax.plot(*function((y, x), (11, 296)))
    ax.plot(*function((y, x), (11, 739)))

    plt.show()
