import sys
from functools import lru_cache
from logging import info

import cv2
import matplotlib.pyplot as plt
import numpy as np
from numpy.lib.stride_tricks import as_strided
from scipy.optimize import curve_fit


def roi_from_polar(
    data: np.array,
    center: list = None,
    rho: list = None,
    theta=[0, 360],
    fill: float = 0.0,
    return_boolean: bool = False,
    include_superpixels: bool = True,
) -> np.array:
    """Returns the input array in a circular selection, otherwise an arbitrary number. If a pixel is not in the selection the ENTIRE superpixel is considered out of selection. If return_boolean is True then a boolean array is returned instead (useful for mean/stdev operations).

    Args:
        data (np.array): input data
        center (list, optional): pixel coordinates of the circle center. Defaults to None (image center).
        rho (list, optional): radius to exclude. Defaults to None (center to image border).
        theta (list, optional): polar selection angle, 0 is horizonta, anti-clockwise direction. Defaults to [0, 360].
        fill (float, optional): number to fill the outer selection. Defaults to 0.0.
        return_boolean (bool, optional): if set to true, function returns a boolean array of the roi. Defaults to False.
        include_superpixels (bool, optional): if set to true then exclude entire superpixel if one pixel is in the occulter

    Returns:
        np.array: array containing the input data inside the selection, and fill otherwise
    """
    height, width = data.shape
    theta_min, theta_max = theta
    rho_min, rho_max = rho
    if center is None:
        center = [int(height / 2), int(width / 2)]
    else:
        center = list(center)

    if rho is None:
        rho_max = np.min([height - center[0], width - center[1]])
        rho = [0.0, rho_max]

    if include_superpixels:
        # make a map that is HALF THE SIZE, do condition, then resize to make all the superpixel outside selection
        height = int(height / 2)
        width = int(width / 2)
        rho_min = int(rho_min / 2)
        rho_max = int(rho_max / 2)
        center[0] = int(center[0] / 2)
        center[1] = int(center[1] / 2)

    rho_coords, phi_coords = map_polar_coordinates(
        height,
        width,
        tuple([center[0], center[1]]),
    )  # cast it to a tuple (which is hashable)

    theta_condition = np.logical_and(phi_coords >= theta_min, phi_coords < theta_max)
    rho_condition = np.logical_and(
        rho_coords > rho_min, rho_coords <= rho_max
    )  # half the radius because half the map
    condition = np.logical_and(rho_condition, theta_condition)

    if include_superpixels:
        # resize condition to correct shape
        condition = condition.repeat(2, axis=0).repeat(2, axis=1)

    if return_boolean:
        return np.where(condition, True, False)
    else:
        return np.where(condition, data, fill)


def tile_double(a):
    height, width = a.shape
    hs, ws = a.strides
    tiles = as_strided(a, (height, 2, width, 2), (hs, 0, ws, 0))
    return tiles.reshape(2 * height, 2 * width)


@lru_cache
def map_polar_coordinates(height, width, center):
    y_center, x_center = center

    i_list, j_list = np.arange(width), np.arange(height)
    x_coords, y_coords = np.meshgrid(i_list, j_list)
    # Map polar coordinates, 0 = horizontal dx, anti-clockwise angles
    rho_coords = np.sqrt((x_coords - x_center) ** 2 + (y_coords - y_center) ** 2)
    phi_coords = (
        (np.arctan2(y_coords - y_center, x_coords - x_center) * 180 / np.pi) + 360
    ) % 360

    return rho_coords, phi_coords


def nrgf(
    data: np.array,
    y_center: int = None,
    x_center: int = None,
    rho_min: int = None,
    step: int = 1,
    phi_to_mean=[0.0, 360],
    output_phi=[0.0, 360],
) -> np.array:
    """
    Performs nrgf filtering on the image, starting from center and radius. Mean is performed between phi_to_mean, 0 is horizontal right, anti-clockwise.


    Args:
        data (np.array): input array
        y_center (int, optional): pixel y coordinate of the nrgf center. Defaults to None (image y center).
        x_center (int, optional): pixel x coordinate of the nrgf center. Defaults to (image x center).
        rho_min (int, optional): minimun radius in pixels to perform nrgf to. Defaults to None (radius 0).
        step (int, optional): step to which apply the nrgf from center, in pixels. Defaults to 1 pixel.
        phi_to_mean (list[float, float], optional): polar angle to calculate the mean value from. Defaults to [0, 360].
        output_phi (list[float, float], optional): polar angle to include in output data. Defaults to [0, 360].

    Returns:
        np.array: nrgf-filtered input data
    """
    height, width = data.shape

    if (y_center is None) or (x_center is None) or (rho_min is None):
        info("Calculating occulter position...")
        y_center, x_center, rho_min = find_occulter_position(data)
    center = [int(y_center), int(x_center)]

    newdata = np.zeros(shape=data.shape, dtype=float)
    i_list, j_list = np.arange(width), np.arange(height)
    x_coords, y_coords = np.meshgrid(i_list, j_list)
    # Map polar coordinates
    rho_coords, phi_coords = map_polar_coordinates(
        height, width, tuple(center)
    )  # cast it to a tuple (which is hashable)

    mean_phi_condition = np.logical_and(
        phi_coords >= phi_to_mean[0], phi_coords < phi_to_mean[1]
    )  # Exclude angle from mean

    out_phi_condition = np.logical_and(
        phi_coords >= output_phi[0], phi_coords < output_phi[1]
    )  # Exclude angle from filter

    rho_max = int(np.max(rho_coords))
    rho_step = step

    print("Applying nrgf filter...")
    for r in range(rho_min, rho_max, rho_step):
        rho_condition = np.logical_and(rho_coords > r, rho_coords <= r + rho_step)
        condition = np.logical_and(rho_condition, out_phi_condition)
        # condition = np.logical_and(rho_condition, mean_phi_condition)
        mean_condition = np.logical_and(rho_condition, mean_phi_condition)
        mean_over_ROI = np.mean(data, where=mean_condition)
        std_over_ROI = np.std(data, where=mean_condition)
        if std_over_ROI > 0:
            newdata = np.where(
                condition,
                (data - mean_over_ROI) / std_over_ROI,
                newdata,
            )
        else:
            newdata = np.where(condition, 0, newdata)

    return newdata


def find_occulter_hough(
    data: np.array, minr: float = 1, maxr: float = None, **kwargs
) -> tuple:
    """Uses Hough Gradient from cv2 and computes the coronagraph occulter coordinates. Returns Y, X, Radius of the occulter.

    Args:
        data (np.array): input data

    Returns:
        tuple: occulter y, x, r
    """
    info("Searching occulter, this may take a while...")
    if maxr is None:
        maxr = np.max(data.shape)
    minDist = np.max(data.shape)  # minimum distance between circles

    data = 255 * data / np.max(data)
    blurred = cv2.medianBlur(data.astype("uint8"), 5)
    accumulator = 10
    circles = cv2.HoughCircles(
        image=blurred,
        method=cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=minDist,
        param1=200,
        param2=accumulator,
        minRadius=minr,
        maxRadius=maxr,
    )
    try:
        while len(circles[0]) > 1:
            print(f"{len(circles[0])} circles found, retrying...")
            accumulator += 5
            circles = cv2.HoughCircles(
                image=blurred,
                method=cv2.HOUGH_GRADIENT,
                dp=1.2,
                minDist=minDist,
                param1=200,
                param2=accumulator,
                minRadius=minr,
                maxRadius=maxr,
            )
    except TypeError:
        print("Failed to find the occulter.")
    x, y, r = circles[0, 0]
    return y, x, r  # BEWARE OF THIS REVERSION: Y THEN X


def find_occulter_position(
    data: np.array, method: str = "sigmoid", threshold: float = 4.0
):
    """Finds the occulter position of an image.

    Args:
        data (np.array): input data
        method (str, optional): Method to find occulter edges. If "sigmoid" it will try to fit four sigmoids at the image edges centers, inferring the occulter edges from the parameters. If "algo" it will start from the image edge center and infer the occulter position when DN[i] > threshold*mean(DN[:i]) Defaults to "sigmoid".
        threshold (float, optional): Threshold for the algo method. Defaults to 4.0.

    Raises:
        UnboundLocalError: couldn't converge

    Returns:
        list: occulter y, occulter x, occulter radius
    """
    # works if occulter is not entirely inside a single quadrant, fits
    # a sigmoid to find occulter bounds
    half_y = int(data.shape[0] / 2)
    half_x = int(data.shape[1] / 2)
    values_x_0 = np.flip(data[half_y, :half_x])
    values_x_1 = data[half_y, half_x:]
    values_y_0 = np.flip(data[:half_y, half_x])
    values_y_1 = data[half_y:, half_x]

    occulter_bounds = [0.0] * 4

    show = False
    if show:
        fig, ax = plt.subplots(2, 2, constrained_layout=True)
        ax = ax.ravel()
        titles = [
            "Lower vertical",
            "Upper vertical",
            "Right horizontal",
            "Left horizontal",
        ]

    if method == "sigmoid":
        for idx, half_array in enumerate(
            [values_y_0, values_y_1, values_x_0, values_x_1]
        ):
            # Artificial plateau after maximum
            # max_half_array = np.max(half_array)
            # for i, element in enumerate(half_array):
            #    if element == max_half_array:
            #        max_i = i
            #        break
            # half_array[max_i:] = max_half_array

            # x = np.arange(0, half_y, 1)
            x = np.arange(0, int(len(half_array / 2)), 1)

            params = [
                # np.mean(half_array[int(len(half_array) / 2) :]),
                np.max(half_array),
                # np.mean(half_array[: int(len(half_array) / 2)]),
                np.min(half_array),
                0.5,
                len(half_array) / 2,
            ]
            params, pcov = curve_fit(sigmoid, x, half_array, params)
            occulter_bounds[idx] = params[3]

            if show:
                axis = ax[idx]
                axis.scatter(x, half_array, c="grey")
                axis.plot(x, sigmoid(x, *params), c="black")
                axis.set_title(titles[idx])
                axis.set_xlabel("Pixels from center")
                axis.set_ylabel("DN")

        try:
            occulter_bounds[0] = half_y - occulter_bounds[0]
            occulter_bounds[1] = half_y + occulter_bounds[1]
            occulter_bounds[2] = half_x - occulter_bounds[2]
            occulter_bounds[3] = half_x + occulter_bounds[3]
        except UnboundLocalError:
            raise ValueError("Edges not found, try to change the threshold")

    elif method == "algo":
        # OLD, ALGORITM INSTEAD OF FITTING - less stable
        min_points = 5
        threshold = threshold

        for i, val in enumerate(values_x_0):
            if i > min_points:
                mean = np.mean(values_x_0[:i])
                if val > (threshold * mean):
                    occulter_start_x = half_x - i
                    break
        for i, val in enumerate(values_x_1):
            if i > min_points:
                mean = np.mean(values_x_1[:i])
                if val > (threshold * mean):
                    occulter_end_x = half_x + i
                    break
        for i, val in enumerate(values_y_0):
            if i > min_points:
                mean = np.mean(values_y_0[:i])
                if val > (threshold * mean):
                    occulter_start_y = half_y - i
                    break
        for i, val in enumerate(values_y_1):
            if i > min_points:
                mean = np.mean(values_y_1[:i])
                if val > (threshold * mean):
                    occulter_end_y = half_y + i
                    break
        try:
            occulter_bounds[0] = occulter_start_y
            occulter_bounds[1] = occulter_end_y
            occulter_bounds[2] = occulter_start_x
            occulter_bounds[3] = occulter_end_x
        except UnboundLocalError:
            raise ValueError(
                "ERROR: occulter edges not found, try to change the threshold"
            )
    elif method == "hugh":
        pass  # TODO
    if show:
        plt.show()

    y_center = int(np.mean([occulter_bounds[:2]]))
    x_center = int(np.mean([occulter_bounds[2:]]))
    radius = int(
        np.mean(
            [
                np.abs(occulter_bounds[1] - occulter_bounds[0]) / 2,
                np.abs(occulter_bounds[3] - occulter_bounds[2]) / 2,
            ]
        )
    )

    return [y_center, x_center, radius]


def sigmoid(x, max, min, slope, intercept):
    sigma = slope * (x - intercept)
    return max * np.exp(sigma) / (1 + np.exp(sigma)) + min
