import os
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import constants
from scipy.optimize import curve_fit

from micropolarray.cameras import PolarCam
from micropolarray.processing.demosaic import merge_polarizations, split_polarizations


# timer decorator
def timer(func):
    """Use this to time function execution

    Args:
        func (function): function of which to measure execution time
    """

    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"Function {func.__name__} took {round(end - start, 4)} s to run")
        return result

    return wrapper


def _make_abs_and_create_dir(filename: str):
    path = Path(filename)

    if not path.is_absolute():  # suppose it is in cwd
        path = path.joinpath(Path().cwd(), path)

    if path.suffix:
        if not path.parent.exists():
            path.parent.mkdir(parents=True)
    else:
        if not path.exists():
            path.mkdir()
    return str(path)


def sigma_DN(pix_DN):
    gain = 6.93
    sigma_DN = np.sqrt(gain * pix_DN) / gain
    return sigma_DN


def fix_data(data: np.array, min, max):
    if not (min and max):
        return data
    fixed_data = data.copy()
    fixed_data = np.where(fixed_data > min, fixed_data, min)
    fixed_data = np.where(fixed_data < max, fixed_data, max)
    return fixed_data


def mean_minus_std(data: np.array, stds_n: int = 1) -> float:
    """Returns the value at the mean - standard deviation for the input data

    Args:
        data (np.array): input data
        stds_n (int, optional): number of standard deviations. Defaults to 1.

    Returns:
        float: mean value - n*stdevs
    """
    return np.mean(data) - stds_n * np.std(data)


def mean_plus_std(data: np.array, stds_n: int = 1) -> float:
    """Returns the value at the mean + standard deviation for the input data

    Args:
        data (np.array): input data
        stds_n (int, optional): number of standard deviations. Defaults to 1.

    Returns:
        float: mean value + n*stdevs
    """
    return np.mean(data) + stds_n * np.std(data)


def median_minus_std(data: np.array, stds_n: int = 1) -> float:
    """Returns the value at the median - median deviation for the input data

    Args:
        data (np.array): input data
        stds_n (int, optional): number of standard deviations. Defaults to 1.

    Returns:
        float: median value - n*mediandevs
    """
    median = np.median(data)
    median_std = np.median(np.abs(data - median))
    return median - stds_n * median_std


def median_plus_std(data: np.array, stds_n: int = 1) -> float:
    """Returns the value at the median + median deviation for the input data

    Args:
        data (np.array): input data
        stds_n (int, optional): number of standard deviations. Defaults to 1.

    Returns:
        float: median value + n*mediandevs
    """
    median = np.median(data)
    median_std = np.median(np.abs(data - median))
    return median + stds_n * median_std


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


def align_keywords_and_data(header, data, sun_center, platescale, binning=1):
    """Fixes antarticor keywords and data to reflect each other.

    Args:
        header (dict): fits file header
        data (ndarray): data as np array
        platescale (float): plate scale in arcsec/pixel
        binning (int, optional): binning applied to image. Defaults to 1 (no binning).

    Returns:
        header, data: new fixed header and data
    """

    single_pol_images = split_polarizations(data)
    # data = np.rot90(data, k=-1)
    # data = np.flip(data, axis=0)
    for i in range(4):
        single_pol_images[i] = np.rot90(single_pol_images[i], k=-1)
        single_pol_images[i] = np.flip(single_pol_images[i], axis=0)
    data = merge_polarizations(single_pol_images)

    header["NAXIS1"] = data.shape[0]
    header["NAXIS2"] = data.shape[1]
    height = header["NAXIS1"]
    width = header["NAXIS2"]
    rotation_angle = -9  # degrees
    if binning > 1:
        platescale = platescale * binning

    header["DATE-OBS"] = header["DATE-OBS"] + "T" + header["TIME-OBS"]

    header["WCSNAME"] = "helioprojective-cartesian"
    # header["DSUN_OBS"] = 1.495978707e11
    header["CTYPE1"] = "HPLN-TAN"
    header["CTYPE2"] = "HPLT-TAN"
    header["CDELT1"] = platescale
    header["CDELT2"] = platescale
    header["CUNIT1"] = "arcsec"
    header["CUNIT2"] = "arcsec"
    header["CRVAL1"] = 0
    header["CRVAL2"] = 0
    header["CROTA2"] = rotation_angle

    y, x = sun_center
    # if year == 2021:
    #    y, x, _ = PolarCam().occulter_pos_2021
    # elif year == 2022:
    #    y, x, _ = PolarCam().occulter_pos_last
    relative_y = y / 1952
    relative_x = x / 1952
    sun_x = int(width * relative_x)
    sun_y = int(height * relative_y)

    # one changes because of rotation, the other because of jhelioviewer representation
    header["CRPIX1"] = height - sun_y  # y, checked
    header["CRPIX2"] = width - sun_x  # x, checked

    return header, data


def get_Bsun_units(
    diffuser_I: float,
    texp_image: float = 1.0,
    texp_diffuser: float = 1.0,
) -> float:
    """Returns the conversion unit for expressing brightness in units of sun brightness. Usage is
    data [units of B_sun] = data[DN] * get_Bsun_units(mean_Bsun_brightness, texp_image, texp_diffuser)

    Args:
        mean_sun_brightness (float): diffuser mean in DN.
        texp_image (float, optional): image exposure time. Defaults to 1.0.
        texp_diffuser (float, optional): diffuser exposure time. Defaults to 1.0.

    Returns:
        float: Bsun units conversion factor
    """
    diffusion_solid_angle = 1.083 * 1.0e-5
    diffuser_transmittancy = 0.28
    Bsun_unit = (
        diffusion_solid_angle * diffuser_transmittancy * texp_diffuser / texp_image
    )
    Bsun_unit = (
        (1.0 / texp_image)
        * diffuser_transmittancy
        * diffusion_solid_angle
        / (diffuser_I / texp_diffuser)
    )

    return Bsun_unit


def get_malus_normalization(four_peaks_images, show_hist=False):
    S_max = np.zeros_like(four_peaks_images[0])  # tk_sum = tk_0 + tk_45 + tk_90 + tk_45
    S_max = 0.5 * np.sum(four_peaks_images, axis=0)
    # Normalizing S, has a spike of which maximum is taken
    bins = 1000
    histo = np.histogram(S_max, bins=bins)
    maxvalue = np.max(histo[0])
    index = np.where(histo[0] == maxvalue)[0][0]
    normalizing_S = (histo[1][index] + histo[1][index + 1] + histo[1][index - 1]) / 3
    # normalizing_S = np.max(S_max) # old

    # ----------------------------------------------
    # fit gaussian to S for normalization signal
    def gauss(x, norm, x_0, sigma):
        return norm * np.exp(-((x - x_0) ** 2) / (4 * sigma**2))

    hist_roi = 10  # bins around max value
    xvalues = np.array(histo[1])[index - hist_roi : index + hist_roi]
    yvalues = np.array(histo[0])[index - hist_roi : index + hist_roi]
    yvalues_sum = np.sum(yvalues)
    yvalues = yvalues / yvalues_sum
    xvalues = np.array(
        [value + (xvalues[1] - xvalues[0]) / 2 for value in xvalues]
    )  # shift each bin to center
    prediction = [
        yvalues[int(len(yvalues) / 2)],  # normalization
        xvalues[int(len(xvalues) / 2)],  # center
        xvalues[int(len(xvalues) / 2) + int(hist_roi / 2)]
        - xvalues[int(len(xvalues) / 2)],  # sigma
    ]
    params, cov = curve_fit(
        gauss,
        xvalues,
        yvalues,
        prediction,
    )
    normalizing_S = params[1] + 4 * params[2]  # center of gaussian + 2sigma
    # 3sigma -> P = 2.7e-3 outliers
    # 4sigma -> P = 6.3e-5 outliers
    # ----------------------------------------------
    if show_hist:
        index = 5
        fig, ax = plt.subplots(figsize=(9, 9))
        ax.stairs(histo[0], histo[1], label=f"S, max = {np.max(S_max)}")
        ax.axvline(normalizing_S, color="red", label="normalizing_S")
        ax.plot(
            xvalues,
            gauss(xvalues, params[0] * yvalues_sum, params[1], params[2]),
            label="Fitted curve for normalizing S",
        )
        ax.set_title(f"Normalizing S (t_0 + t_45 + t_90 + t_135)")
        ax.set_xlabel("S [DN]")
        ax.set_ylabel("Counts")
        ax.legend()
        plt.show()
        sys.exit()

    return normalizing_S
