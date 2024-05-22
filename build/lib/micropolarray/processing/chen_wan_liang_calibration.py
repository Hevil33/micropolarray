import sys
from logging import info, warning
from pathlib import Path

import numpy as np
from astropy.io import fits
from numba import njit
from tqdm import tqdm

from micropolarray.cameras import PolarCam
from micropolarray.processing.nrgf import roi_from_polar
from micropolarray.utils import _make_abs_and_create_dir


def calculate_chen_wan_lian_calibration(
    polarizer_orientations,
    filenames_list,
    micropol_phases_previsions,
    output_dir,
    occulter=True,
    dark_filename=None,
    flat_filename=None,
):
    """Performs calibration from Chen-Wang-Liang paper 2014

    Args:
        polarizer_orientations (list[float]): List of polarizer orienataions
        filenames_list (list[str]): List of filenames coupled with
        micropol_phases_previsions (list[float]): Previsions of the micropolarizer orientations inside a superpixel
        output_dir (str): output path for the calibration matrices
        occulter (bool, optional): wether to exclude the occulter area. Defaults to True.
        dark_filename (str, optional): path to the dark to be subtracted from the images. Defaults to None.
        flat_filename (str, optional): path to the dark to be subtracted from the images. Defaults to None.

    Raises:
        ValueError: polarizer orientation list and filenames list do not have the same lenght
        ValueError: any of 0,45,90,-45 polarizarions is not included in the polarizer orientation list
    """

    DEBUG = False

    if type(filenames_list) is not list:
        filenames_list = [filenames_list]

    pols_n = len(polarizer_orientations)
    filenames_n = len(filenames_list)
    if pols_n != filenames_n:
        raise ValueError("Polarizer orientations do not match filenames.")

    if not np.all(np.isin([0, 45, 90, -45], polarizer_orientations)):
        raise ValueError(
            "All (0, 45, 90, -45) pols must be included in the polarizer orientation array"
        )
    # Have to be sorted
    polarizer_orientations, filenames_list = (
        list(t)
        for t in zip(*sorted(zip(polarizer_orientations, filenames_list)))
    )

    angle_dic = {micropol_phases_previsions[i]: i for i in range(4)}

    with fits.open(filenames_list[0]) as file:
        data = file[0].data  # get data dimension
    occulter_flag = np.ones_like(data)  # 0 if not a occulted px, 1 otherwise
    if occulter:
        # Mean values from coronal observations 2022_12_03
        # (campagna_2022/mean_occulter_pos.py)

        occulter_y, occulter_x, occulter_r = PolarCam().occulter_pos_nov2022
        occulter_r += 10  # Better to overoccult

        occulter_flag = roi_from_polar(
            occulter_flag, [occulter_y, occulter_x], [0, occulter_r]
        )
        for super_y in range(0, occulter_flag.shape[0], 2):
            for super_x in range(0, occulter_flag.shape[1], 2):
                if np.any(
                    occulter_flag[super_y : super_y + 2, super_x : super_x + 2]
                ):
                    occulter_flag[
                        super_y : super_y + 2, super_x : super_x + 2
                    ] = 1
                    continue
    else:
        occulter_flag *= 0

    # Collect dark
    if dark_filename:
        with fits.open(dark_filename) as file:
            dark = file[0].data
    # Collect flat field, normalize it
    if flat_filename:
        with fits.open(flat_filename) as file:
            flat = file[0].data
    if flat_filename and dark_filename:
        flat -= dark  # correct flat too
        flat = np.where(flat > 0, flat, 1.0)
        if occulter:
            flat = np.where(occulter_flag, 1.0, flat)

    flat_max = np.max(flat, axis=(0, 1))
    normalized_flat = np.where(occulter_flag, 1.0, flat / flat_max)

    all_data_arr = [0.0] * filenames_n
    for idx, filename in enumerate(filenames_list):
        with fits.open(filename) as file:
            all_data_arr[idx] = file[0].data
            if dark_filename is not None:
                all_data_arr[idx] -= dark
                all_data_arr[idx] = np.where(
                    all_data_arr[idx] >= 0, all_data_arr[idx], 0.0
                )
            if flat_filename is not None:
                all_data_arr[idx] = np.where(
                    normalized_flat != 0,
                    all_data_arr[idx] / normalized_flat,
                    all_data_arr[idx],
                )

    all_data_arr = np.array(all_data_arr)
    _, height, width = all_data_arr.shape

    if DEBUG:
        warning("Running in DEBUG mode")
        height = int(height / 2)
        width = int(width / 2)
        all_data_arr = all_data_arr[:, 0:height, 0:width]

    # Stokes input vector: S = [pol] * [S_0, S_1, S_2, 1] * [width, height]
    S_input = np.zeros(shape=(filenames_n, 4, height, width))

    single_pol_images = np.array(
        [
            [
                all_data_arr[pol_idx, y::2, x::2]
                for y in range(2)
                for x in range(2)
            ]
            for pol_idx in range(pols_n)
        ]
    )

    sorted_single_pol_images = np.array(single_pol_images, copy=True)
    sorted_single_pol_images[:, 0, :, :] = single_pol_images[
        :, angle_dic[0], :, :
    ]
    sorted_single_pol_images[:, 1, :, :] = single_pol_images[
        :, angle_dic[45], :, :
    ]
    sorted_single_pol_images[:, 2, :, :] = single_pol_images[
        :, angle_dic[-45], :, :
    ]
    sorted_single_pol_images[:, 3, :, :] = single_pol_images[
        :, angle_dic[90], :, :
    ]

    _build_S_input(sorted_single_pol_images, S_input)  # speeds up A LOT

    W_ideal = np.array(
        [
            [0.5, 0.5, 0.5, 0.5],
            [
                0.5 * np.cos(2.0 * np.deg2rad(micropol_phases_previsions[i]))
                for i in range(4)
            ],
            [
                0.5 * np.sin(2.0 * np.deg2rad(micropol_phases_previsions[i]))
                for i in range(4)
            ],
        ]
    ).T  # shape=(4,3)

    C_ij = np.zeros(shape=(pols_n, 4, 4, height, width))  # correction matrix
    d_ij = np.zeros(
        shape=(pols_n, height, width)
    )  # offsets, each element is equal in the superpixel so we can save one image with 4 pixel to save space

    if DEBUG:
        height = 8
        width = 8

    for pol_idx in tqdm(range(pols_n)):
        for super_y in tqdm(range(0, height, 2)):
            for super_x in range(0, width, 2):
                S = np.array([S_input[pol_idx, :, super_y, super_x]]).T

                S_inv_ij = np.linalg.pinv(S)
                P_ij = all_data_arr[
                    pol_idx, super_y : super_y + 2, super_x : super_x + 2
                ].reshape(4, 1)

                W_d = P_ij @ S_inv_ij
                # W_d = W_d.T
                W = W_d[:, :3]
                d = W_d[:, 3]
                try:
                    W_inv = np.linalg.pinv(W)
                except np.linalg.LinAlgError:
                    W_inv = 0.0 * W.T

                d_ij[
                    pol_idx, super_y : super_y + 2, super_x : super_x + 2
                ] = d.reshape(2, 2)

                C = W_ideal @ W_inv
                for i in range(2):
                    for j in range(2):
                        C_ij[pol_idx, :, :, super_y + i, super_x + j] = C

    C_ij = np.mean(C_ij, axis=0)
    d_ij = np.mean(d_ij, axis=0)

    output_dir = _make_abs_and_create_dir(output_dir)
    for i in range(4):
        for j in range(4):
            hdu = fits.PrimaryHDU(data=C_ij[i, j])
            hdu.writeto(output_dir + f"/C{i}{j}.fits", overwrite=True)
    hdu = fits.PrimaryHDU(data=d_ij)
    hdu.writeto(output_dir + "/d_ij.fits", overwrite=True)


@njit
def _build_S_input(single_pol_images, S_input):
    # ANGLES_LIST positions:
    #    0 <==> 0
    #    1 <==> 45
    #    2 <==> -45
    #    3 <==> 90
    pols_n = single_pol_images.shape[0]
    height = single_pol_images.shape[2] * 2
    width = single_pol_images.shape[3] * 2

    S_input[:, -1, :, :] += 1
    # Initialize stokes vector
    for pol_idx in range(pols_n):
        for super_y in range(0, height, 2):
            y = int(super_y / 2)
            for super_x in range(0, width, 2):
                x = int(super_x / 2)
                S_input[
                    pol_idx, 0, super_y : super_y + 2, super_x : super_x + 2
                ] = (
                    single_pol_images[pol_idx, 0, y, x]
                    + single_pol_images[pol_idx, 3, y, x]
                )
                S_input[
                    pol_idx, 1, super_y : super_y + 2, super_x : super_x + 2
                ] = (
                    single_pol_images[pol_idx, 0, y, x]
                    - single_pol_images[pol_idx, 3, y, x]
                )
                S_input[
                    pol_idx, 2, super_y : super_y + 2, super_x : super_x + 2
                ] = (
                    single_pol_images[pol_idx, 1, y, x]
                    - single_pol_images[pol_idx, 2, y, x]
                )


def chen_wan_liang_calibration(data, calibration_matrices_dir: str):
    """Calibrates the images using Chen-Wang-Liang 2014 paper calibration

    Args:
        data (np.array): data to be calibrated
        calibration_matrices_dir (str): path to the calibration matrices

    Returns:
        np.array: calibrated data
    """
    info("Applying Chen-Wan-Liang correction...")
    height, width = data.shape
    C_ij = np.zeros(shape=(4, 4, height, width))
    d_ij = np.zeros_like(data)
    corrected_data = np.zeros_like(data)
    ifov_corrected_data = np.zeros_like(data)
    correction_path = Path(calibration_matrices_dir)
    if not correction_path.is_absolute():  # suppose it is in cwd
        correction_path = correction_path.joinpath(
            Path().cwd(), correction_path
        )

    for i in range(4):
        for j in range(4):
            with fits.open(
                calibration_matrices_dir + f"/C{i}{j}.fits"
            ) as file:
                C_ij[i, j] = file[0].data
    with fits.open(calibration_matrices_dir + f"/d_ij.fits") as file:
        d_ij = file[0].data

    P_minus_d = data - d_ij

    for super_y in range(0, height, 2):
        for super_x in range(0, width, 2):
            mult = C_ij[:, :, super_y, super_x] @ P_minus_d[
                super_y : super_y + 2, super_x : super_x + 2
            ].reshape(4, 1)

            corrected_data[
                super_y : super_y + 2, super_x : super_x + 2
            ] = mult.reshape(2, 2)

    ifov_corrected_data = _ifov_jitcorrect(corrected_data, height, width)

    return ifov_corrected_data


@njit
def _ifov_jitcorrect(data, height, width):
    corrected_data = np.zeros(shape=(height, width))
    # correct IFOV error, skip first rows/cols
    for super_y in range(2, height, 2):
        for super_x in range(2, width, 2):
            for i in range(2):
                for j in range(2):
                    corrected_data[super_y + j, super_x + i] = 0.25 * (
                        data[super_y + j - 2, super_x + i - 2]
                        + data[super_y + j - 2, super_x + i]
                        + data[super_y + j, super_x + i - 2]
                        + data[super_y + j, super_x + i]
                    )
    return corrected_data
