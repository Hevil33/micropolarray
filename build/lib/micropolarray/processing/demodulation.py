from micropolarray.processing.rebin import (
    standard_jitrebin,
    micropolarray_jitrebin,
    trim_to_match_binning,
)
from micropolarray.processing.nrgf import (
    find_occulter_position,
    roi_from_polar,
)

from scipy.optimize import curve_fit
import glob
import os
from astropy.io import fits
import re
import numpy as np
import sys
import matplotlib.pyplot as plt
from tqdm import tqdm
import multiprocessing as mp
from itertools import product
import time
from logging import warning, info, error


class Demodulator:
    """Demodulation class needed for MicroPolarizerArrayImage
    demodulation."""

    def __init__(self, demo_matrices_path: str, binning: bool = False):
        self.demo_matrices_path = demo_matrices_path
        self.binning = binning  # needed for a correct image binning
        self.mij = self._get_demodulation_tensor(demo_matrices_path)

    def rebin(self, binning: int):
        warning("USE OF BINNED DEMODULATION MATRIX IS INCORRECT")
        binned_demodulator = Demodulator(self.demo_matrices_path)
        end_y, end_x = self.mij[0, 0].shape
        end_y, end_x = trim_to_match_binning(end_y, end_x, binning)
        new_mij = np.zeros(
            shape=(
                self.mij.shape[0],
                self.mij.shape[1],
                int(self.mij.shape[2] / binning),
                int(self.mij.shape[3] / binning),
            )
        )
        for i in range(3):
            for j in range(4):
                single_mij = self.mij[i, j, :end_y, :end_x]
                new_mij[i, j] = standard_jitrebin(
                    single_mij,
                    single_mij.shape[0],
                    single_mij.shape[1],
                    binning=binning,
                ) / (binning * binning)
                # new_mij[i, j] = micropolarray_jitrebin(
                #    single_mij,
                #    single_mij.shape[0],
                #    single_mij.shape[1],
                #    binning=binning,
                # )
        binned_demodulator.mij = new_mij
        return binned_demodulator

    def _get_demodulation_tensor(self, binning: bool = False) -> np.ndarray:
        if not os.path.exists(self.demo_matrices_path):
            raise FileNotFoundError("self.demo_matrices_path not found.")
        mij_filenames_list = glob.glob(
            self.demo_matrices_path + os.path.sep + "*.fits"
        )

        with fits.open(mij_filenames_list[0]) as firsthul:
            sample_data = np.array(firsthul[0].data)
        Mij = np.zeros(
            shape=(3, 4, sample_data.shape[0], sample_data.shape[1]),
            dtype=float,
        )
        for mij in mij_filenames_list:
            searchresult = re.search("[Mm][0-9]{2}", mij.split("/")[-1])
            if searchresult is not None:  # Exclude files not matching m/Mij
                i, j = re.search("[Mm][0-9]{2}", mij.split("/")[-1]).group()[
                    -2:
                ]  # Searches for pattern M/m + ij as last string before .fits
            else:
                continue
            i = int(i)
            j = int(j)
            with fits.open(mij) as hul:
                Mij[i, j] = hul[0].data

        return Mij


def calculate_demodulation_tensor(
    polarizer_orientations,
    filenames_list,
    micropol_phases_previsions,
    gain,  #  needed for errors
    output_dir,
    occulter=True,
    parallelize=True,
    dark_filename=None,
    flat_filename=None,
):
    # polarizations = array of polarizer orientations
    # filenames_list = list of filenames
    firstcall = True
    if not np.all(np.isin([0, 45, 90, -45], polarizer_orientations)):
        raise ValueError(
            "All (0, 45, 90, -45) pols must be included in the polarizer orientation array"
        )
    # Have to be sorted
    polarizer_orientations, filenames_list = (
        list(t)
        for t in zip(*sorted(zip(polarizer_orientations, filenames_list)))
    )
    micropol_phases_previsions = np.array(micropol_phases_previsions)
    rad_micropol_phases_previsions = np.deg2rad(micropol_phases_previsions)

    # Flag occulter position to not be fitted, expand to superpixel.
    with fits.open(filenames_list[0]) as file:
        data = file[0].data  # get data dimension
    occulter_flag = np.ones_like(data)  # 0 if not a occulted px, 1 otherwise
    if occulter:
        # Mean values from coronal observations 2022_12_03
        # (campagna_2022/mean_occulter_pos.py)
        occulter_y = 917
        occulter_x = 948
        occulter_r = 524 + 10  # Better to overoccult

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
            dark = np.array(file[0].data, dtype=np.float)
    # Collect flat field, normalize it
    if flat_filename:
        with fits.open(flat_filename) as file:
            flat = np.array(file[0].data, dtype=np.float)
    if flat_filename and dark_filename:
        flat -= dark  # correct flat too
        flat = np.where(flat > 0, flat, 1.0)
        if occulter:
            flat = np.where(occulter_flag, 1.0, flat)

    flat_max = np.max(flat, axis=(0, 1))
    normalized_flat = np.where(occulter_flag, 1.0, flat / flat_max)

    # collect data
    all_data_arr = [0.0] * len(filenames_list)
    for idx, filename in enumerate(filenames_list):
        with fits.open(filename) as file:
            all_data_arr[idx] = np.array(file[0].data, dtype=float)
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

    DEBUG = False
    if DEBUG:
        height = int(height / 2)
        width = int(width / 2)
        all_data_arr = all_data_arr[:, 0:height, 0:width]
        parallelize = False

    # parallelize
    chunks_n = 4  # Will be divided into chunks_n*chunks_n squares
    chunk_size_x = int(width / chunks_n)
    chunk_size_y = int(height / chunks_n)
    splitted_data = np.zeros(
        shape=(
            chunks_n * chunks_n,
            len(polarizer_orientations),
            chunk_size_y,
            chunk_size_x,
        )
    )
    splitted_occulter = np.zeros(
        shape=(chunks_n * chunks_n, chunk_size_y, chunk_size_x)
    )
    for i in range(chunks_n):
        for j in range(chunks_n):
            splitted_data[i + chunks_n * j] = np.array(
                all_data_arr[
                    :,
                    i * (chunk_size_x) : (i + 1) * chunk_size_x,
                    j * (chunk_size_y) : (j + 1) * chunk_size_y,
                ]
            )  # shape = (chunks_n*chunks_n, len(filenames_list), chunk_size_y, chunk_size_x)
            splitted_occulter[i + j * chunks_n] = np.array(
                occulter_flag[
                    i * (chunk_size_x) : (i + 1) * chunk_size_x,
                    j * (chunk_size_y) : (j + 1) * chunk_size_y,
                ]
            )  # shape = (chunks_n*chunks_n, chunk_size_y, chunk_size_x)

    # Normalizing S, has a spike of which maximum is taken
    S_max = np.zeros(
        shape=(height, width)
    )  # tk_sum = tk_0 + tk_45 + tk_90 + tk_45
    for pol, image in zip(polarizer_orientations, all_data_arr):
        if pol in [0, 90, 45, -45]:
            S_max += 0.5 * image
    bins = 1000
    histo = np.histogram(S_max, bins=bins)
    index = np.where(histo[0] == np.max(histo[0]))[0][0]
    normalizing_S = 0.5 * (histo[1][index] + histo[1][index + 1])

    if False:
        histo_0 = np.histogram(all_data_arr[5], bins=1000)
        fig, ax = plt.subplots()
        ax.stairs(histo_0[0], histo_0[1])
        ax.stairs(histo[0], histo[1])
        ax.vlines(normalizing_S, ymin=0, ymax=100000, colors="red")
        plt.show()
        sys.exit()

    args = (
        [
            splitted_data[i],
            normalizing_S,
            splitted_occulter[i],
            polarizer_orientations,
            rad_micropol_phases_previsions,
            gain,
        ]
        for i in range(chunks_n * chunks_n)
    )

    starting_time = time.perf_counter()
    loc_time = time.strftime("%H:%M:%S  (%Y/%m/%d)", time.localtime())
    info(f"Starting parallel calculation")

    if parallelize:
        try:
            with mp.Pool(processes=chunks_n * chunks_n) as p:
                result = p.starmap(
                    compute_demodulation_by_chunk,
                    args,
                )
        except:
            error("Fit not found")
            ending_time = time.perf_counter()

            info(f"Elapsed : {(ending_time - starting_time)/60:3.2f} mins")
            sys.exit()
    else:
        arglist = [arg for arg in args]
        result = [[0.0, 0.0]] * chunks_n * chunks_n

        for i in range(chunks_n * chunks_n):
            result[i] = compute_demodulation_by_chunk(*arglist[i])

    loc_time = time.strftime("%H:%M:%S (%Y/%m/%d)", time.localtime())
    info(f"Ending parallel calculation")

    ending_time = time.perf_counter()
    info(f"Elapsed : {(ending_time - starting_time)/60:3.2f} mins")

    result = np.array(result)
    m_ij = np.zeros(shape=(3, 4, height, width))
    tks = np.zeros(shape=(height, width))
    efficiences = np.zeros(shape=(height, width))
    phases = np.zeros(shape=(height, width))

    for i in range(chunks_n):
        for j in range(chunks_n):
            m_ij[
                :,
                :,
                i * (chunk_size_x) : (i + 1) * chunk_size_x,
                j * (chunk_size_y) : (j + 1) * chunk_size_y,
            ] = result[i + chunks_n * j, 0].reshape(
                3, 4, chunk_size_y, chunk_size_x
            )
            tks[
                i * (chunk_size_x) : (i + 1) * chunk_size_x,
                j * (chunk_size_y) : (j + 1) * chunk_size_y,
            ] = result[i + chunks_n * j, 1].reshape(chunk_size_y, chunk_size_x)
            efficiences[
                i * (chunk_size_x) : (i + 1) * chunk_size_x,
                j * (chunk_size_y) : (j + 1) * chunk_size_y,
            ] = result[i + chunks_n * j, 2].reshape(chunk_size_y, chunk_size_x)
            phases[
                i * (chunk_size_x) : (i + 1) * chunk_size_x,
                j * (chunk_size_y) : (j + 1) * chunk_size_y,
            ] = result[i + chunks_n * j, 3].reshape(chunk_size_y, chunk_size_x)

    phases = np.deg2rad(phases)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for i in range(3):
        for j in range(4):
            hdu = fits.PrimaryHDU(data=m_ij[i, j])
            hdu.writeto(output_dir + f"M{i}{j}.fits", overwrite=True)
    hdu = fits.PrimaryHDU(data=tks)
    hdu.writeto(output_dir + "transmittancies.fits", overwrite=True)
    hdu = fits.PrimaryHDU(data=efficiences)
    hdu.writeto(output_dir + "efficiences.fits", overwrite=True)
    hdu = fits.PrimaryHDU(data=phases)
    hdu.writeto(output_dir + "phases.fits", overwrite=True)

    info("Demodulation matrices and fit data successfully saved!")

    return


"""
    else:

        # Preemptly compute the theoretical demo matrix to save time
        theo_modulation_matrix = np.array(
            [
                [0.5, 0.5, 0.5, 0.5],
                [
                    0.5 * np.cos(2.0 * rad_micropol_phases_previsions[i])
                    for i in range(4)
                ],
                [
                    0.5 * np.sin(2.0 * rad_micropol_phases_previsions[i])
                    for i in range(4)
                ],
            ]
        )
        theo_demodulation_matrix = np.linalg.pinv(theo_modulation_matrix)

        # Save sum of throughputs for later loop
        S_max = np.zeros(
            shape=(height, width)
        )  # tk_sum = tk_0 + tk_45 + tk_90 + tk_45
        for pol, image in zip(polarizer_orientations, all_data_arr):
            if pol in [0, 90, 45, -45]:
                S_max += 0.5 * image
        max_S_all_pixls = np.max(S_max)

        m_ij = np.zeros(shape=(4, 3, height, width))
        modulation_matrix = np.zeros(shape=(3, 4))
        superpix_params = np.zeros(shape=(4, 3))

        predictions = np.zeros(shape=(4, 3))
        predictions[:, 1] = 0.85  # Efficiency prediction
        predictions[:, 2] = rad_micropol_phases_previsions  # Angle prediction

        bounds = np.zeros(shape=(4, 2, 3))
        bounds[:, 0, 1], bounds[:, 1, 1] = 0.1, 1.0  # Efficiency bounds
        bounds[:, 0, 2] = (
            rad_micropol_phases_previsions - 15
        )  # Lower angle bounds
        bounds[:, 1, 2] = (
            rad_micropol_phases_previsions + 15
        )  # Upper angle bounds

        starting_time = time.perf_counter()
        # fit for each superpixel
        for super_y in tqdm(range(0, height, 2)):
            for super_x in range(0, width, 2):
                # for super_y in range(0, 4, 2):
                #    for super_x in range(0, 4, 2):
                if not (
                    np.any(
                        occulter_flag[
                            super_y : super_y + 2, super_x : super_x + 2
                        ]
                    )
                ):
                    S_superpix_arr = S_max[
                        super_y : super_y + 2, super_x : super_x + 2
                    ].reshape(4)
                    superpix_arr = all_data_arr[
                        :, super_y : super_y + 2, super_x : super_x + 2
                    ].reshape(len(filenames_list), 4)
                    # dn to el
                    # sqrt
                    # convert to dn
                    sigma = (np.sqrt(superpix_arr[:])) / S_superpix_arr

                    # max_S = S_superpix_arr  # 4 max S of superpixel
                    max_S = max_S_all_pixls  # Maximmum S over all image
                    # max_S = np.max(superpix_arr, axis=1)  # Maximum over a superpixel

                    if False:  # Debug
                        print(f"{max_S.shape = }")

                        tk = np.max(superpix_arr, axis=1) / max_S
                        print(f"{tk.shape = }")

                        print(f"{all_data_arr.shape = }")
                        print(f"{all_data_arr[0:3,0:2,0:2] = }")
                        print(f"{superpix_arr = }")
                        print(f"{superpix_arr = }")
                        print(f"{max_S = }")
                    superpix_arr /= max_S
                    # print(f"{all_data_arr.shape[1:3] = }")

                    if False:  # Liberatore method, lock throughput
                        predictions[:, 0] = tk
                        bounds[:, 0, 0] = tk - 1.0e-10
                        bounds[:, 1, 0] = tk + 1.0e-10
                    else:  # Fit throughput too
                        predictions[:, 0] = [0.85] * 4
                        bounds[:, 0, 0] = 0.1
                        bounds[:, 1, 0] = 1.0

                    for pixel_num in range(4):
                        try:
                            (
                                superpix_params[pixel_num],
                                superpix_pcov,
                            ) = curve_fit(
                                Malus,
                                polarizations_rad,
                                superpix_arr[:, pixel_num],
                                predictions[pixel_num],
                                sigma=sigma[:, pixel_num],
                                bounds=bounds[pixel_num],
                            )
                        except RuntimeError:
                            print("Fit not found")
                            ending_time = time.perf_counter()
                            print(f"{pixel_num = }")
                            print(f"{super_x = }")
                            print(f"{super_y = }")

                            print(
                                f"Elapsed : {(ending_time - starting_time)/60:3.2f} mins"
                            )
                            sys.exit()

                    # Compute modulation matrix and its inverse
                    t = superpix_params[:, 0]
                    eff = superpix_params[:, 1]
                    phi = superpix_params[:, 2]
                    modulation_matrix = np.array(
                        [
                            0.5 * t,
                            0.5 * t * eff * np.cos(2.0 * phi),
                            0.5 * t * eff * np.sin(2.0 * phi),
                        ]
                    )
                    demodulation_matrix = np.linalg.pinv(modulation_matrix)

                    for i in range(2):
                        for j in range(2):
                            m_ij[
                                :, :, super_y + i, super_x + j
                            ] = demodulation_matrix

                    if False:  # Debug
                        fig, ax = plt.subplots(figsize=(9, 9))
                        for i in range(4):
                            ax.scatter(
                                np.rad2deg(polarizations_rad),
                                superpix_arr[:, i],
                                # yerr=sigma[:, i],
                                label=f"points {i}",
                            )
                            min = np.min(polarizations_rad)
                            max = np.max(polarizations_rad)
                            x = np.arange(min, max, (max - min) / 100)
                            x = np.arange(-np.pi / 2, np.pi, np.pi / 100)
                            ax.plot(
                                np.rad2deg(x),
                                Malus(x, *superpix_params[i]),
                                label=f"t = {superpix_params[i,0]:2.2f}, e = {superpix_params[i, 1]:2.2f}, phi = {np.rad2deg(superpix_params[i, 2]):2.2f}",
                            )
                        plt.legend()
                        plt.show()

                        # print(m_ij)
                        print(f"{t = }")
                        print(f"{eff = }")
                        print(f"phi = {np.rad2deg(phi)}")
                        print(f"{modulation_matrix = }")
                        print(f"{demodulation_matrix = }")
                        sys.exit()
                else:
                    for i in range(2):
                        for j in range(2):
                            m_ij[
                                :, :, super_y + i, super_x + j
                            ] = theo_demodulation_matrix

        m_ij = m_ij.swapaxes(0, 1)
        for i in range(3):
            for j in range(4):
                hdu = fits.PrimaryHDU(data=m_ij[i, j])
                hdu.writeto(output_dir + f"M{i}{j}.fits", overwrite=True)

        return
"""


def compute_demodulation_by_chunk(
    splitted_dara_arr,
    normalizing_S,
    splitted_occulter_flag,
    polarizer_orientations,
    rad_micropol_phases_previsions,
    gain,
):
    # Preemptly compute the theoretical demo matrix to save time
    theo_modulation_matrix = np.array(
        [
            [0.5, 0.5, 0.5, 0.5],
            [
                0.5 * np.cos(2.0 * rad_micropol_phases_previsions[i])
                for i in range(4)
            ],
            [
                0.5 * np.sin(2.0 * rad_micropol_phases_previsions[i])
                for i in range(4)
            ],
        ]
    )
    theo_demodulation_matrix = np.linalg.pinv(theo_modulation_matrix)

    num_of_points, height, width = splitted_dara_arr.shape
    rad_micropol_phases_previsions = np.array(rad_micropol_phases_previsions)
    polarizations_rad = np.deg2rad(polarizer_orientations)
    tk_prediction = 0.5
    efficiency_prediction = 0.8

    # Checked
    sigma_S2 = np.sqrt(0.5 * normalizing_S / gain)
    normalizing_S2 = normalizing_S * normalizing_S
    pix_DN_sigma = np.sqrt(
        splitted_dara_arr / (gain * normalizing_S2)
        + sigma_S2
        * (splitted_dara_arr * splitted_dara_arr)
        / (normalizing_S2 * normalizing_S2)
    )
    normalized_splitted_data = splitted_dara_arr / normalizing_S
    all_zeros = np.zeros(shape=(num_of_points))

    m_ij = np.zeros(shape=(4, 3, height, width))
    tk_data = np.ones(shape=(height, width)) * tk_prediction
    eff_data = np.ones(shape=(height, width)) * efficiency_prediction
    phase_data = np.zeros(shape=(height, width))
    modulation_matrix = np.zeros(shape=(3, 4))
    superpix_params = np.zeros(shape=(4, 3))

    predictions = np.zeros(shape=(4, 3))
    predictions[:, 1] = efficiency_prediction  # Efficiency prediction
    predictions[:, 2] = rad_micropol_phases_previsions  # Angle prediction

    bounds = np.zeros(shape=(4, 2, 3))
    bounds[:, 0, 1], bounds[:, 1, 1] = 0.1, 1.0  # Efficiency bounds
    bounds[:, 0, 2] = rad_micropol_phases_previsions - 15  # Lower angle bounds
    bounds[:, 1, 2] = rad_micropol_phases_previsions + 15  # Upper angle bounds

    # Fit throughput too
    predictions[:, 0] = [tk_prediction] * 4
    bounds[:, 0, 0] = 0.1
    bounds[:, 1, 0] = 1.0

    # Fit for each superpixel. Use theoretical demodulation matrix for
    # occulter if present.
    DEBUG = False
    if DEBUG:
        x_start, x_end = 4, 8
        y_start, y_end = 4, 8
    else:
        y_start, y_end = 0, height
        x_start, x_end = 0, width
    for super_y in range(y_start, y_end, 2):
        for super_x in range(x_start, x_end, 2):
            if not (
                np.any(
                    splitted_occulter_flag[
                        super_y : super_y + 2, super_x : super_x + 2
                    ]
                )
            ):
                normalized_superpix_arr = normalized_splitted_data[
                    :, super_y : super_y + 2, super_x : super_x + 2
                ].reshape(num_of_points, 4)

                # sigma_pix = np.sqrt(
                #    superpix_arr[:] / normalizing_S2
                #    + sigma_S2
                #    * (superpix_arr[:] * superpix_arr[:])
                #    / (normalizing_S2 * normalizing_S2)
                # )
                sigma_pix = pix_DN_sigma[
                    :, super_y : super_y + 2, super_x : super_x + 2
                ].reshape(num_of_points, 4)
                sigma_pix = np.where(sigma_pix != 0.0, sigma_pix, 1.0e-5)

                if False:  # Liberatore method, lock throughput
                    tk = np.max(superpix_arr, axis=0) / normalizing_S
                    predictions[:, 0] = tk
                    bounds[:, 0, 0] = tk - 1.0e-10
                    bounds[:, 1, 0] = tk + 1.0e-10

                for pixel_num in range(4):
                    if np.array_equal(
                        normalized_superpix_arr[:, pixel_num], all_zeros
                    ):  # catch bad pixels
                        fit_success = False
                        break
                    try:
                        (
                            superpix_params[pixel_num],
                            superpix_pcov,
                        ) = curve_fit(
                            Malus,
                            polarizations_rad,
                            normalized_superpix_arr[:, pixel_num],
                            predictions[pixel_num],
                            sigma=sigma_pix[:, pixel_num],
                            bounds=bounds[pixel_num],
                        )
                        fit_success = True
                    except RuntimeError:
                        fit_success = False
                        break

                if False:  # Debug
                    colors = ["blue", "orange", "green", "red"]
                    fig, ax = plt.subplots(figsize=(9, 9))
                    for i in range(4):
                        ax.errorbar(
                            np.rad2deg(polarizations_rad),
                            normalized_superpix_arr[:, i],
                            yerr=sigma_pix[:, i],
                            xerr=[1.0] * len(polarizations_rad),
                            label=f"points {i}",
                            fmt="None",
                            color=colors[i],
                        )
                        min = np.min(polarizations_rad)
                        max = np.max(polarizations_rad)
                        x = np.arange(min, max, (max - min) / 100)
                        x = np.arange(-np.pi / 2, np.pi, np.pi / 100)
                        ax.plot(
                            np.rad2deg(x),
                            Malus(x, *superpix_params[i]),
                            label=f"t = {superpix_params[i,0]:2.2f}, e = {superpix_params[i, 1]:2.2f}, phi = {np.rad2deg(superpix_params[i, 2]):2.2f}",
                        )

                    plt.legend()
                    plt.show()

                if not fit_success:
                    for i in range(2):
                        for j in range(2):
                            m_ij[
                                :, :, super_y + i, super_x + j
                            ] = theo_demodulation_matrix
                    continue

                # Compute modulation matrix and its inverse
                t = superpix_params[:, 0]
                eff = superpix_params[:, 1]
                phi = superpix_params[:, 2]
                modulation_matrix = np.array(
                    [
                        0.5 * t,
                        0.5 * t * eff * np.cos(2.0 * phi),
                        0.5 * t * eff * np.sin(2.0 * phi),
                    ]
                )

                demodulation_matrix = np.linalg.pinv(modulation_matrix)

                for i in range(2):
                    for j in range(2):
                        m_ij[
                            :, :, super_y + i, super_x + j
                        ] = demodulation_matrix

                tk_data[
                    super_y : super_y + 2, super_x : super_x + 2
                ] = np.array(t).reshape(2, 2)
                eff_data[
                    super_y : super_y + 2, super_x : super_x + 2
                ] = np.array(eff).reshape(2, 2)
                phase_data[
                    super_y : super_y + 2, super_x : super_x + 2
                ] = np.array(phi).reshape(2, 2)

            else:
                for i in range(2):
                    for j in range(2):
                        m_ij[
                            :, :, super_y + i, super_x + j
                        ] = theo_demodulation_matrix
                phase_data[
                    super_y : super_y + 2, super_x : super_x + 2
                ] = rad_micropol_phases_previsions.reshape(2, 2)
                tk_data[
                    super_y : super_y + 2, super_x : super_x + 2
                ] = np.array([[1.0, 1.0], [1.0, 1.0]])
                eff_data[
                    super_y : super_y + 2, super_x : super_x + 2
                ] = np.array([[1.0, 1.0], [1.0, 1.0]])

    m_ij_chunk = m_ij.swapaxes(0, 1)

    return m_ij_chunk, tk_data, eff_data, phase_data


def Malus(angle, throughput, efficiency, phase):
    modulated_efficiency = efficiency * (
        np.cos(2.0 * phase) * np.cos(2.0 * angle)
        + np.sin(2.0 * phase) * np.sin(2.0 * angle)
    )
    return 0.5 * throughput * (1.0 + modulated_efficiency)
