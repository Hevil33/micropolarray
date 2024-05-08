import glob
import multiprocessing as mp
import os
import re
import sys
import time
import traceback
from logging import error, info, warning
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from scipy.optimize import curve_fit
from tqdm import tqdm

from micropolarray.cameras import PolarCam
from micropolarray.processing.chen_wan_liang_calibration import _ifov_jitcorrect
from micropolarray.processing.nrgf import find_occulter_position, roi_from_polar
from micropolarray.processing.rebin import (
    micropolarray_rebin,
    standard_rebin,
    trim_to_match_binning,
)
from micropolarray.utils import mean_plus_std, merge_polarizations

# Shape of the demodulation matrix
N_PIXELS_IN_SUPERPIX = 4
N_MALUS_PARAMS = 3


class Demodulator:
    """Demodulation class needed for MicropolImage
    demodulation."""

    def __init__(self, demo_matrices_path: str):
        self.n_pixels_in_superpix = N_PIXELS_IN_SUPERPIX
        self.n_malus_params = N_MALUS_PARAMS
        self.demo_matrices_path = demo_matrices_path

        self.mij, self.fit_found_flags = self._get_demodulation_tensor()

    @property
    def Cij(self) -> np.ndarray:
        covariance_tensor_fnames = glob.glob(
            os.path.join(self.demo_matrices_path, "covariance_tensor", "*")
        )

        if not covariance_tensor_fnames:
            return None
        with fits.open(covariance_tensor_fnames[0]) as firsthul:
            sample_matrix = np.array(firsthul[0].data)

        Cij = np.zeros(
            shape=(
                self.n_malus_params,
                self.n_malus_params,
                sample_matrix.shape[0],
                sample_matrix.shape[1],
            ),
            dtype=float,
        )

        matches = 0
        for filename in covariance_tensor_fnames:
            pattern_query = re.search("[0-9]{2}", filename.split(os.path.sep)[-1])
            if pattern_query is not None:  # Exclude files not matching m/Mij
                matches += 1
                i, j = pattern_query.group()[
                    -2:
                ]  # Searches for pattern ij as last string before .fits
                i, j = int(i), int(j)
                with fits.open(filename) as hul:
                    Cij[i, j] = hul[0].data

        if matches != (self.n_malus_params * self.n_malus_params):
            raise ValueError("Incomplete covariance tensor in the selected folder. ")

        return Cij

    @property
    def eff(self) -> np.ndarray:
        with fits.open(
            os.path.join(self.demo_matrices_path, "efficiences.fits")
        ) as firsthul:
            _ = np.array(firsthul[0].data)
        return _

    @property
    def tk(self) -> np.ndarray:
        with fits.open(
            os.path.join(self.demo_matrices_path, "transmittancies.fits")
        ) as firsthul:
            _ = np.array(firsthul[0].data)
        return _

    @property
    def phi(self) -> np.ndarray:
        with fits.open(
            os.path.join(self.demo_matrices_path, "phases.fits")
        ) as firsthul:
            _ = np.array(firsthul[0].data)
        return _

    def _get_demodulation_tensor(self):
        """Reads files "MIJ.fits" from path folder and returns a (3,4,y,x)
        numpy array representing the demodulation tensor.

        Args:
            binning (bool, optional): _description_. Defaults to False.

        Raises:
            FileNotFoundError: couldn't find the matrices in the specified path

        Returns:
            ndarray: (3, 4, *data.shape) array containing the demodulation tensor
        """

        if not os.path.exists(self.demo_matrices_path):
            raise FileNotFoundError("self.demo_matrices_path not found.")

        # look for first matrix file and get dimensions
        filenames_list = glob.glob(self.demo_matrices_path + os.path.sep + "*.fits")
        if not filenames_list:
            raise FileNotFoundError("No fits files in selected folder.")

        for filename in filenames_list:
            if re.search("[0-9]{2}", filename.split(os.path.sep)[-1]) is not None:
                with fits.open(filename) as firsthul:
                    sample_matrix = np.array(firsthul[0].data)
                break

        Mij = np.zeros(
            shape=(
                self.n_malus_params,
                self.n_pixels_in_superpix,
                sample_matrix.shape[0],
                sample_matrix.shape[1],
            ),
            dtype=float,
        )
        fit_found_flags = None

        matches = 0
        for filename in filenames_list:
            pattern_query = re.search("[0-9]{2}", filename.split(os.path.sep)[-1])
            if pattern_query is not None:  # Exclude files not matching m/Mij
                matches += 1
                i, j = pattern_query.group()[
                    -2:
                ]  # Searches for pattern ij as last string before .fits
                i, j = int(i), int(j)
                with fits.open(filename) as hul:
                    Mij[i, j] = hul[0].data
            if Path(filename).stem == "fit_found_flag":
                with fits.open(filename) as hul:
                    fit_found_flags = hul[0].data

        if matches != 12:
            raise ValueError(
                "Missing matrices in the selected folder. Check correct folder name and files pattern '*ij*.fits'."
            )

        return Mij, fit_found_flags

    def show(self, vmin=-1, vmax=1, cmap="Greys") -> tuple:
        """Shows the demodulation tensor

        Args:
            vmin (int, optional): Minimum shown value. Defaults to -1.
            vmax (int, optional): Maximum shown value. Defaults to 1.
            cmap (str, optional): Colormap of the plot. Defaults to "Greys".

        Returns:
            tuple: fig, ax tuple as returned by matplotlib.pyplot.subplots
        """
        fig, ax = plt.subplots(
            3,
            4,
            dpi=300,
            figsize=(4, 3),
            constrained_layout=True,
            sharex="col",
            sharey="row",
        )
        for i in range(3):
            for j in range(4):
                mappable = ax[i, j].imshow(
                    self.mij[i, j], cmap=cmap, vmin=vmin, vmax=vmax
                )
                ax[i, j].set_title(rf"M$_{i}$$_{j}$")
        for ax in fig.get_axes():
            ax.label_outer()

        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
        fig.colorbar(mappable, cax=cbar_ax)

        return fig, ax

    def rebin(self, binning):  # TODO
        """DO NOT USE THIS, calculate the tensor from the binned images"""
        if (int(self.mij.shape[2] / binning) % 2) or (
            int(self.mij.shape[3] / binning) % 2
        ):
            raise ValueError(
                f"incorrect binning, resulting matrix would be {int(self.mij.shape[2] / binning)}x{int(self.mij.shape[3] / binning)} (not even values)."
            )
        new_demodulator = Demodulator(self.demo_matrices_path)
        new_mij = np.zeros(
            shape=(
                new_demodulator.n_malus_params,
                new_demodulator.n_pixels_in_superpix,
                int(new_demodulator.mij.shape[2] / binning),
                int(new_demodulator.mij.shape[3] / binning),
            )
        )
        for j in range(new_demodulator.n_malus_params):
            for i in range(new_demodulator.n_pixels_in_superpix):
                new_mij[j, i] = standard_rebin(new_demodulator.mij[j, i], binning) / (
                    binning * binning
                )
        new_demodulator.mij = new_mij

        return new_demodulator

    def rot90(self, k=1):
        # NB: rotation could switch micropol positions inside superpixel, but for demo matrix all px in superpix are equal
        for i in range(self.n_malus_params):
            for j in range(self.n_pixels_in_superpix):
                self.mij[i, j] = np.rot90(self.mij[i, j], k=k)

    def flip(self, axis):
        for i in range(self.n_malus_params):
            for j in range(self.n_pixels_in_superpix):
                self.mij[i, j] = np.flip(self.mij[i, j], axis=axis)


def calculate_demodulation_tensor(
    polarizer_orientations: list,
    filenames_list: list,
    micropol_phases_previsions: list,
    gain: float,  #  needed for errors
    output_dir: str,
    binning: int = 1,
    occulter: list = None,
    procs_grid: list = [4, 4],
    dark_filename: str = None,
    flat_filename: str = None,
    normalizing_S=None,
    tk_boundary: list = None,
    eff_boundary: list = None,
    DEBUG: bool = False,
):
    """Calculates the demodulation tensor images and saves them. Requires a set of images with different polarizations to fit a Malus curve model.

    Args:
        polarizer_orientations (list[float]): List containing the orientations of the incoming light for each image.
        filenames_list (list[str]): List of input images filenames to read. Must include [0, 45, 90, -45].
        micropol_phases_previsions (list[float]): Previsions for the micropolarizer orientations required to initialize fit.
        gain (float): Detector [e-/DN], required to compute errors.
        output_dir (str): output folder to save matrix to.
        binning (int, optional): Output matrices binning. Defaults to 1 (no binning). Be warned that binning matrices AFTER calculation is an incorrect operation.
        occulter (list, optional): occulter y, x center and radius to exclude from calculations. Defaults to None.
        procs_grid ([int, int], optional): number of processors per side [Y, X], parallelization will be done in a Y x X grid. Defaults to [4,4] (16 procs in a 4x4 grid).
        dark_filename (str, optional): Dark image filename to correct input images. Defaults to None.
        flat_filename (str, optional): Flat image filename to correct input images. Defaults to None.
        normalizing_S (float or np.ndarray, optional): maximum signal used to normalize single pixel signal. If not set will be estimated as the 4sigma of the signal distribution.
        tk_boundary (list): if provided, sets the transmittancy [initial guess, boundary_inf, boundary_sup] of the Malus curve (max value). Defaults to [0.5, 0.1, 1.-1.e-6].
        eff_boundary (list): if provided, sets the efficiency [initial guess, boundary_inf, boundary_sup] of the Malus curve (max value). Defaults to [0.5, 0.1, 1.-1.e-6].

    Raises:
        ValueError: Raised if any among [0, 45, 90, -45] is not included in the input polarizations.

    Notes:
        In the binning process the sum of values is considered, which is ok because data is normalized over the maximum S before being fitted.
    """

    correct_ifov = False

    output_path = Path(output_dir)
    if not output_path.is_dir():  # create the path if it doesnt exist
        output_path.mkdir(parents=True)

    # polarizations = array of polarizer orientations
    # filenames_list = list of filenames
    available_norms = [[0, 45, 90, -45], [0, 45, 90, 135]]
    if (normalizing_S is None) and (
        not np.all(np.isin(available_norms[0], polarizer_orientations))
        and not np.all(np.isin(available_norms[1], polarizer_orientations))
    ):
        raise ValueError(
            f"Each one among (0, 45, 90, -45 / 135) polarizations must be included in the polarizer orientation array. Provided {polarizer_orientations}"
        )  # for calculating normalizing_S

    polarizer_orientations, filenames_list = (
        list(t) for t in zip(*sorted(zip(polarizer_orientations, filenames_list)))
    )
    micropol_phases_previsions = np.array(micropol_phases_previsions)
    rad_micropol_phases_previsions = np.deg2rad(micropol_phases_previsions)

    # Flag occulter position to not be fitted, expand to superpixel.
    with fits.open(filenames_list[0]) as file:
        data = file[0].data  # get data dimension

    # Count binning before dimensions
    data = np.array(data, dtype=float)
    data = micropolarray_rebin(data, binning=binning)
    height, width = data.shape

    occulter_flag = np.ones_like(data)  # 0 if not a occulted px, 1 otherwise
    if occulter:
        # info("Cleaning occulter...")
        ## Mean values from coronal observations 2022_12_03
        ## (campagna_2022/mean_occulter_pos.py)
        # occulter_y, occulter_x, occulter_r = PolarCam().occulter_pos_last
        # overoccult = 16
        # overoccult = 0
        # occulter_r = occulter_r + overoccult

        # Match binning if needed
        binned_occulter = [int(val / binning) for val in occulter]
        occulter_y, occulter_x, occulter_r = binned_occulter

        occulter_flag = roi_from_polar(
            occulter_flag, [occulter_y, occulter_x], [0, occulter_r]
        )
        for super_y in range(0, occulter_flag.shape[0], 2):
            for super_x in range(0, occulter_flag.shape[1], 2):
                if np.any(occulter_flag[super_y : super_y + 2, super_x : super_x + 2]):
                    occulter_flag[super_y : super_y + 2, super_x : super_x + 2] = 1
                    continue
    else:
        occulter_flag *= 0

    # Collect dark
    if dark_filename:
        with fits.open(dark_filename) as file:
            dark = np.array(file[0].data, dtype=np.float)
            dark = micropolarray_rebin(dark, binning)
    # Collect flat field, normalize it
    if flat_filename:
        with fits.open(flat_filename) as file:
            flat = np.array(file[0].data, dtype=np.float)
        if correct_ifov:
            flat = _ifov_jitcorrect(flat, *flat.shape)
        flat = micropolarray_rebin(flat, binning)
        # flat_max = np.max(flat, axis=(0, 1))
        flat_max = mean_plus_std(flat, stds_n=1)
    """
    if flat_filename and dark_filename:
        flat -= dark  # correct flat too
        flat = np.where(flat > 0, flat, 1.0)
        if occulter:
            flat = np.where(occulter_flag, 1.0, flat)
        # flat_max = np.max(flat, axis=(0, 1))
        flat_max = mean_plus_std(flat, stds_n=1)
    """
    if flat_filename:
        normalized_flat = np.where(occulter_flag, 1.0, flat / flat_max)

    # collect data
    all_data_arr = [0.0] * len(filenames_list)
    info("Collecting data from files...")
    for idx, filename in enumerate(filenames_list):
        with fits.open(filename) as file:
            all_data_arr[idx] = np.array(file[0].data, dtype=float)
            if correct_ifov:
                all_data_arr[idx] = _ifov_jitcorrect(
                    all_data_arr[idx], *all_data_arr[idx].shape
                )
            all_data_arr[idx] = micropolarray_rebin(all_data_arr[idx], binning)
            if dark_filename is not None:
                all_data_arr[idx] -= dark
                all_data_arr[idx] = np.where(
                    all_data_arr[idx] >= 0, all_data_arr[idx], 0.0
                )
            if flat_filename is not None:
                np.divide(
                    all_data_arr[idx],
                    normalized_flat,
                    out=all_data_arr[idx],
                    where=normalized_flat != 0.0,
                )
    all_data_arr = np.array(all_data_arr)

    if normalizing_S is None:
        info("Calculating normalization...")
        S_max = np.zeros(shape=(height, width))  # tk_sum = tk_0 + tk_45 + tk_90 + tk_45
        for chosen_norm in available_norms:
            if np.all(np.isin(chosen_norm, polarizer_orientations)):
                norm_S_angle_list = chosen_norm
        for pol, image in zip(polarizer_orientations, all_data_arr):
            if pol in norm_S_angle_list:
                S_max += 0.5 * image
        # Normalizing S, has a spike of which maximum is taken
        bins = 1000
        histo = np.histogram(S_max, bins=bins)
        maxvalue = np.max(histo[0])
        index = np.where(histo[0] == maxvalue)[0][0]
        normalizing_S = (
            histo[1][index] + histo[1][index + 1] + histo[1][index - 1]
        ) / 3
        # normalizing_S = np.max(S_max) # old

        if False:
            fig, ax = plt.subplots()
            ax.stairs(*histo)
            ax.axvline(normalizing_S, color="r")
            plt.show()

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
    elif type(normalizing_S) is not np.ndarray:  # its a number
        normalizing_S *= binning * binning  # account binning
        normalizing_S = np.ones(shape=(height, width), dtype=float) * normalizing_S
    elif (
        type(normalizing_S) is np.ndarray and len(normalizing_S.shape) == 2
    ):  # its an image
        normalizing_S = micropolarray_rebin(normalizing_S, binning)
    elif (
        type(normalizing_S) is np.ndarray and len(normalizing_S.shape) == 3
    ):  # series of images
        S_n = normalizing_S.shape[0]
        _ = []
        for i in range(S_n):
            _.append(micropolarray_rebin(normalizing_S[i], binning))
        normalizing_S = np.array(_)

    # correct S=0 error
    normalizing_S = np.where(normalizing_S >= 0, normalizing_S, 1)

    if DEBUG:
        procs_grid = [1, 1]

    # parallelize into a procs_per_size x procs_per_size grid
    info("Splitting into subdomains to parallelize...")
    chunks_n_y, chunks_n_x = procs_grid
    chunk_size_y = int(height / chunks_n_y)
    chunk_size_x = int(width / chunks_n_x)
    if (chunk_size_x % 2) or (chunk_size_y % 2):
        raise ValueError(
            f"cant decompose into a {procs_grid[0]}x{procs_grid[1]} grid (odd side grid {chunk_size_y}x{chunk_size_x}). Try changing the number of processors."
        )
    splitted_data = np.zeros(
        shape=(
            chunks_n_y * chunks_n_x,
            len(polarizer_orientations),
            chunk_size_y,
            chunk_size_x,
        ),
        dtype=float,
    )
    splitted_pixel_errors = np.zeros_like(splitted_data, dtype=float)
    splitted_occulter = np.zeros(
        shape=(chunks_n_y * chunks_n_x, chunk_size_y, chunk_size_x)
    )

    # calculate normalized data and its error
    # sigma_data = sqrt(e) = sqrt(data / gain) (poisson)
    # sigma_norm = sqrt(data / gain) = sqrt(norm / gain / S) (propagation)

    np.divide(all_data_arr, normalizing_S, out=all_data_arr)
    pixel_errors = np.zeros_like(all_data_arr)
    np.divide(all_data_arr, normalizing_S * gain, out=pixel_errors)
    np.sqrt(pixel_errors, out=pixel_errors)

    for i in range(chunks_n_y):
        for j in range(chunks_n_x):
            splitted_data[i + chunks_n_y * j] = np.array(
                all_data_arr[
                    :,
                    i * (chunk_size_y) : (i + 1) * chunk_size_y,
                    j * (chunk_size_x) : (j + 1) * chunk_size_x,
                ]
            )  # shape = (chunks_n*chunks_n, len(filenames_list), chunk_size_y, chunk_size_x)
            splitted_pixel_errors[i + chunks_n_y * j] = np.array(
                pixel_errors[
                    :,
                    i * (chunk_size_y) : (i + 1) * chunk_size_y,
                    j * (chunk_size_x) : (j + 1) * chunk_size_x,
                ]
            )
            splitted_occulter[i + chunks_n_y * j] = np.array(
                occulter_flag[
                    i * (chunk_size_y) : (i + 1) * chunk_size_y,
                    j * (chunk_size_x) : (j + 1) * chunk_size_x,
                ]
            )  # shape = (chunks_n*chunks_n, chunk_size_y, chunk_size_x)

    # Checked errors
    # sigma_S2 = np.sqrt(0.5 * normalizing_S / gain)
    # splitted_sigma_S2 = np.sqrt(0.5 * splitted_normalizing_S / gain)
    # normalizing_S2 = normalizing_S * normalizing_S
    # splitted_S2 = splitted_normalizing_S * splitted_normalizing_S
    # pix_DN_sigma = np.sqrt(
    #    splitted_dara_arr / (gain * normalizing_S2)
    #    + sigma_S2
    #    * (splitted_dara_arr * splitted_dara_arr)
    #    / (normalizing_S2 * normalizing_S2)
    # )
    # pix_DN_sigma = (
    #    np.sqrt(splitted_dara_arr / gain) / normalizing_S
    # )  # poisson error on the photoelectrons
    # pix_DN_sigma = np.sqrt(
    #    np.divide(
    #        splitted_normalized_dara_arr[:], splitted_normalizing_S * gain
    #    )
    # )

    # pix_DN_sigma = np.sqrt(splitted_normalized_dara_arr / gain)

    # Debug
    if False:
        index = 0
        histo_0 = np.histogram(all_data_arr[index], bins=1000)
        fig, ax = plt.subplots(figsize=(9, 9))
        ax.stairs(histo_0[0], histo_0[1], label="sample image")
        ax.stairs(histo[0], histo[1], label=f"S, max = {np.max(S_max)}")
        ax.axvline(normalizing_S, color="red", label="normalizing_S")
        # ax.plot(
        #    xvalues,
        #    gauss(xvalues, params[0] * yvalues_sum, params[1], params[2]),
        #    label="Fitted curve for normalizing S",
        # )
        ax.set_title(f"Prepol at {polarizer_orientations[index]} deg")
        ax.set_xlabel("S [DN]")
        ax.set_ylabel("Counts")
        ax.legend()
        plt.show()
        sys.exit()

    args = (
        [
            splitted_data[i],
            splitted_pixel_errors[i],
            splitted_occulter[i],
            polarizer_orientations,
            rad_micropol_phases_previsions,
            tk_boundary,
            eff_boundary,
            DEBUG,
        ]
        for i in range(chunks_n_y * chunks_n_x)
    )

    starting_time = time.perf_counter()
    loc_time = time.strftime("%H:%M:%S  (%Y/%m/%d)", time.localtime())
    info(f"Starting parallel calculation ({procs_grid[0]}x{procs_grid[1]}) processors")

    if procs_grid != [1, 1]:
        try:
            with mp.Pool(processes=chunks_n_y * chunks_n_x) as p:
                result = p.starmap(
                    compute_demodulation_by_chunk,
                    args,
                )

        except Exception as e:
            traceback.print_exc()
            ending_time = time.perf_counter()

            info(f"Elapsed : {(ending_time - starting_time)/60:3.2f} mins")
            sys.exit()

    else:
        arglist = [arg for arg in args]
        result = [[0.0, 0.0]] * chunks_n_y * chunks_n_x

        for i in range(chunks_n_y * chunks_n_x):
            result[i] = compute_demodulation_by_chunk(*arglist[i])

    loc_time = time.strftime("%H:%M:%S (%Y/%m/%d)", time.localtime())
    info(f"Ending parallel calculation")

    ending_time = time.perf_counter()
    info(f"Elapsed : {(ending_time - starting_time)/60:3.2f} mins")

    result = np.array(result, dtype=object)
    m_ij = np.zeros(
        shape=(
            N_MALUS_PARAMS,
            N_PIXELS_IN_SUPERPIX,
            int(height / 2),
            int(width / 2),
        )
    )
    fit_found_flag = np.zeros(shape=(int(height / 2), int(width / 2)))
    covariance_tensor = np.zeros(shape=(N_MALUS_PARAMS, N_MALUS_PARAMS, height, width))

    tks = np.zeros(shape=(height, width))
    efficiences = np.zeros(shape=(height, width))
    phases = np.zeros(shape=(height, width))
    chisqs = np.zeros(shape=(height, width))

    def _merge_parameter(parameter: np.ndarray, param_ID: int):
        for i in range(chunks_n_y):
            for j in range(chunks_n_x):
                parameter[
                    i * (chunk_size_y) : (i + 1) * chunk_size_y,
                    j * (chunk_size_x) : (j + 1) * chunk_size_x,
                ] = result[i + chunks_n_y * j, param_ID].reshape(
                    chunk_size_y, chunk_size_x
                )

    _merge_parameter(tks, 1)
    _merge_parameter(efficiences, 2)
    _merge_parameter(phases, 3)
    _merge_parameter(chisqs, 6)

    half_chunk_size_y = int(chunk_size_y / 2)
    half_chunk_size_x = int(chunk_size_x / 2)

    # merge flags
    for i in range(chunks_n_y):
        for j in range(chunks_n_x):
            fit_found_flag[
                i * (half_chunk_size_y) : (i + 1) * half_chunk_size_y,
                j * (half_chunk_size_x) : (j + 1) * half_chunk_size_x,
            ] = result[i + chunks_n_y * j, 4].reshape(
                half_chunk_size_y,
                half_chunk_size_x,
            )

    # merge demodulation tensor
    for i in range(chunks_n_y):
        for j in range(chunks_n_x):
            shaped_result = result[i + chunks_n_y * j, 0].reshape(
                N_MALUS_PARAMS,
                N_PIXELS_IN_SUPERPIX,
                half_chunk_size_y,
                half_chunk_size_x,
            )
            m_ij[
                :,
                :,
                i * (half_chunk_size_y) : (i + 1) * half_chunk_size_y,
                j * (half_chunk_size_x) : (j + 1) * half_chunk_size_x,
            ] = shaped_result

    # merge covariance tensor
    for i in range(chunks_n_y):
        for j in range(chunks_n_x):
            covariance_tensor[
                :,
                :,
                i * (chunk_size_y) : (i + 1) * chunk_size_y,
                j * (chunk_size_x) : (j + 1) * chunk_size_x,
            ] = result[i + chunks_n_y * j, 5].reshape(
                N_MALUS_PARAMS, N_MALUS_PARAMS, chunk_size_y, chunk_size_x
            )

    phases = np.rad2deg(phases)

    if DEBUG:
        # prevents overwriting
        sys.exit()

    # if not os.path.exists(output_dir):
    #    p = Path(output_dir)
    #    # p.mkdir(parents=True, exist_ok=True)
    #    os.makedirs(output_dir)

    output_str = str(output_path)
    for i in range(N_MALUS_PARAMS):
        for j in range(N_PIXELS_IN_SUPERPIX):
            hdu = fits.PrimaryHDU(data=m_ij[i, j])
            hdu.writeto(output_str + os.path.sep + f"M{i}{j}.fits", overwrite=True)

    Path(output_str + os.path.sep + "covariance_tensor").mkdir(
        parents=True, exist_ok=True
    )
    for i in range(N_MALUS_PARAMS):
        for j in range(N_MALUS_PARAMS):
            hdu = fits.PrimaryHDU(data=covariance_tensor[i, j])
            hdu.writeto(
                output_str
                + os.path.sep
                + f"covariance_tensor"
                + os.path.sep
                + f"C{i}{j}.fits",
                overwrite=True,
            )

    for parameter_data, parameter_name in zip(
        [tks, efficiences, phases, fit_found_flag, chisqs],
        [
            "transmittancies",
            "efficiences",
            "phases",
            "fit_found_flag",
            os.path.sep + "covariance_tensor" + os.path.sep + "reduced_chisquare",
        ],
    ):
        hdu = fits.PrimaryHDU(data=parameter_data)
        hdu.writeto(output_str + os.path.sep + parameter_name + ".fits", overwrite=True)

    info("Demodulation matrices and fit data successfully saved!")

    return


def compute_demodulation_by_chunk(
    splitted_normalized_dara_arr,
    splitted_pixel_erorrs,
    splitted_occulter_flag,
    polarizer_orientations,
    rad_micropol_phases_previsions,
    tk_boundary,
    eff_boundary,
    DEBUG,
):
    """Utility function to parallelize calculations."""
    N_MALUS_PARAMS = 3
    N_PIXELS_IN_SUPERPIX = 4

    dof = len(polarizer_orientations) - 2
    if tk_boundary is None:
        tk_boundary = [0.5, 0.1, 1.0 - 1.0e-6]
        dof -= 1
    if eff_boundary is None:
        eff_boundary = [0.5, 0.1, 1.0 - 1.0e-6]
        dof -= 1

    # Preemptly compute the theoretical demo matrix to save time
    theo_modulation_matrix = np.array(
        [
            [0.5, 0.5, 0.5, 0.5],
            [
                0.5 * np.cos(2.0 * rad_micropol_phases_previsions[i])
                for i in range(N_PIXELS_IN_SUPERPIX)
            ],
            [
                0.5 * np.sin(2.0 * rad_micropol_phases_previsions[i])
                for i in range(N_PIXELS_IN_SUPERPIX)
            ],
        ],
        dtype=float,
    )
    theo_modulation_matrix = theo_modulation_matrix.T
    theo_demodulation_matrix = np.linalg.pinv(theo_modulation_matrix)

    num_of_points, height, width = splitted_normalized_dara_arr.shape
    rad_micropol_phases_previsions = np.array(
        rad_micropol_phases_previsions, dtype=float
    )
    polarizations_rad = np.deg2rad(polarizer_orientations)
    tk_prediction = tk_boundary[0]
    efficiency_prediction = eff_boundary[0]

    all_zeros = np.zeros(shape=(num_of_points))
    m_ij = np.zeros(
        shape=(
            N_MALUS_PARAMS,
            N_PIXELS_IN_SUPERPIX,
            int(height / 2),
            int(width / 2),
        )
    )  # demodulation matrix
    fit_found = np.zeros_like(m_ij[0, 0])
    covariance_tensor = np.zeros(shape=(N_MALUS_PARAMS, N_MALUS_PARAMS, height, width))
    tk_data = np.ones(shape=(height, width)) * tk_prediction
    eff_data = np.ones(shape=(height, width)) * efficiency_prediction
    phase_data = np.zeros(shape=(height, width))
    chisq_data = np.zeros(shape=(height, width))
    phase_data[0::2, 0::2] = rad_micropol_phases_previsions[0]
    phase_data[0::2, 1::2] = rad_micropol_phases_previsions[1]
    phase_data[1::2, 0::2] = rad_micropol_phases_previsions[2]
    phase_data[1::2, 1::2] = rad_micropol_phases_previsions[3]

    superpix_params = np.zeros(shape=(N_PIXELS_IN_SUPERPIX, N_MALUS_PARAMS))
    superpix_covtensor = np.zeros(
        shape=(N_MALUS_PARAMS, N_MALUS_PARAMS, N_PIXELS_IN_SUPERPIX)
    )
    chisq = np.zeros(shape=(4))

    predictions = np.zeros(shape=(N_PIXELS_IN_SUPERPIX, N_MALUS_PARAMS))
    predictions[:, 0] = tk_prediction  # Throughput prediction
    predictions[:, 1] = efficiency_prediction  # Efficiency prediction
    predictions[:, 2] = rad_micropol_phases_previsions  # Angle prediction

    bounds = np.zeros(shape=(N_PIXELS_IN_SUPERPIX, 2, N_MALUS_PARAMS))
    bounds[:, 0, 0], bounds[:, 1, 0] = tk_boundary[1:]  # Throughput bounds
    bounds[:, 0, 1], bounds[:, 1, 1] = eff_boundary[1:]  # Efficiency bounds
    bounds[:, 0, 2] = rad_micropol_phases_previsions - 15  # Lower angle bounds
    bounds[:, 1, 2] = rad_micropol_phases_previsions + 15  # Upper angle bounds

    # Fit for each superpixel. Use theoretical demodulation matrix for
    # occulter if present.
    if DEBUG:
        x_start, x_end = 100, 150
        # x_start, x_end = 500, 510
        # x_start, x_end = 0, 2
        y_start, y_end = 100, 150
        # y_start, y_end = 500, 510
        # y_start, y_end = 0, 2
    else:
        y_start, y_end = 0, height
        x_start, x_end = 0, width
    milestones = [
        int(height / 4),
        int(height / 4) + 1,
        int(height / 2),
        int(height / 2) + 1,
        int(3 * height / 4),
        int(3 * height / 4) + 1,
        int(height),
        int(height) + 1,
    ]  # used for printing progress
    for super_y in range(y_start, y_end, 2):
        if super_y in milestones:
            print(f"Thread at {super_y / height*100:.2f} %", flush=True)
        for super_x in range(x_start, x_end, 2):
            if not (
                np.any(
                    splitted_occulter_flag[super_y : super_y + 2, super_x : super_x + 2]
                )
            ):
                normalized_superpix_arr = splitted_normalized_dara_arr[
                    :, super_y : super_y + 2, super_x : super_x + 2
                ].reshape(num_of_points, N_PIXELS_IN_SUPERPIX)

                sigma_pix = splitted_pixel_erorrs[
                    :, super_y : super_y + 2, super_x : super_x + 2
                ].reshape(num_of_points, N_PIXELS_IN_SUPERPIX)
                sigma_pix = np.where(sigma_pix != 0.0, sigma_pix, 1.0e-5)

                for pixel_num in range(N_PIXELS_IN_SUPERPIX):
                    if np.array_equal(
                        normalized_superpix_arr[:, pixel_num], all_zeros
                    ):  # catch bad pixels
                        fit_success = False
                        break
                    try:
                        (
                            superpix_params[pixel_num],
                            superpix_covtensor[:, :, pixel_num],
                            # superpix_pcov,
                        ) = curve_fit(
                            Malus,
                            polarizations_rad,
                            normalized_superpix_arr[:, pixel_num],
                            predictions[pixel_num],
                            sigma=sigma_pix[:, pixel_num],
                            absolute_sigma=True,
                            bounds=bounds[pixel_num],
                            xtol=1.0e-5,  # may save time
                        )
                        fit_success = True

                        residuals = (
                            Malus(polarizations_rad, *superpix_params[pixel_num])
                            - normalized_superpix_arr[:, pixel_num]
                        )
                        chisq[pixel_num] = (
                            np.sum((residuals * residuals) / sigma_pix[:, pixel_num])
                            / dof
                        )

                    except:  # catches all exceptions
                        fit_success = False
                        break

                if DEBUG:  # DEBUG
                    colors = ["blue", "orange", "green", "red"]
                    fig, ax = plt.subplots(dpi=200, constrained_layout=True)
                    for i in range(4):
                        print(f"{np.min(normalized_superpix_arr[:, i]) = :3.2f}")
                        ax.errorbar(
                            np.rad2deg(polarizations_rad),
                            normalized_superpix_arr[:, i],
                            yerr=sigma_pix[:, i],
                            xerr=[1.0] * len(polarizations_rad),
                            label=f"points {i}",
                            fmt="k-",
                            color=colors[i],
                            linestyle="none",
                        )
                        min = np.min(polarizations_rad)
                        max = np.max(polarizations_rad)
                        x = np.arange(min, max, (max - min) / 100)
                        ax.plot(
                            np.rad2deg(x),
                            Malus(x, *superpix_params[i]),
                            label=f"t = {superpix_params[i, 0]:2.5f}, e = {superpix_params[i, 1]:2.5f}, phi = {np.rad2deg(superpix_params[i, 2]):2.4f}",
                            color=colors[i],
                        )
                        print(Malus(superpix_params[i, 2], *superpix_params[i]))
                        print(superpix_params)
                        ax.axhline(1.0)
                        ax.set_title(f"super_y = {super_y}, super_x = {super_x},")
                        ax.set_xlabel("Prepolarizer orientations [deg]")
                        ax.set_ylabel("signal / S")

                    plt.legend()
                    plt.show()

                if not fit_success:
                    m_ij[:, :, int(super_y / 2), int(super_x / 2)] = (
                        theo_demodulation_matrix
                    )
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
                    ],
                    dtype=float,
                )
                modulation_matrix = modulation_matrix.T
                demodulation_matrix = np.linalg.pinv(modulation_matrix)

                if DEBUG:
                    print()
                    print("MOD")
                    print(modulation_matrix)
                    print(theo_modulation_matrix)

                    print()
                    print("DEMOD")
                    print(demodulation_matrix)
                    print(theo_demodulation_matrix)

                    print()
                    print("params")
                    print(t)
                    print(eff)
                    print(phi)

                    print()
                    print("covariance matrix")
                    for i in range(N_PIXELS_IN_SUPERPIX):
                        print(superpix_covtensor[:, :, i])

                    print()
                    print("chi square")
                    print(chisq)
                    print("---")

                # Remove matrices with big numbers
                if np.any(demodulation_matrix > 100) or np.any(
                    demodulation_matrix < -100
                ):
                    demodulation_matrix = theo_demodulation_matrix
                    fit_success = False

                m_ij[:, :, int(super_y / 2), int(super_x / 2)] = demodulation_matrix
                fit_found[int(super_y / 2), int(super_x / 2)] = 1.0 * fit_success

                covariance_tensor[
                    :, :, super_y : super_y + 2, super_x : super_x + 2
                ] = superpix_covtensor.reshape(N_MALUS_PARAMS, N_MALUS_PARAMS, 2, 2)

                chisq_data[super_y : super_y + 2, super_x : super_x + 2] = np.array(
                    chisq, dtype=float
                ).reshape(2, 2)

                tk_data[super_y : super_y + 2, super_x : super_x + 2] = np.array(
                    t, dtype=float
                ).reshape(2, 2)
                eff_data[super_y : super_y + 2, super_x : super_x + 2] = np.array(
                    eff, dtype=float
                ).reshape(2, 2)
                phase_data[super_y : super_y + 2, super_x : super_x + 2] = np.array(
                    phi, dtype=float
                ).reshape(2, 2)

            else:  # pixel is in occulter region
                m_ij[:, :, int(super_y / 2), int(super_x / 2)] = (
                    theo_demodulation_matrix
                )
                phase_data[super_y : super_y + 2, super_x : super_x + 2] = (
                    rad_micropol_phases_previsions.reshape(2, 2)
                )
                tk_data[super_y : super_y + 2, super_x : super_x + 2] = np.array(
                    [
                        [tk_prediction, tk_prediction],
                        [tk_prediction, tk_prediction],
                    ],
                    dtype=float,
                )
                eff_data[super_y : super_y + 2, super_x : super_x + 2] = np.array(
                    [
                        [efficiency_prediction, efficiency_prediction],
                        [efficiency_prediction, efficiency_prediction],
                    ],
                    dtype=float,
                )

    return m_ij, tk_data, eff_data, phase_data, fit_found, covariance_tensor, chisq_data


def Malus(angle, throughput, efficiency, phase):
    # original
    modulated_efficiency = efficiency * (
        np.cos(2.0 * phase) * np.cos(2.0 * angle)
        + np.sin(2.0 * phase) * np.sin(2.0 * angle)
    )
    return 0.5 * throughput * (1.0 + modulated_efficiency)
