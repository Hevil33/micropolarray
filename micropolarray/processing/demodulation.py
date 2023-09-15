from micropolarray.processing.rebin import (
    standard_rebin,
    micropolarray_rebin,
    trim_to_match_binning,
)
from micropolarray.processing.nrgf import (
    find_occulter_position,
    roi_from_polar,
)
from micropolarray.cameras import PolarCam

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
import time
from logging import warning, info, error
from micropolarray.processing.chen_wan_liang_calibration import (
    ifov_jitcorrect,
)
from micropolarray.utils import mean_plus_std

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

        self.mij = self._get_demodulation_tensor()

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
        filenames_list = glob.glob(
            self.demo_matrices_path + os.path.sep + "*.fits"
        )

        with fits.open(filenames_list[0]) as firsthul:
            sample_data = np.array(firsthul[0].data)
        Mij = np.zeros(
            shape=(
                self.n_malus_params,
                self.n_pixels_in_superpix,
                sample_data.shape[0],
                sample_data.shape[1],
            ),
            dtype=float,
        )

        matches = 0
        for filename in filenames_list:
            if (
                re.search("[Mm][0-9]{2}", filename.split(os.path.sep)[-1])
                is not None
            ):  # Exclude files not matching m/Mij
                matches += 1
                i, j = re.search(
                    "[Mm][0-9]{2}", filename.split(os.path.sep)[-1]
                ).group()[
                    -2:
                ]  # Searches for pattern M/m + ij as last string before .fits
                i = int(i)
                j = int(j)
                with fits.open(filename) as hul:
                    Mij[i, j] = hul[0].data
        if matches != 12:
            raise ValueError(
                "Some matrices were not found in the selected folder. Check correct folder name and files pattern Mij.fits or mij.fits"
            )

        return Mij

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
                new_mij[j, i] = standard_rebin(
                    new_demodulator.mij[j, i], binning
                ) / (binning * binning)
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
    polarizer_orientations,
    filenames_list,
    micropol_phases_previsions,
    gain,  #  needed for errors
    output_dir,
    binning=1,
    occulter=None,
    procs_grid=[4, 4],
    dark_filename=None,
    flat_filename=None,
    normalizing_S=None,
    DEBUG=False,
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
        normalizing_S (float, optional): maximum signal used to normalize single pixel signal. Defaults to None.

    Raises:
        ValueError: Raised if any among [0, 45, 90, -45] is not included in the input polarizations.

    Notes:
        In the binning process the sum of values is considered, which is ok because data is normalized over the maximum S before being fitted.
    """

    correct_ifov = True

    # polarizations = array of polarizer orientations
    # filenames_list = list of filenames
    firstcall = True
    if (normalizing_S is None) and (
        not np.all(np.isin([0, 45, 90, -45], polarizer_orientations))
    ):
        raise ValueError(
            "Each one among (0, 45, 90, -45) polarizations must be included in the polarizer orientation array"
        )  # for calculating normalizing_S
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

    print("qui")
    # Count binning before dimensions
    data = np.array(data, dtype=float)
    data = micropolarray_rebin(data, *data.shape, binning=binning)
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
            dark = micropolarray_rebin(dark, *dark.shape, binning)
    # Collect flat field, normalize it
    if flat_filename:
        with fits.open(flat_filename) as file:
            flat = np.array(file[0].data, dtype=np.float)
        if correct_ifov:
            flat = ifov_jitcorrect(flat, *flat.shape)
        flat = micropolarray_rebin(flat, *flat.shape, binning)
        # flat_max = np.max(flat, axis=(0, 1))
        flat_max = mean_plus_std(flat, stds_n=1)
    if flat_filename and dark_filename:
        flat -= dark  # correct flat too
        flat = np.where(flat > 0, flat, 1.0)
        if occulter:
            flat = np.where(occulter_flag, 1.0, flat)
        # flat_max = np.max(flat, axis=(0, 1))
        flat_max = mean_plus_std(flat, stds_n=1)
    if flat_filename:
        normalized_flat = np.where(occulter_flag, 1.0, flat / flat_max)

    # collect data
    all_data_arr = [0.0] * len(filenames_list)
    info("Collecting data from files...")
    for idx, filename in enumerate(filenames_list):
        with fits.open(filename) as file:
            all_data_arr[idx] = np.array(file[0].data, dtype=float)
            if correct_ifov:
                all_data_arr[idx] = ifov_jitcorrect(
                    all_data_arr[idx], *all_data_arr[idx].shape
                )
            all_data_arr[idx] = micropolarray_rebin(
                all_data_arr[idx], *all_data_arr[idx].shape, binning
            )
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
        )
    )
    splitted_occulter = np.zeros(
        shape=(chunks_n_y * chunks_n_x, chunk_size_y, chunk_size_x)
    )
    for i in range(chunks_n_y):
        for j in range(chunks_n_x):
            splitted_data[i + chunks_n_y * j] = np.array(
                all_data_arr[
                    :,
                    i * (chunk_size_y) : (i + 1) * chunk_size_y,
                    j * (chunk_size_x) : (j + 1) * chunk_size_x,
                ]
            )  # shape = (chunks_n*chunks_n, len(filenames_list), chunk_size_y, chunk_size_x)
            splitted_occulter[i + chunks_n_y * j] = np.array(
                occulter_flag[
                    i * (chunk_size_y) : (i + 1) * chunk_size_y,
                    j * (chunk_size_x) : (j + 1) * chunk_size_x,
                ]
            )  # shape = (chunks_n*chunks_n, chunk_size_y, chunk_size_x)

    if normalizing_S is None:
        info("Calculating normalization...")
        S_max = np.zeros(
            shape=(height, width)
        )  # tk_sum = tk_0 + tk_45 + tk_90 + tk_45
        for pol, image in zip(polarizer_orientations, all_data_arr):
            if pol in [0, 90, 45, -45]:
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
        normalizing_S = (
            params[1] + 4 * params[2]
        )  # center of gaussian + 2sigma
        # 3sigma -> P = 2.7e-3 outliers
        # 4sigma -> P = 6.3e-5 outliers
        # ----------------------------------------------
    else:
        normalizing_S *= binning * binning  # account binning

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
            normalizing_S,
            splitted_occulter[i],
            polarizer_orientations,
            rad_micropol_phases_previsions,
            gain,
            DEBUG,
        ]
        for i in range(chunks_n_y * chunks_n_x)
    )

    starting_time = time.perf_counter()
    loc_time = time.strftime("%H:%M:%S  (%Y/%m/%d)", time.localtime())
    info(
        f"Starting parallel calculation ({procs_grid[0]}x{procs_grid[1]}) processors"
    )

    if procs_grid != [1, 1]:
        try:
            with mp.Pool(processes=chunks_n_y * chunks_n_x) as p:
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
        result = [[0.0, 0.0]] * chunks_n_y * chunks_n_x

        for i in range(chunks_n_y * chunks_n_x):
            result[i] = compute_demodulation_by_chunk(*arglist[i])

    loc_time = time.strftime("%H:%M:%S (%Y/%m/%d)", time.localtime())
    info(f"Ending parallel calculation")

    ending_time = time.perf_counter()
    info(f"Elapsed : {(ending_time - starting_time)/60:3.2f} mins")

    result = np.array(result)
    m_ij = np.zeros(
        shape=(N_MALUS_PARAMS, N_PIXELS_IN_SUPERPIX, height, width)
    )
    tks = np.zeros(shape=(height, width))
    efficiences = np.zeros(shape=(height, width))
    phases = np.zeros(shape=(height, width))

    for i in range(chunks_n_y):
        for j in range(chunks_n_x):
            m_ij[
                :,
                :,
                i * (chunk_size_y) : (i + 1) * chunk_size_y,
                j * (chunk_size_x) : (j + 1) * chunk_size_x,
            ] = result[i + chunks_n_y * j, 0].reshape(
                N_MALUS_PARAMS,
                N_PIXELS_IN_SUPERPIX,
                chunk_size_y,
                chunk_size_x,
            )
            tks[
                i * (chunk_size_y) : (i + 1) * chunk_size_y,
                j * (chunk_size_x) : (j + 1) * chunk_size_x,
            ] = result[i + chunks_n_y * j, 1].reshape(
                chunk_size_y, chunk_size_x
            )
            efficiences[
                i * (chunk_size_y) : (i + 1) * chunk_size_y,
                j * (chunk_size_x) : (j + 1) * chunk_size_x,
            ] = result[i + chunks_n_y * j, 2].reshape(
                chunk_size_y, chunk_size_x
            )
            phases[
                i * (chunk_size_y) : (i + 1) * chunk_size_y,
                j * (chunk_size_x) : (j + 1) * chunk_size_x,
            ] = result[i + chunks_n_y * j, 3].reshape(
                chunk_size_y, chunk_size_x
            )

    phases = np.rad2deg(phases)

    if DEBUG:
        # prevents overwriting
        sys.exit()

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for i in range(N_MALUS_PARAMS):
        for j in range(N_PIXELS_IN_SUPERPIX):
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


def compute_demodulation_by_chunk(
    splitted_dara_arr,
    normalizing_S,
    splitted_occulter_flag,
    polarizer_orientations,
    rad_micropol_phases_previsions,
    gain,
    DEBUG,
):
    """Utility function to parallelize calculations."""
    N_MALUS_PARAMS = 3
    N_PIXELS_IN_SUPERPIX = 4
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
        ]
    )
    theo_modulation_matrix = theo_modulation_matrix.T
    theo_demodulation_matrix = np.linalg.pinv(theo_modulation_matrix)

    num_of_points, height, width = splitted_dara_arr.shape
    rad_micropol_phases_previsions = np.array(rad_micropol_phases_previsions)
    polarizations_rad = np.deg2rad(polarizer_orientations)
    tk_prediction = 0.5
    efficiency_prediction = 0.4

    # Checked errors
    sigma_S2 = np.sqrt(0.5 * normalizing_S / gain)
    normalizing_S2 = normalizing_S * normalizing_S
    # pix_DN_sigma = np.sqrt(
    #    splitted_dara_arr / (gain * normalizing_S2)
    #    + sigma_S2
    #    * (splitted_dara_arr * splitted_dara_arr)
    #    / (normalizing_S2 * normalizing_S2)
    # )
    pix_DN_sigma = (
        np.sqrt(splitted_dara_arr / gain) / normalizing_S
    )  # poisson error on the photoelectrons

    normalized_splitted_data = splitted_dara_arr / normalizing_S
    all_zeros = np.zeros(shape=(num_of_points))

    m_ij = np.zeros(
        shape=(N_MALUS_PARAMS, N_PIXELS_IN_SUPERPIX, height, width)
    )  # demodulation matrix
    tk_data = np.ones(shape=(height, width)) * tk_prediction
    eff_data = np.ones(shape=(height, width)) * efficiency_prediction
    phase_data = np.zeros(shape=(height, width))
    superpix_params = np.zeros(shape=(N_PIXELS_IN_SUPERPIX, N_MALUS_PARAMS))

    predictions = np.zeros(shape=(N_PIXELS_IN_SUPERPIX, N_MALUS_PARAMS))
    predictions[:, 0] = tk_prediction  # Throughput prediction
    predictions[:, 1] = efficiency_prediction  # Efficiency prediction
    predictions[:, 2] = rad_micropol_phases_previsions  # Angle prediction

    bounds = np.zeros(shape=(N_PIXELS_IN_SUPERPIX, 2, N_MALUS_PARAMS))
    bounds[:, 0, 0], bounds[:, 1, 0] = 0.1, 0.9999999  # Throughput bounds

    # bounds[:, 0, 0], bounds[:, 1, 0] = (
    #    0.1,
    #    2.0,
    # )  # tk is multiplied by 0.5 so it can actually be > 1 and still physical
    bounds[:, 0, 1], bounds[:, 1, 1] = 0.1, 0.9999999  # Efficiency bounds
    bounds[:, 0, 2] = rad_micropol_phases_previsions - 15  # Lower angle bounds
    bounds[:, 1, 2] = rad_micropol_phases_previsions + 15  # Upper angle bounds

    # Fit for each superpixel. Use theoretical demodulation matrix for
    # occulter if present.
    if DEBUG:
        x_start, x_end = 100, 110
        y_start, y_end = 100, 110
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
                    splitted_occulter_flag[
                        super_y : super_y + 2, super_x : super_x + 2
                    ]
                )
            ):
                normalized_superpix_arr = normalized_splitted_data[
                    :, super_y : super_y + 2, super_x : super_x + 2
                ].reshape(num_of_points, N_PIXELS_IN_SUPERPIX)

                sigma_pix = pix_DN_sigma[
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

                if DEBUG:  # DEBUG
                    colors = ["blue", "orange", "green", "red"]
                    fig, ax = plt.subplots(
                        figsize=(9, 9), constrained_layout=True
                    )
                    for i in range(4):
                        ax.errorbar(
                            np.rad2deg(polarizations_rad),
                            normalized_superpix_arr[:, i],
                            yerr=sigma_pix[:, i],
                            xerr=[1.0] * len(polarizations_rad),
                            label=f"points {i}",
                            fmt="k-",
                            color=colors[i],
                        )
                        min = np.min(polarizations_rad)
                        # min = 225
                        max = np.max(polarizations_rad)
                        # max = 405
                        x = np.arange(min, max, (max - min) / 100)
                        ax.plot(
                            np.rad2deg(x),
                            Malus(x, *superpix_params[i]),
                            label=f"t = {superpix_params[i,0]:2.2f}, e = {superpix_params[i, 1]:2.2f}, phi = {np.rad2deg(superpix_params[i, 2]):2.2f}",
                            color=colors[i],
                        )
                        ax.set_title(
                            f"super_y = {super_y}, super_x = {super_x},"
                        )
                        ax.set_xlabel("Prepolarizer orientations [deg]")
                        ax.set_ylabel("signal / S")

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
                    print("---")

                # Remove matrices with big numbers
                if np.any(demodulation_matrix > 100) or np.any(
                    demodulation_matrix < -100
                ):
                    demodulation_matrix = theo_demodulation_matrix

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

            else:  # pixel is in occulter region
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
                ] = np.array(
                    [
                        [tk_prediction, tk_prediction],
                        [tk_prediction, tk_prediction],
                    ]
                )
                eff_data[
                    super_y : super_y + 2, super_x : super_x + 2
                ] = np.array(
                    [
                        [efficiency_prediction, efficiency_prediction],
                        [efficiency_prediction, efficiency_prediction],
                    ]
                )

    m_ij_chunk = m_ij

    return m_ij_chunk, tk_data, eff_data, phase_data


def Malus(angle, throughput, efficiency, phase):
    # original
    modulated_efficiency = efficiency * (
        np.cos(2.0 * phase) * np.cos(2.0 * angle)
        + np.sin(2.0 * phase) * np.sin(2.0 * angle)
    )
    return 0.5 * throughput * (1.0 + modulated_efficiency)
