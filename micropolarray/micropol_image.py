from __future__ import annotations

from dataclasses import dataclass
from logging import debug, error, info, warning
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from PIL import Image as PILImage

from micropolarray.cameras import Camera, PolarCam
from micropolarray.image import Image
from micropolarray.polarization_functions import AoLP, DoLP, pB
from micropolarray.processing.chen_wan_liang_calibration import ifov_jitcorrect
from micropolarray.processing.congrid import congrid
from micropolarray.processing.demodulation import Demodulator
from micropolarray.processing.demosaic import (
    demosaic,
    merge_polarizations,
    split_polarizations,
)
from micropolarray.processing.nrgf import roi_from_polar
from micropolarray.processing.rebin import micropolarray_rebin
from micropolarray.processing.shift import shift, shift_micropol
from micropolarray.utils import (
    fix_data,
    make_abs_and_create_dir,
    mean_minus_std,
    mean_plus_std,
    timer,
)


@dataclass
class PolParam:
    """Auxiliary class for polarization parameters.

    Members:
        ID (str): parameter identifier
        data (np.array): parameter image as numpy 2D array
        title (str): brief title of the parameter, useful for plotting
        measure_unit (str): initial measure units of the parameter
        fix_data (bool): controls whether data has to be constrained to [0, 4096] interval (not implemented yet)
    """

    ID: str
    data: np.ndarray
    title: str
    measure_unit: str
    fix_data: bool


DEFAULT_ANGLES_DIC = None  # sets the micropolarizer orientations with a dictionary {angle : position in superpix 1->3}


def set_default_angles(angles_dic: dict):
    """Sets the default micropolarizer orientations for images.

    Args:
        angles_dic (dict): dictionary {value : pos} where value is the angle in degrees from -90 to 90 and pos is the pixel position in superpixel, from 0 to 3 (position [y, x], fast index x)
    """
    global DEFAULT_ANGLES_DIC
    DEFAULT_ANGLES_DIC = angles_dic


class MicropolImage(Image):
    """Micro-polarizer array image class. Can be initialized from a 2d array, a list of 1 or more file names (use the boolean keyword averageimages to select if sum or average is taken) or another MicropolImage. Dark and flat micropolarray images can also be provided to automatically correct the result."""

    first_call = True  # Avoid repeating messages

    def __init__(
        self,
        initializer: str | np.ndarray | list | MicropolImage,
        angle_dic: dict = None,
        demosaic_mode: str = "adjacent",
        dark: MicropolImage = None,
        flat: MicropolImage = None,
        averageimages: bool = True,
    ):
        self._is_demodulated = False
        self._is_demosaiced = False
        self._binning = 1
        self._flat_subtracted = False
        self._dark_subtracted = False
        if angle_dic is None:
            global DEFAULT_ANGLES_DIC
            if DEFAULT_ANGLES_DIC is None:
                if MicropolImage.first_call:
                    warning(
                        f"Micropolarizer orientation dictionary defaults to {PolarCam().angle_dic}, set it via set_default_angles(camera)\n"
                    )
                MicropolImage.first_call = False
                DEFAULT_ANGLES_DIC = PolarCam().angle_dic
            angle_dic = DEFAULT_ANGLES_DIC
        self.angle_dic = angle_dic
        self.demosaic_mode = demosaic_mode
        self.demosaiced_images = None

        if type(initializer) is str and len(initializer) > 1:
            self._num_of_images = len(initializer)
        else:
            self._num_of_images = 1
        super().__init__(
            initializer=initializer, averageimages=averageimages
        )  # Call generic Image() constructor

        if (type(initializer) is list) or (type(initializer) is str):
            self._init_micropolimage_from_file(initializer)
        elif type(initializer) is np.ndarray:
            self._init_micropolimage_from_data(initializer)
        elif type(initializer) is MicropolImage:
            self._init_micropolimage_from_image(initializer)

        self._update_stokes_derived_internal_dataclasses()

        # Apply corrections if needed
        if dark is not None:
            self.subtract_dark(dark=dark)
        if flat is not None:
            self.correct_flat(flat=flat)
        elif MicropolImage.first_call:
            warning("Remember to set dark")
            MicropolImage.first_call = False

        self.height, self.width = self.data.shape

    def _init_micropolimage_from_file(self, filenames: str):
        self._set_data_and_Stokes(self.data)

    def _init_micropolimage_from_data(self, data: np.array):
        self._set_data_and_Stokes(self.data)
        if self.header is None:
            self.header = self.set_default_header(data)
        else:
            self._update_dims_in_header(self.data)

    def _init_micropolimage_from_image(self, image: MicropolImage):
        self._is_demodulated = image._is_demodulated
        self._is_demosaiced = image._is_demosaiced
        self._binning = image._binning
        self._dark_subtracted = image._dark_subtracted
        self._flat_subtracted = image._flat_subtracted
        self.angle_dic = image.angle_dic
        self.demosaic_mode = image.demosaic_mode

        self.data = image.data
        self.header = image.header
        self.single_pol_subimages = image.single_pol_subimages
        self.demosaiced_images = image.demosaiced_images
        self.Stokes_vec = image.Stokes_vec
        self.polparam_list = image.polparam_list

    # ----------------------------------------------------------------
    # ---------------------- STOKES COMPONENTS -----------------------
    # ----------------------------------------------------------------

    def _update_Stokes_vec(self) -> None:
        if not self._is_demosaiced:
            self.Stokes_vec = self._get_theo_Stokes_vec_components(
                self.single_pol_subimages
            )
        else:
            self.Stokes_vec = self._get_theo_Stokes_vec_components(
                self.demosaiced_images
            )

    def _update_stokes_derived_internal_dataclasses(self) -> None:
        self.I = PolParam(
            "I", self.Stokes_vec[0], "Stokes I", "DN", fix_data=False
        )
        self.Q = PolParam(
            "Q", self.Stokes_vec[1], "Stokes Q", "DN", fix_data=False
        )
        self.U = PolParam(
            "U", self.Stokes_vec[2], "Stokes U", "DN", fix_data=False
        )
        self.pB = PolParam(
            "pB",
            pB(self.Stokes_vec),
            "Polarized brightness",
            "DN",
            fix_data=False,
        )
        self.AoLP = PolParam(
            "AoLP",
            AoLP(self.Stokes_vec),
            "Angle of linear polarization",
            "rad",
            fix_data=False,
        )
        self.DoLP = PolParam(
            "DoLP",
            DoLP(self.Stokes_vec),
            "Degree of linear polarization",
            "",
            fix_data=False,
        )
        self.polparam_list = [
            self.I,
            self.Q,
            self.U,
            self.pB,
            self.AoLP,
            self.DoLP,
        ]

    def demodulate(self, demodulator: Demodulator) -> MicropolImage:
        """Returns a MicropolImage with polarization parameters calculated from the demodulation tensor provided.

        Args:
            demodulator (Demodulator): Demodulator object containing the demodulation tensor components (see processing.new_demodulation)

        Raises:
            ValueError: raised if image and demodulator do not have the same dimension, for example in case of different binning

        Returns:
            MicropolImage: copy of the input image with I, Q, U, pB, DoLP, AoLP calculated from the demodulation tensor.
        """
        if (self.height, self.width) != (
            demodulator.mij.shape[2],
            demodulator.mij.shape[3],
        ):
            print(demodulator.mij.shape[2], demodulator.mij.shape[3])
            print(self.height, self.width)
            raise ValueError(
                "Image and demodulator do not have the same dimensions, check binning."
            )
        info("Demodulating...")
        demodulated_image = MicropolImage(self)
        demodulated_image.Stokes_vec = (
            demodulated_image._get_Stokes_from_demodulator(demodulator)
        )
        demodulated_image._update_single_pol_subimages()
        demodulated_image._update_stokes_derived_internal_dataclasses()
        demodulated_image._is_demodulated = True

        info("Image correctly demodulated")
        return demodulated_image

    def set_data_only(self, data: np.array = None) -> None:
        if data is None:
            data = self.data
        self.data = data
        self.height = data.shape[0]
        self.width = data.shape[1]
        if (data.shape[0] % 2) or (data.shape[1] % 2):
            warning(
                "Odd number of pixels is incompatible"
                " with micropolarizer arrays operations."
            )
        self._update_single_pol_subimages()
        if self._is_demosaiced:
            self.demosaic()
        self._update_Stokes_vec()
        self._update_stokes_derived_internal_dataclasses()
        if self.header is None:
            self.header = self.set_default_header(data)
        else:
            self._update_dims_in_header(self.data)

    def _set_data_and_Stokes(self, data: np.array = None) -> None:
        """Set image data and derived polarization informations, and
        consequently change header."""
        if data is None:
            data = self.data
        self.data = data
        if (data.shape[0] % 2) or (data.shape[1] % 2):
            warning(
                "Odd number of pixels is incompatible"
                " with micropolarizer arrays operations."
            )
        self.height, self.width = data.shape
        self._update_single_pol_subimages()

        if not self._is_demosaiced:
            self.Stokes_vec = self._get_theo_Stokes_vec_components(
                self.single_pol_subimages
            )
            self._update_stokes_derived_internal_dataclasses()
        else:
            self.demosaic()
            self.Stokes_vec = self._get_theo_Stokes_vec_components(
                self.demosaiced_images
            )

    def _update_single_pol_subimages(self) -> None:
        single_pol_subimages = split_polarizations(self.data)

        self.single_pol_subimages = single_pol_subimages
        self.pol0 = PolParam(
            "0",
            single_pol_subimages[self.angle_dic[0]],
            "0 deg orientation pixels",
            "DN",
            fix_data=False,
        )
        self.pol45 = PolParam(
            "45",
            single_pol_subimages[self.angle_dic[45]],
            "45 deg orientation pixels",
            "DN",
            fix_data=False,
        )
        self.pol_45 = PolParam(
            "-45",
            single_pol_subimages[self.angle_dic[-45]],
            "-45 deg orientation pixels",
            "DN",
            fix_data=False,
        )
        self.pol90 = PolParam(
            "90",
            single_pol_subimages[self.angle_dic[90]],
            "90 deg orientation pixels",
            "DN",
            fix_data=False,
        )

    def _get_theo_Stokes_vec_components(self, single_pol_images) -> np.array:
        """
        Computes stokes vector components from four polarized images at four angles, angle_dic describes the coupling between
        poled_images_array[i] <--> angle_dic[i]
        Return:
            stokes vector, shape=(3, poled_images.shape[1], poled_images.shape[0])
        """
        I = 0.5 * np.sum(single_pol_images, axis=0)
        Q = (
            single_pol_images[self.angle_dic[0]]
            - single_pol_images[self.angle_dic[90]]
        )
        U = (
            single_pol_images[self.angle_dic[45]]
            - single_pol_images[self.angle_dic[-45]]
        )

        S = np.array([I, Q, U], dtype=float)
        return S

    def _get_Stokes_from_demodulator(
        self, demodulator: Demodulator
    ) -> np.array:
        """
        Computes stokes vector components from four polarized images at four angles, angle_dic describes the coupling between
        poled_images_array[i] <--> angle_dic[i]. Assumes:

        I = M_00 * I_1 + M_01 * I_2 + M_02 * I_3 + M_03 * I_4
        Q = M_10 * I_1 + M_11 * I_2 + M_12 * I_3 + M_13 * I_4
        U = M_20 * I_1 + M_21 * I_2 + M_22 * I_3 + M_23 * I_4

        Return:
            stokes vector, shape=(3, poled_images.shape[1], poled_images.shape[0])
        """
        # Corrected with demodulation matrices, S.shape = (4, n, m)
        num_of_malus_parameters = 3  # 3 multiplication params
        pixels_in_superpix = 4
        mij = demodulator.mij

        self.demosaic()
        demosaiced_images = self.demosaiced_images
        # demosaiced_images = demosaic(self.data, self.demosaic_mode)

        # IMG = np.array(
        #    [
        #        demosaiced_images[self.angle_dic[0]],
        #        demosaiced_images[self.angle_dic[45]],
        #        demosaiced_images[self.angle_dic[-45]],
        #        demosaiced_images[self.angle_dic[90]],
        #    ],
        #    dtype=float,
        # )  # Liberatore article/thesis

        IMG = np.array(
            [demo_image for demo_image in demosaiced_images],
            dtype=float,
        )
        if (mij[0, 0].shape[0] != IMG[0].shape[0]) or (
            mij[0, 0].shape[1] != IMG[0].shape[1]
        ):
            raise ValueError(
                f"demodulation matrix and demosaiced images have different shape {mij[0,0].shape} {IMG[0].shape}. Check that binning is correct."
            )  # sanity check

        T_ij = np.zeros(
            shape=(
                num_of_malus_parameters,
                pixels_in_superpix,
                *self.data.shape,
            )
        )
        for i in range(num_of_malus_parameters):
            for j in range(pixels_in_superpix):
                temp_tij = np.multiply(
                    demodulator.mij[i, j, :, :], IMG[j, :, :]
                )  # Matrix product
                T_ij[i, j, :, :] = temp_tij

        I = T_ij[0, 0] + T_ij[0, 1] + T_ij[0, 2] + T_ij[0, 3]
        Q = T_ij[1, 0] + T_ij[1, 1] + T_ij[1, 2] + T_ij[1, 3]
        U = T_ij[2, 0] + T_ij[2, 1] + T_ij[2, 2] + T_ij[2, 3]

        S = np.array([I, Q, U], dtype=float)
        return S

    def subtract_dark(self, dark: MicropolImage) -> MicropolImage:
        """Correctly subtracts the input dark image from the image

        Args:
            dark (MicropolImage): dark to subtract

        Returns:
            MicropolImage: copy of input image with dark subtracted
        """
        self.data = self.data - dark.data
        self.data = np.where(self.data >= 0, self.data, 0)  # Fix data
        self._set_data_and_Stokes()
        self._dark_subtracted = True
        return self

    def correct_flat(self, flat: MicropolImage) -> MicropolImage:
        """Normalizes the flat and uses it to correct the image.

        Args:
            flat (MicropolImage): flat image, does not need to be normalized.

        Returns:
            MicropolImage: copy of input image corrected by flat
        """
        normalized_flat = flat.data / np.max(flat.data)
        self.data = np.where(
            normalized_flat != 0, self.data / normalized_flat, self.data
        )
        # self.data = np.where(self.data >= 0, self.data, 0)
        # self.data = np.where(self.data < 4096, self.data, 4096)
        self._set_data_and_Stokes()
        self._flat_subtracted = True
        return self

    def correct_ifov(self) -> MicropolImage:
        """Corrects differences in single pixels fields of view inside each superpixel

        Returns:
            MicropolImage: image with data corrected for field of view differences
        """
        corrected_data = self.data.copy()
        corrected_data = ifov_jitcorrect(self.data, self.height, self.width)
        self._set_data_and_Stokes(corrected_data)
        return self

    # ----------------------------------------------------------------
    # ------------------------------ SHOW ----------------------------
    # ----------------------------------------------------------------

    def show_with_pol_params(self, cmap="Greys_r") -> tuple:
        """Returns a fig for each set of image parameters. User must call
        plt.show after this is called.
        Returned parameters:
        - Original image
        - Stokes vector I, Q, U
        - Angle, degree of linear polarizaimagetion Polarized brightness


        Args:
            cmap (str, optional): colormap string. Defaults to "Greys_r".

        Returns:
            tuple: a (figure, axis) couple same as matplotlib.pyplot.subplots for the image data and another for the six polarization parameters
        """
        data_ratio = self.data.shape[0] / self.data.shape[1]
        image_fig, imageax = plt.subplots(dpi=200, constrained_layout=True)
        mappable = imageax.imshow(
            self.data,
            cmap=cmap,
            vmin=mean_minus_std(self.data),
            vmax=mean_plus_std(self.data),
        )
        avg = np.mean(self.data)
        stdev = np.std(self.data)
        imageax.set_title(
            f"Image data (avrg {avg:3.2f}+-{stdev:3.2f})",
            color="black",
        )
        imageax.set_xlabel("x [px]")
        imageax.set_ylabel("y [px]")
        image_fig.colorbar(
            mappable, ax=imageax, label="[DN]", fraction=data_ratio * 0.05
        )
        stokes_fig, stokesax = plt.subplots(
            2, 3, figsize=(14, 9), constrained_layout=True
        )

        stokesax = stokesax.ravel()
        for i, stokes in enumerate(stokesax):
            vmin = mean_minus_std(self.polparam_list[i].data)
            vmax = mean_plus_std(self.polparam_list[i].data)
            avg = np.mean(self.polparam_list[i].data)
            stdev = np.std(self.polparam_list[i].data)
            mappable_stokes = stokes.imshow(
                self.polparam_list[i].data, cmap=cmap, vmin=vmin, vmax=vmax
            )
            stokes.set_title(
                self.polparam_list[i].title
                + f" (avrg {avg:3.2f}+-{stdev:3.2f})",
                color="black",
            )
            stokes.set_xlabel("x [px]")
            stokes.set_ylabel("y [px]")
            stokes_fig.colorbar(
                mappable_stokes,
                ax=stokes,
                label=self.polparam_list[i].measure_unit,
                fraction=data_ratio * 0.05,
            )
        return image_fig, imageax, stokes_fig, stokesax

    def show_single_pol_images(self, cmap="Greys_r"):
        """Plots the four polarizations images.

        Args:
            cmap (str, optional): colormap for the plot. Defaults to "Greys_r".

        Returns:
            tuple: a (figure, axis) couple same as matplotlib.pyplot.subplots
        """
        data_ratio = self.data.shape[0] / self.data.shape[1]
        fig, ax = plt.subplots(2, 2, figsize=(9, 9), constrained_layout=True)
        ax = ax.ravel()
        polslist = [self.pol0, self.pol45, self.pol90, self.pol_45]
        for i, singlepolax in enumerate(ax):
            mappable = singlepolax.imshow(polslist[i].data, cmap=cmap)
            singlepolax.set_title(polslist[i].title)
            singlepolax.set_xlabel("x [px]")
            singlepolax.set_ylabel("y [px]")
            fig.colorbar(
                mappable,
                ax=singlepolax,
                label=polslist[i].measure_unit,
                fraction=data_ratio * 0.05,
                pad=0.01,
            )
        return fig, ax

    def show_demo_images(self, cmap="Greys_r"):
        """Plots the four demosaiced images.

        Args:
            cmap (str, optional): colormap for the plot. Defaults to "Greys_r".

        Returns:
            tuple: a (figure, axis) couple same as matplotlib.pyplot.subplots
        """
        if not self._is_demosaiced:
            error("Image is not yet demosaiced.")
        data_ratio = self.data.shape[0] / self.data.shape[1]
        fig, ax = plt.subplots(2, 2, figsize=(9, 9), constrained_layout=True)
        ax = ax.ravel()
        demo_images_list = self.demosaiced_images
        for i, single_demo_ax in enumerate(ax):
            mappable = single_demo_ax.imshow(
                demo_images_list[i],
                cmap=cmap,
                vmin=mean_minus_std(demo_images_list[i]),
                vmax=mean_plus_std(demo_images_list[i]),
            )
            single_demo_ax.set_title(
                f"Demosaiced image {list(self.angle_dic.keys())[list(self.angle_dic.values()).index(i)]}"
            )
            single_demo_ax.set_xlabel("x [px]")
            single_demo_ax.set_ylabel("y [px]")
            fig.colorbar(
                mappable,
                ax=single_demo_ax,
                label="DN",
                fraction=data_ratio * 0.05,
                pad=0.01,
            )
        return fig, ax

    def show_pol_param(self, polparam: PolParam, cmap="Greys_r"):
        """Plots a single polarization parameter given as input

        Args:
            polparam (PolParam): image PolParam containing the parameter to plot. Can be one among [self.I, self.Q, self.U, self.pB, self.AoLP, self.DoLP]
            cmap (str, optional): colormap for the plot. Defaults to "Greys_r".

        Returns:
            tuple: a (figure, axis) couple same as matplotlib.pyplot.subplots
        """
        data_ratio = self.data.shape[0] / self.data.shape[1]
        fig, ax = plt.subplots(dpi=200)
        mappable = ax.imshow(
            polparam.data,
            cmap=cmap,
            vmin=mean_minus_std(polparam.data),
            vmax=mean_plus_std(polparam.data),
        )
        ax.set_title(polparam.title)
        ax.set_xlabel("x [px]")
        ax.set_ylabel("y [px]")
        fig.colorbar(
            mappable,
            ax=ax,
            label=polparam.measure_unit,
            fraction=data_ratio * 0.05,
        )
        return fig, ax

    # ----------------------------------------------------------------
    # -------------------------- SAVING ------------------------------
    # ----------------------------------------------------------------

    def save_single_pol_images(
        self, filename: str, fixto: list[float, float] = None
    ) -> None:
        """Saves the four polarized images as fits files

        Args:
            filename (str): filename of the output image. The four images will be saved as filename_POLXX.fits
            fixto (list[float, float], optional): set the minimum and maximum value for the output images. Defaults to None.

        Raises:
            ValueError: an invalid file name is provided
        """
        polslist = [self.pol0, self.pol45, self.pol90, self.pol_45]
        filepath = Path(make_abs_and_create_dir(filename))
        if filepath.suffix != ".fits":
            raise ValueError("filename must be a valid file name, not folder.")
        group_filepath = filepath.joinpath(filepath.parent, filepath.stem)
        for single_pol in polslist:
            hdr = self.header.copy()
            hdr["POL"] = (single_pol.ID, "Micropolarizer orientation")
            if fixto:
                data = fix_data(single_pol.data, *fixto)
            else:
                data = single_pol.data
            hdu = fits.PrimaryHDU(
                data=data,
                header=hdr,
                do_not_scale_image_data=True,
                uint=False,
            )
            filename_with_ID = str(
                group_filepath.joinpath(
                    str(group_filepath) + "POL" + str(single_pol.ID) + ".fits"
                )
            )
            hdu.writeto(filename_with_ID, overwrite=True)
        info(f'All params successfully saved to "{filename}"')

    def save_param_as_fits(
        self,
        filename: str,
        polparam: PolParam,
        fixto: list[float, float] = None,
    ) -> None:
        """Saves chosen polarization parameter as a fits file

        Args:
            filename (str): filename of the output image.
            polparam (PolParam): polarization parameter to save. Can be one among [self.I, self.Q, self.U, self.pB, self.AoLP, self.DoLP]
            fixto (list[float, float], optional): set the minimum and maximum value for the output images. Defaults to None.

        Raises:
            ValueError: filename is not a valid .fits file
        """
        filepath = Path(make_abs_and_create_dir(filename))
        if filepath.suffix != ".fits":
            raise ValueError("filename must be a valid file name, not folder.")
        hdr = self.header.copy()
        hdr["PARAM"] = (str(polparam.title), "Polarization parameter")
        hdr["UNITS"] = (str(polparam.measure_unit), "Measure units")
        if fixto:
            data = fix_data(polparam.data, *fixto)
        else:
            data = polparam.data
        hdu = fits.PrimaryHDU(
            data=data,
            header=hdr,
            do_not_scale_image_data=True,
            uint=False,
        )
        filename_with_ID = str(
            filepath.joinpath(
                filepath.parent, filepath.stem + "_" + polparam.ID + ".fits"
            )
        )

        # filename = make_abs_and_create_dir(filename)
        # filename_with_ID = (
        #    filename.split(".")[-2] + "_" + polparam.ID + ".fits"
        # )
        hdu.writeto(filename_with_ID, overwrite=True)
        info(f'"{filename_with_ID}" {polparam.ID} successfully saved')

    def save_all_pol_params_as_fits(self, filename: str) -> None:
        """Saves the image and all polarization parameters as fits file with the same name

        Args:
            filename (str): filename of the output image. Will be saved as filename_[I, Q, U, pB, AoLP, DoLP].fits

        Raises:
            ValueError: filename is not a valid .fits file
        """
        filepath = Path(filename)
        if filepath.suffix != ".fits":
            raise ValueError("filename must be a valid file name, not folder.")
        filepath = Path(make_abs_and_create_dir(filename))
        group_filename = str(filepath.joinpath(filepath.parent, filepath.stem))
        for param in self.polparam_list:
            hdr = self.header.copy()
            hdr["PARAM"] = (str(param.title), "Polarization parameter")
            hdr["UNITS"] = (str(param.measure_unit), "Measure units")
            if param.fix_data:
                data = fix_data(param.data)
            else:
                data = param.data
            hdu = fits.PrimaryHDU(
                data=data,
                header=hdr,
                do_not_scale_image_data=True,
                uint=False,
            )
            filename_with_ID = group_filename + "_" + param.ID + ".fits"
            hdu.writeto(filename_with_ID, overwrite=True)
        info(f'All params successfully saved to "{group_filename}"')

    def save_demosaiced_images_as_fits(
        self, filename: str, fixto: list[float, float] = None
    ) -> None:
        """Saves the four demosaiced images as fits files

        Args:
            filename (str): filename of the output image. The four images will be saved as filename_POLXX.fits
            fixto (list[float, float], optional): set the minimum and maximum value for the output images. Defaults to None.

        Raises:
            ValueError: an invalid file name is provided
        """
        if not self._is_demosaiced:
            raise ValueError("Demosaiced images not yet calculated.")
        imageHdr = self.header.copy()
        filepath = Path(filename)
        if not filepath.suffix:
            raise ValueError("filename must be a valid file name, not folder.")
        filepath = Path(make_abs_and_create_dir(filename))
        group_filename = str(filepath.joinpath(filepath.parent, filepath.stem))
        for i, demo_image in enumerate(self.demosaiced_images):
            POL_ID = list(self.angle_dic.keys())[
                list(self.angle_dic.values()).index(i)
            ]
            imageHdr["POL"] = (int(POL_ID), "Micropolarizer orientation")
            if fixto:
                data = fix_data(demo_image, *fixto)
            else:
                data = demo_image
            hdu = fits.PrimaryHDU(
                data=data,
                header=imageHdr,
                do_not_scale_image_data=True,
                uint=False,
            )
            new_filename = group_filename + "_POL" + str(POL_ID) + ".fits"
            hdu.writeto(new_filename, overwrite=True)
        info(
            f'Demosaiced images successfully saved to "{group_filename}_POLX.fits"'
        )

    # ----------------------------------------------------------------
    # -------------------- DATA MANIPULATION -------------------------
    # ----------------------------------------------------------------
    def demosaic(self) -> MicropolImage:
        """Returns a demosaiced copy of the image with updated polarization parameters.

        Returns:
            MicropolImage: demosaiced image
        """
        self.demosaiced_images = demosaic(self.data, option=self.demosaic_mode)
        self.Stokes_vec = self._get_theo_Stokes_vec_components(
            self.demosaiced_images
        )
        self._update_stokes_derived_internal_dataclasses()
        self._is_demosaiced = True

        return self

    def rebin(self, binning: int) -> MicropolImage:
        """Rebins the micropolarizer array image, binned each
        binningxbinning. Sum bins by default.

        Args:
            binning (int): binning to perform. A value of n will be translated in a nxn binning.

        Raises:
            ValueError: negative binning provided

        Returns:
            MicropolImage: copy of the input image, rebinned.
        """
        if binning <= 0:
            raise ValueError(f"Negative binning {binning}x{binning}")
        rebinned_image = MicropolImage(self)
        rebinned_data = micropolarray_rebin(
            np.array(rebinned_image.data, dtype=np.double),
            *rebinned_image.data.shape,
            binning,
        )

        rebinned_image._set_data_and_Stokes(rebinned_data)

        return rebinned_image

    def congrid(self, newdim_y: int, newdim_x: int) -> MicropolImage:
        # Trim to nearest superpixel
        if (newdim_y % 2) or (newdim_x % 2):
            while newdim_y % 2:
                newdim_y = newdim_y - 1
            while newdim_x % 2:
                newdim_x = newdim_x - 1
            warning(
                f"New dimension was incompatible with superpixels. Trimmed to ({newdim_y}, {newdim_x})"
            )

        new_subdims = [int(newdim_y / 2), int(newdim_x / 2)]
        congridded_pol_images = [0.0] * 4
        for subimage_i, pol_subimage in enumerate(self.single_pol_subimages):
            congridded_pol_images[subimage_i] = congrid(
                pol_subimage, new_subdims
            )

        newdata = np.zeros(shape=(newdim_y, newdim_x))
        for pol_super_y in range(new_subdims[0]):
            for pol_super_x in range(new_subdims[1]):
                newdata[
                    pol_super_y * 2 : pol_super_y * 2 + 2,
                    pol_super_x * 2 : pol_super_x * 2 + 2,
                ] = [
                    [
                        congridded_pol_images[0][pol_super_y, pol_super_x],
                        congridded_pol_images[1][pol_super_y, pol_super_x],
                    ],
                    [
                        congridded_pol_images[2][pol_super_y, pol_super_x],
                        congridded_pol_images[3][pol_super_y, pol_super_x],
                    ],
                ]
        newimage = MicropolImage(newdata)
        return newimage

    def rotate(self, angle: float) -> MicropolImage:
        """Rotates an image of angle degrees, counter-clockwise."""

        single_pols = split_polarizations(self.data)
        for i in range(4):
            image = PILImage.fromarray(single_pols[i])
            image = image.rotate(angle)
            single_pols[i] = np.asarray(image)
        data = merge_polarizations(single_pols)

        return MicropolImage(data)

    def mask_occulter(
        self,
        y: int = PolarCam().occulter_pos_last[0],
        x: int = PolarCam().occulter_pos_last[1],
        r: int = PolarCam().occulter_pos_last[2],
        overoccult: int = 0,
        camera=PolarCam(),
    ) -> None:
        """Masks occulter for all image parameters

        Args:
            y (int, optional): Occulter y position. Defaults to PolarCam().occulter_pos_last[0].
            x (int, optional): Occulter x position. Defaults to PolarCam().occulter_pos_last[1].
            r (int, optional): Occulter radius. Defaults to PolarCam().occulter_pos_last[2].
            overoccult (int, optional): Pixels to overoccult. Defaults to 0.
            camera (_type_, optional): Camera image type. Defaults to PolarCam().

        Returns:
            None
        """
        # y, x, r = camera.occulter_pos_last

        relative_y = y / camera.h_image
        relative_x = x / camera.w_image
        relative_r = (r + overoccult) / camera.h_image

        abs_y = int(relative_y * self.height)
        abs_x = int(relative_x * self.width)
        abs_r = int(relative_r * self.height)

        self.data = roi_from_polar(
            self.data,
            (abs_y, abs_x),
            [abs_r, 2 * np.max((self.height, self.width))],
        )
        self.single_pol_subimages = [
            roi_from_polar(
                data,
                (int(abs_y / 2), int(abs_x / 2)),
                [
                    int(abs_r / 2),
                    2 * np.max((int(self.height / 2), int(self.width / 2))),
                ],
            )
            for data in self.single_pol_subimages
        ]
        self._update_single_pol_subimages()  # Updates polparam-like
        if self._is_demosaiced:
            self.demosaiced_images = [
                roi_from_polar(
                    data,
                    (abs_y, abs_x),
                    (abs_r, 2 * np.max([self.height, self.width])),
                )
                for data in self.demosaiced_images
            ]
        for param in self.polparam_list:
            param.data = roi_from_polar(
                param.data,
                (
                    int(relative_y * param.data.shape[0]),
                    int(relative_x * param.data.shape[1]),
                ),
                [
                    int(
                        relative_r
                        * (param.data.shape[0] + param.data.shape[1])
                        / 2
                    ),
                    2 * np.max((param.data.shape[0], param.data.shape[0])),
                ],
            )

    def shift(self, y: int, x: int) -> MicropolImage:
        """Shifts image by y, x pixels and fills with 0 the remaining space. Positive numbers for up/right shift and negative for down/left shift. Image is split into polarizations, each one is shifted, then they are merged again.

        Args:
            y (int): vertical shift in pix
            x (int): horizontal shift in pix

        Returns:
            MicropolImage: shifted image copied from the original
        """
        # newdata = shift(self.data, y, x)
        newdata = shift_micropol(self.data, y, x)
        newimage = MicropolImage(self)
        newimage._set_data_and_Stokes(newdata)

        return newimage

    # ----------------------------------------------------------------
    # ------------------------ OVERLOADING ---------------------------
    # ----------------------------------------------------------------
    def __add__(self, second) -> MicropolImage:
        if type(self) is type(second):
            newdata = self.data + second.data
            newimage = MicropolImage(self)
            newimage._set_data_and_Stokes(newdata)
            return newimage
        else:
            newdata = self.data + second
            return MicropolImage(newdata, angle_dic=self.angle_dic)

    def __sub__(self, second) -> MicropolImage:
        if type(self) is type(second):
            newdata = self.data - second.data
            newimage = MicropolImage(self)
            newimage._set_data_and_Stokes(newdata)
            return newimage
        else:
            newdata = self.data - second
            return MicropolImage(newdata, angle_dic=self.angle_dic)

    def __mul__(self, second) -> MicropolImage:
        if type(self) is type(second):
            newdata = self.data * second.data
            newimage = MicropolImage(self)
            newimage._set_data_and_Stokes(newdata)
            return newimage
        else:
            newdata = self.data * second
            return MicropolImage(newdata, angle_dic=self.angle_dic)

    def __truediv__(self, second) -> MicropolImage:
        if type(self) is type(second):
            newdata = np.where(second.data != 0, self.data / second.data, 4096)
            newimage = MicropolImage(self)
            newimage._set_data_and_Stokes(newdata)
            return newimage
        else:
            newdata = np.where(second != 0, self.data / second, 4096)
            return MicropolImage(newdata, angle_dic=self.angle_dic)


# provide shorter aliases
# PolarcamImage = MicropolImage
# MicropolImage = MicropolImage
