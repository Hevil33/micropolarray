from __future__ import annotations
from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from micropolarray_lib.utils import make_abs_and_create_dir
from logging import warning, info
from micropolarray_lib.utils import make_abs_and_create_dir, fix_data
import datetime
from pathlib import Path


class Image:
    """Basic image class. Can be initialized from a filename, a filenames list, a numpy array or another Image instance. If multiple filenames are provided, will perform a mean over them."""

    def __init__(
        self,
        initializer: str | np.ndarray | Image,
    ):
        self.header = None
        if type(initializer) is str or type(initializer) is list:
            self._init_image_from_file(initializer)
        elif type(initializer) is np.ndarray:
            self._init_image_from_data(initializer)
        elif type(initializer) is Image:
            self._init_image_from_image(initializer)

    def _init_image_from_data(self, data: np.array) -> None:
        self._set_data(np.array(data))
        self.filename = "image.fits"

    def _init_image_from_file(self, filenames) -> None:
        filenames_list = (
            [filenames] if type(filenames) is not list else filenames
        )
        filenames_len = len(filenames_list)
        if filenames_len == 0:
            raise NameError("Can't load files, empty filenames list.")
        datetimes = [0] * filenames_len

        firstcall = True
        for idx, filename in enumerate(filenames_list):
            with fits.open(filename) as hul:
                hul.verify("fix")

                try:  # standard format
                    datetimes[idx] = datetime.datetime.strptime(
                        hul[0].header["DATE-OBS"],
                        "%Y-%m-%dT%H:%M:%S",
                    )
                except ValueError:  # antarticor format
                    datetimes[idx] = datetime.datetime.strptime(
                        hul[0].header["DATE-OBS"]
                        + "-"
                        + hul[0].header["TIME-OBS"],
                        "%Y-%m-%d-%H:%M:%S",
                    )
                except KeyError:
                    pass
                if idx == 0:
                    mean_data = hul[0].data
                    self.header = hul[0].header
                else:
                    if firstcall:
                        info(f"Averaging {filenames_len} images...")
                        firstcall = False
                    mean_data = mean_data + hul[0].data
        datetimes = [datetime for datetime in datetimes if datetime != 0]
        if len(datetimes) == 0:
            datetimes = [0]
        mean_data = mean_data / filenames_len
        self._set_data(np.array(mean_data))

        if filenames_len > 1:
            self.header["AVGFROM"] = (
                filenames_len,
                "Number of files the images is the average result of.",
            )
            datetimes = sorted(datetimes)
            if not datetimes[0] == 0:
                self.header["STRT-OBS"] = (
                    str(datetimes[0]),
                    "Date and time of first image.",
                )
                self.header["END-OBS"] = (
                    str(datetimes[-1]),
                    "Date and time of last image.",
                )
            tempfilename = filenames[0].split(os.path.sep)
            tempfilename[-1] = "MEAN_" + tempfilename[-1]
            self.filename = os.path.sep.join(tempfilename)
        else:
            self.filename = filenames[0]

    def _init_image_from_image(self, image: Image):
        self._set_data(image.data.copy())
        self.filename = image.filename

    def _set_data(self, data: np.array):
        """Set image data and derived polarization informations, and
        consequently change header."""
        fixed_data = data
        # fixed_data = np.where(fixed_data > -4096, fixed_data, -4096)
        # fixed_data = np.where(fixed_data < 4096, fixed_data, 4096)
        self.data = fixed_data
        self.height, self.width = data.shape
        if self.header is None:
            self.header = self._set_default_header(data)
        else:
            self._update_dims_in_header(self.data)

    def _set_default_header(self, data: np.array):
        """Set a default header when initializing image with data instead of
        .fits filename."""
        header = fits.PrimaryHDU(data=data).header
        return header

    def _update_dims_in_header(self, data: np.array):
        self.header["NAXIS1"] = data.shape[1]
        self.header["NAXIS2"] = data.shape[0]

    # ----------------------------------------------------------------
    # -------------------------- SAVING ------------------------------
    # ----------------------------------------------------------------

    def save_as_fits(
        self, filename: str, fixto: str[float, float] = [0, 4096]
    ):
        """Save image as fits file with current instance header."""
        filepath = Path(filename)
        if not filepath.suffix:
            raise ValueError("filename must be a valid file name, not folder.")
        filepath = Path(make_abs_and_create_dir(filename))
        if fixto:
            data = fix_data(self.data, *fixto)
        else:
            data = self.data
        hdu = fits.PrimaryHDU(
            data=data,
            header=self.header,
            do_not_scale_image_data=True,
            uint=False,
        )
        filename = str(
            filepath.joinpath(filepath.parent, filepath.stem + ".fits")
        )
        hdu.writeto(filename, overwrite=True)
        info(f'Image successfully saved to "{filename}".')

    # ----------------------------------------------------------------
    # ------------------------------ SHOW ----------------------------
    # ----------------------------------------------------------------

    def show(self, cmap="Greys_r", vmin=None, vmax=None):
        """Shows image data with colorbar."""
        data_to_plot = self.data
        data_ratio = data_to_plot.shape[0] / data_to_plot.shape[1]
        image_fig, imageax = plt.subplots(figsize=(9, 9))
        if vmin is None:
            vmin = np.min(data_to_plot)
        if vmax is None:
            vmax = np.max(data_to_plot)
        pos = imageax.imshow(data_to_plot, cmap=cmap, vmin=vmin, vmax=vmax)
        imageax.set_title("Image data", color="black")
        imageax.set_xlabel("x")
        imageax.set_ylabel("y")
        image_fig.colorbar(
            pos, ax=imageax, label="DN", fraction=data_ratio * 0.05
        )

        return image_fig, imageax

    def show_histogram(self, bins: int = 1000):
        """Shows a histogram of the image"""
        fig, ax = plt.subplots(figsize=(9, 9))
        histo = np.histogram(self.data, bins=bins)
        ax.stairs(*histo)
        ax.set_title("Image histogram", color="black")
        ax.set_xlabel("Signal [DN]")
        ax.set_ylabel("Number of pixels")

        return fig, ax

    def __add__(self, second):
        if type(self) is type(second):
            newdata = self.data + second.data
        else:
            newdata = self.data + second
        return Image(newdata)

    def __sub__(self, second):
        if type(self) is type(second):
            newdata = self.data - second.data
        else:
            newdata = self.data - second
        return Image(newdata)

    def __mul__(self, second):
        if type(self) is type(second):
            newdata = self.data * second.data
        else:
            newdata = self.data * second
        return Image(newdata)

    def __truediv__(self, second):
        if type(self) is type(second):
            newdata = self.data / second.data
        else:
            newdata = self.data / second
        return Image(newdata)
