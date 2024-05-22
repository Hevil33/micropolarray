import datetime
import multiprocessing as mp
import os
import sys
from logging import critical, info
from pathlib import Path

import numpy as np
import pytz
import tqdm
from astropy.io import fits


def three_bytes_to_two_ints(filecontent):
    """Needed for parallelization, this will be run by each thread for a slice of the original array.

    Returns:
        np.array: array of saved data
    """
    newarr = np.zeros(shape=int(len(filecontent) * 2 / 3))
    position = 0
    for i in range(0, len(filecontent), 3):
        binsum = "".join([bin(j)[2:].zfill(8) for j in filecontent[i : i + 3]])
        newarr[position] = int(binsum[0:12], 2)
        newarr[position + 1] = int(binsum[16:24] + binsum[12:16], 2)
        position += 2
    return newarr


def nparr_from_binary(filename):
    """Converts a PolarCam binary file into a numpy array. Bytes are saved like this

    - 24 bit (3 bytes)
        1             |   3                |     2
        111111111111       | 1111               | 11111111
    - 2 numbers
        First number 12bit | Second number (little endian) 8+4=12 bit

    Args:
        filename (str): name of the file to be converted

    Raises:
        ValueError: file lenghts is indivisible by the number of chunks requested to parallelize operations

    Returns:
        np.array: array of data from file
    """
    with open(filename, mode="rb") as file:
        filecontent = file.read()  #  serial representation
    image_dimension = 1952
    newarr = np.zeros(shape=image_dimension * image_dimension)
    chunks_n = 32
    chunk_size = len(filecontent) / chunks_n
    if chunk_size % 1 or (chunk_size / 3) % 1:
        raise ValueError("Indivisible by chunks")
    chunk_size = int(chunk_size)
    splitted = np.array(
        [filecontent[i * chunk_size : (i + 1) * chunk_size] for i in range(chunks_n)]
    )
    with mp.Pool(processes=chunks_n) as p:
        result = p.map(three_bytes_to_two_ints, splitted)
    newarr = np.array(result).reshape((1952, 1952))
    return newarr


def convert_set(filenames, new_filename, height, width):
    """ANTARTICOR ONLY: Sums a set of filenames and converts them to one fits file.


    Args:
        filenames (list): list of file names to be summed before being converted
        new_filename (str): new .fits file name
    """
    abs_new_filename = os.path.abspath(new_filename)
    if abs_new_filename.split(".")[-1] != "fits":
        raise ValueError("Trying to save a .fits file to .bin, check new filename")
    if type(filenames) is not list:
        filenames = [
            filenames,
        ]
    images_n = len(filenames)
    arr = np.zeros(shape=(height, width))
    for filename in tqdm.tqdm(filenames):
        arr += nparr_from_binary(filename) / images_n
    hdu = fits.PrimaryHDU(data=arr)
    date_and_time = datetime.datetime.now(tz=pytz.timezone("Australia/Perth")).strftime(
        "%Y-%m-%dT%H:%M:%S%z"
    )
    hdu.header["CREATED"] = (
        str(date_and_time),
        "Datetime conversion from bin to fits file (Dome C timezone).",
    )
    hdu.writeto(abs_new_filename, overwrite=True)


def average_rawfiles_to_fits(
    filenames: list, new_filename: str, height: int, width: int
):
    """Saves the mean of a list of rawfiles to a new fits file.

    Args:
        filenames (list): list of raw filenames
        new_filename (str): new fits filename
        height (int): image height in pix
        width (int): image width in pix

    Raises:
        ValueError: trying to save in a file that does not end with .fits
    """
    merge_rawfiles_to_fits(
        filenames=filenames,
        new_filename=new_filename,
        height=height,
        width=width,
        mode="average",
    )

    """
    abs_new_filename = str(Path(new_filename).absolute())
    if abs_new_filename.split(".")[-1] != "fits":
        raise ValueError("Output filename must have a .fits extension.")
    if type(filenames) is not list:
        filenames = [
            filenames,
        ]
    images_n = len(filenames)
    arr = np.zeros(shape=(height, width))
    for filename in tqdm.tqdm(filenames):
        with open(filename, mode="rb") as file:
            buffer = file.read()
        data = np.ndarray(shape=(height, width), dtype="<u2", buffer=buffer)

        arr += data / images_n
    hdu = fits.PrimaryHDU(data=arr)
    date_and_time = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S%z")
    hdu.header["CREATED"] = (
        str(date_and_time),
        "Datetime conversion from bin to fits file (Dome C timezone).",
    )
    hdu.writeto(abs_new_filename, overwrite=True)
    """


def merge_rawfiles_to_fits(
    filenames: list, new_filename: str, height: int, width: int, mode="sum"
):
    """Saves the average or sum of a list of rawfiles to a new fits file.

    Args:
        filenames (list): list of raw filenames
        new_filename (str): new fits filename
        height (int): image height in pix
        width (int): image width in pix
        mode (str): wether to "average" or "sum" the images. Defaults to "sum".

    Raises:
        ValueError: trying to save in a file that does not end with .fits
    """
    abs_new_filename = str(Path(new_filename).absolute())
    if abs_new_filename.split(".")[-1] != "fits":
        raise ValueError("Output filename must have a .fits extension.")
    if type(filenames) is not list:
        filenames = [
            filenames,
        ]
    images_n = len(filenames)
    arr = np.zeros(shape=(height, width))
    for filename in tqdm.tqdm(filenames):
        with open(filename, mode="rb") as file:
            buffer = file.read()
        data = np.ndarray(shape=(height, width), dtype="<u2", buffer=buffer)
        arr += data

    if mode == "average":
        arr /= images_n

    hdu = fits.PrimaryHDU(data=arr)
    date_and_time = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S%z")
    hdu.header["CREATED"] = (
        str(date_and_time),
        "Datetime conversion from bin to fits file (Dome C timezone).",
    )
    hdu.writeto(abs_new_filename, overwrite=True)


def convert_rawfile_to_fits(
    filename: str, height: int, width: int, remove_old: bool = False
):
    """Converts a raw file to a fits one, using default header

    Args:
        filename (str): raw filename
        height (int): file height
        width (int): file width
        remove_old (bool, optional): remove old raw file after conversion. Defaults to False.

    Raises:
        ValueError: raised if the file does not end with ".raw"
    """
    if not ".raw" in filename:
        raise ValueError("Can't convert: not a row file")
    with open(filename, mode="rb") as file:
        buffer = file.read()
    data = np.ndarray(shape=(height, width), dtype="<u2", buffer=buffer)
    HDU = fits.PrimaryHDU(data=data)
    filename = os.path.abspath(filename)  # prevents bug when "../" is in filename
    new_filename = os.path.pathsep.join(filename.split(".")[:-1]) + ".fits"
    HDU.writeto(new_filename, overwrite=True)
    info(f'Image successfully saved to "{new_filename}".')

    if remove_old:
        os.remove(filename)
