import sys
import time
import tracemalloc
from glob import glob
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg

import micropolarray as ml
from micropolarray.processing.nrgf import remove_outliers_simple


def main():
    external_disk_folder = "/media/herve/TOSHIBA EXT/dottorato/cormag"
    fname = glob(
        "/media/herve/TOSHIBA EXT/dottorato/cormag/2023_flight/data/images/volo fase 1/seq. 10/*.fits"
    )[0]
    dark_folder = external_disk_folder + "/data/2023_07_14/dark"
    dark_files = {}
    for folder in glob(dark_folder + "/*"):
        folderpath = Path(folder)
        dark_files[int(folderpath.name.strip("ms"))] = str(
            folderpath / "average.fits"
        )
    dark_15s = ml.MicropolImage(dark_files[15000])
    dark_60s = ml.MicropolImage(dark_files[60000])
    dark_30s = (dark_15s * (60 - 30) + dark_60s * (30 - 15)) / (60 - 15)
    flat_tilted = ml.MicropolImage(
        external_disk_folder
        + "/data/2023_07_11/diff_suncentered_tilt_30kms.fits",
        dark=dark_30s,
    )
    tilted_image = ml.MicroPolarizerArrayImage(
        fname,
        dark=ml.MicropolImage(dark_files[150000]),
        flat=flat_tilted,
        averageimages=False,
    )


if __name__ == "__main__":
    main()
