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
    image = ml.MicroPolarizerArrayImage(
        "../../../codex/data/20230606_Polarization/T3_430nm/105deg/average.fits"
    )
    image.show()

    plt.show()


if __name__ == "__main__":
    main()
