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
from micropolarray.test_v2 import test_demodulation


def main():
    test_demodulation()


if __name__ == "__main__":
    main()
