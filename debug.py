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
    class test:
        def __init__(self, input_data) -> None:
            self.data = input_data
            self._param = None
            pass

        @property
        def param(self):
            if self._param is None:
                self._param = self.data / 2
            return self._param

        @param.setter
        def param(self, input):
            self._param = input

    test_data = np.ones(shape=(12, 12))
    test_class = test(test_data)
    test_class.param = 2

    image = ml.MicropolImage(test_data)

    print(image.data)

    sys.exit()
    image.data = np.ones(shape=(5, 5))
    print(image.height)
    print(image.width)
    print(image.header["NAXIS1"])

    print(image.I)


if __name__ == "__main__":
    main()
