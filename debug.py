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

    test_data = np.ones(shape=(2, 2))
    test_class = test(test_data)
    print(test_class.data)
    print(test_class.param)
    test_class.param = 3
    print(test_class.param)
    sys.exit()
    image = ml.Image(test_data)
    print(image.new_data)
    image.new_data = 3
    print(image.new_data)
    test(test_data)


if __name__ == "__main__":
    main()
