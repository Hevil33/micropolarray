import sys
import time

import matplotlib.pyplot as plt
import micropolarray as ml
import numpy as np
from micropolarray.processing.nrgf import remove_outliers_simple


def main():
    data = np.full((12, 12), 3)

    data = ml.MicroPolarizerArrayImage(data)

    data.show()
    shifted = data.shift(1, 1)

    shifted.show()
    plt.show()


if __name__ == "__main__":
    main()
