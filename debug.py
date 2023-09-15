import sys
import time
import tracemalloc

import matplotlib.pyplot as plt
import micropolarray as ml
import numpy as np
import numpy.linalg
from micropolarray.processing.nrgf import remove_outliers_simple


def main():
    data = np.ones(shape=(12, 12))

    image = ml.MicropolImage(data)

    for name in image.__dir__():
        if (not "__" in name) and (not hasattr(name, "__call__")):
            print(name)

    print(image.single_pol_subimages.shape)

    print(image.single_pol_subimages[0])
    print(image.pol0.data)
    image.single_pol_subimages[0] = 3
    print(image.pol0.data is image.single_pol_subimages[0])
    print(image.single_pol_subimages[0] is image.pol0.data)

    print(image.pol0.data)

    # image2 = ml.PolarcamImage(data)


if __name__ == "__main__":
    main()
