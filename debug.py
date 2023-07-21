import sys
import time

import matplotlib.pyplot as plt
import micropolarray as ml
import numpy as np
from micropolarray.processing.nrgf import remove_outliers_simple


def main():
    data = np.full((12, 12), 3)

    ml.set_default_angles(ml.Kasi())
    image = ml.MicroPolarizerArrayImage(np.full((12, 12), 2))

    print(image.angle_dic)

    print(image.angle_dic)
    image = ml.MicroPolarizerArrayImage(np.full((12, 12), 2))
    print(image.angle_dic)


if __name__ == "__main__":
    main()
