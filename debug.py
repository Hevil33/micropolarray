import sys
import time

import matplotlib.pyplot as plt
import micropolarray as ml
import numpy as np


def main():
    data = np.zeros(shape=(3000, 4000))
    start = time.perf_counter()
    for j in range(data.shape[0]):
        for i in range(data.shape[1]):
            data[j, i] = j + i + 1
    image = ml.MicroPolarizerArrayImage(data)
    image.show_with_pol_params()
    shifted = image.shift(0, 0)
    end = time.perf_counter()

    print(f"elapsed {end - start}")
    shifted.show_with_pol_params()
    plt.show()

    sys.exit()
    newdata = np.zeros_like(data)
    x = 1
    y = 0

    start = time.perf_counter()
    for j in range((y > 0) * y, data.shape[0] - np.abs(y) * (y < 0)):
        for i in range((x > 0) * x, data.shape[1] - np.abs(x) * (x < 0)):
            print(i, j)
            newdata[j, i] = data[j - y, i - x]

    print(type(newdata))

    newimage = ml.MicroPolarizerArrayImage(newdata)
    newimage.show()
    plt.show()

    zeros = np.zeros_like(image.data)

    end = time.perf_counter()
    print(f"elapsed {end - start}")


if __name__ == "__main__":
    main()
