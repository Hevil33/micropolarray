import numpy as np
import scipy
from scipy.ndimage import median_filter

from micropolarray.micropol_image import MicropolImage
from micropolarray.processing.demosaic import merge_polarizations


def get_hot_pixels(image, threshold=100):
    subimages = image.single_pol_subimages
    blurred_subimages = np.array(
        [scipy.ndimage.median_filter(subimage, size=2) for subimage in subimages]
    )
    contrast = (subimages - blurred_subimages) / (subimages + blurred_subimages)
    diff = np.where(contrast > threshold, 1, 0)
    # diff = subimages - blurred_subimages
    # diff = np.where(diff > threshold, 1, 0)

    newimage = MicropolImage(image)
    newimage._set_data_and_Stokes(merge_polarizations(diff))

    return newimage


def remove_outliers_simple(original, neighbours=2):
    """EXPERIMENTAL DO NOT USE, for improving fitting on occulter position"""
    data = original.copy()
    for i, element in enumerate(data[neighbours:-neighbours]):
        median = np.median(data[i - neighbours : i + neighbours])
        std = np.median(np.abs(data[i - neighbours : i + neighbours] - median))
        if (element < median - 3 * std) or (element > median + 3 * std):
            print()
            print(element)
            print(data[neighbours:-neighbours])
            data[i] = median
            print(data[neighbours:-neighbours])
    return data

    median = np.median(data)

    median_deviation = np.abs(data - np.median(data))
    condition = (data < (median + 3 * median_deviation)) | (
        data > (median - 3 * median_deviation)
    )
    data = np.where(condition, data, median)
    return data
    extreme = 2
    outliers = []
    for i, element in enumerate(data[extreme:-extreme]):
        mean = np.mean([data[i], data[i + 1]])
        std = np.std([data[i], data[i + 1]])
        if (element > (mean + 3 * std)) or (element < (mean - 3 * std)):
            outliers.append(i + 1)
    data = np.delete(data, outliers)
    return data


def reject_outliers(data, m=2.0):
    d = np.abs(data - np.median(data))
    mdev = np.median(d)
    s = d / mdev if mdev else np.zeros(len(d))
    return data[s < m]


def auto_threshold(data: np.ndarray) -> float:
    """Get the threshold following Otsu's algorithm. This assumes that
    there are two populations (noise + signal) and minimizes the intra-
    class variance

    Args:
        data (np.ndarray): array on which to perform the treshold

    Returns:
        float: Otsu's threshold
    """

    def otsu_intraclass_variance(data: np.ndarray, threshold):
        """
        Otsu’s intra-class variance.
        If all pixels are above or below the threshold, this will throw a warning that can safely be ignored.
        """
        return np.nansum(
            [
                np.mean(cls) * np.var(data, where=cls)
                #   weight   ·  intra-class variance
                for cls in [data >= threshold, data < threshold]
            ]
        )  # NaNs only arise if the class is empty, in which case the contribution should be zero, which `nansum` accomplishes.

    data = median_filter(data, size=10)
    print(np.min(data))
    print(np.max(data))
    print("a")
    return min(
        np.linspace(np.min(data) + 1, np.max(data), np.min((data.size, 1000))),
        key=lambda th: otsu_intraclass_variance(data, th),
    )
