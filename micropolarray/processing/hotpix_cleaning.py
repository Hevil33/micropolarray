import numpy as np
import scipy

from micropolarray.micropol_image import MicropolImage
from micropolarray.processing.demosaic import merge_polarizations


def get_hot_pixels(image, threshold=100):
    subimages = image.single_pol_subimages
    blurred_subimages = np.array(
        [
            scipy.ndimage.median_filter(subimage, size=2)
            for subimage in subimages
        ]
    )
    contrast = (subimages - blurred_subimages) / (
        subimages + blurred_subimages
    )
    diff = np.where(contrast > threshold, 1, 0)
    # diff = subimages - blurred_subimages
    # diff = np.where(diff > threshold, 1, 0)

    newimage = MicropolImage(image)
    newimage._set_data_and_Stokes(merge_polarizations(diff))

    return newimage
