import logging

from micropolarray.cameras import Antarticor, Kasi, PolarCam
from micropolarray.image import Image
from micropolarray.micropol_image import MicropolImage
from micropolarray.micropol_image import MicropolImage as MicroPolarizerArrayImage
from micropolarray.micropol_image import MicropolImage as PolarcamImage
from micropolarray.micropol_image import set_default_angles
from micropolarray.processing.chen_wan_liang_calibration import (
    _ifov_jitcorrect,
    chen_wan_liang_calibration,
)
from micropolarray.processing.congrid import congrid
from micropolarray.processing.convert import (
    average_rawfiles_to_fits,
    convert_rawfile_to_fits,
    convert_set,
    merge_rawfiles_to_fits,
)
from micropolarray.processing.demodulation import (
    Demodulator,
    calculate_demodulation_tensor,
)
from micropolarray.processing.demodulation_errors import MicropolImageError
from micropolarray.processing.demosaic import (
    demosaic,
    merge_polarizations,
    split_polarizations,
)
from micropolarray.processing.image_cleaning import auto_threshold, get_hot_pixels
from micropolarray.processing.linear_roi import linear_roi, linear_roi_from_polar
from micropolarray.processing.nrgf import (
    find_occulter_hough,
    find_occulter_position,
    nrgf,
    roi_from_polar,
)
from micropolarray.processing.rebin import trim_to_match_binning
from micropolarray.utils import (
    align_keywords_and_data,
    get_Bsun_units,
    get_malus_normalization,
    mean_minus_std,
    mean_plus_std,
    median_minus_std,
    median_plus_std,
    normalize2pi,
    sigma_DN,
)

logging.basicConfig(
    level=logging.INFO, format="%(levelname)s - %(asctime)s - %(message)s"
)  # tempo, livello, messaggio. livello è warning, debug, info, error, critical

__all__ = []  # Imported modules when "from microppolarray_lib import *" is called
