from micropolarray.image import Image
from micropolarray.micropol_image import (
    MicropolImage,
    set_default_angles,
)
from micropolarray.micropol_image import (
    MicropolImage as PolarcamImage,
)
from micropolarray.micropol_image import (
    MicropolImage as MicroPolarizerArrayImage,
)
from micropolarray.cameras import PolarCam, Kasi, Antarticor

from micropolarray.processing.demodulation import Demodulator
from micropolarray.processing.demodulation import (
    calculate_demodulation_tensor,
)
from micropolarray.processing.nrgf import (
    find_occulter_position,
    roi_from_polar,
    nrgf,
)
from micropolarray.processing.convert import (
    convert_set,
    convert_rawfile_to_fits,
    average_rawfiles_to_fits,
)
from micropolarray.processing.demosaic import (
    demosaic,
    merge_polarizations,
    split_polarizations,
)
from micropolarray.utils import (
    sigma_DN,
    mean_minus_std,
    mean_plus_std,
    median_minus_std,
    median_plus_std,
    get_Bsun_units,
    get_malus_normalization,
)
from micropolarray.processing.chen_wan_liang_calibration import (
    chen_wan_liang_calibration,
    ifov_jitcorrect,
)
from micropolarray.processing.congrid import congrid
from micropolarray.utils import (
    normalize2pi,
    align_keywords_and_data,
)

import logging

logging.basicConfig(
    level=logging.INFO, format="%(levelname)s - %(asctime)s - %(message)s"
)  # tempo, livello, messaggio. livello è warning, debug, info, error, critical

__all__ = (
    []
)  # Imported modules when "from microppolarray_lib import *" is called
