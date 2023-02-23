from micropolarray import MicroPolarizerArrayImage
import numpy as np


def normalize_to_B_sun(
    data,
    sun,
    texp_data=70.0e-3,
    t_exp_sun=70.0e-3,
) -> np.array:
    """Converts DN data in units of B_sun

    Args:
        data (str | list | MicroPolarizerArrayImage | np.array): data to normalize
        sun (str | list | MicroPolarizerArrayImage | np.array): original sun data
        texp_data (float, optional): data exp time. Defaults to 70.0e-3.
        t_exp_sun (float, optional): sun exp time. Defaults to 70.0e-3.

    Returns:
        np.array: normalized data
    """
    image = MicroPolarizerArrayImage(data)
    sun = MicroPolarizerArrayImage(sun)
    k = 1.083 * 1.0e-5  # diffusion angle
    T_diff = 0.28  # diffuser transmittancy

    data_B_sun = (image.data / texp_data) * T_diff * k / (sun.data / t_exp_sun)

    return data_B_sun
