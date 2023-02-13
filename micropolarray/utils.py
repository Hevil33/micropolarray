import os
import numpy as np
import pandas as pd
from scipy import constants
from pathlib import Path


def make_abs_and_create_dir_old(filenames: str):
    cwd = os.getcwd()
    turned_to_list = False
    if type(filenames) is not list:
        turned_to_list = True
        filenames = [filenames]
    # Make path absolute, create dirs if not existing
    for idx, filename in enumerate(filenames):
        if not os.path.isabs(filename):
            filename = cwd + os.path.sep + filename
            filenames[idx] = filename
        parent_dir = os.path.sep.join(filename.split(os.path.sep)[:-1])
        if not os.path.exists(parent_dir):
            os.makedirs(parent_dir)
    if turned_to_list:
        return filenames[0]
    else:
        return filenames


def make_abs_and_create_dir(filename: str):
    path = Path(filename)
    print(path)

    if not path.is_absolute():  # suppose it is in cwd
        path = path.joinpath(Path().cwd(), path)

    if path.suffix:
        if not path.parent.exists():
            path.parent.mkdir(parents=True)
    else:
        if not path.exists():
            path.mkdir()
    return str(path)


def sigma_DN(pix_DN):
    gain = 6.93
    sigma_DN = np.sqrt(gain * pix_DN) / gain
    return sigma_DN


def fix_data(data: np.array, min=0.0, max=4096):
    fixed_data = data.copy()
    fixed_data = np.where(fixed_data > min, fixed_data, min)
    fixed_data = np.where(fixed_data < max, fixed_data, max)
    return fixed_data


def mean_minus_std(data: np.array, stds_n: int = 1) -> float:
    return np.mean(data) - stds_n * np.std(data)


def mean_plus_std(data: np.array, stds_n: int = 1) -> float:
    return np.mean(data) + stds_n * np.std(data)


def normalize2pi(angles_list):
    if type(angles_list) is not list:
        angles_list = [
            angles_list,
        ]
    for i, angle in enumerate(angles_list):
        while angle > 90:
            angle -= 180
        while angle <= -90:
            angle += 180
        angles_list[i] = angle

    return angles_list
