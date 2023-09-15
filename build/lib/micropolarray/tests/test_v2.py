import pytest
import micropolarray as ml
import numpy as np
from astropy.io import fits
from pathlib import Path
import os
import glob


@pytest.fixture(autouse=True)
def dummy_data():
    """Dummy data factory"""

    def _make_dummy_data(dimension):
        dummydata = np.zeros(shape=(dimension, dimension))
        dummydata[0::2, 0::2] = 1
        dummydata[0::2, 1::2] = 2
        dummydata[1::2, 0::2] = 3
        dummydata[1::2, 1::2] = 4
        return dummydata

    return _make_dummy_data


def write_temp_image(tmp_path, data):
    """Writes images to the temp folder for testing"""
    image = fits.PrimaryHDU(
        data=data, do_not_scale_image_data=True, uint=False
    )
    image.writeto(tmp_path / "sample_image.fits")


def test_image_initialization(dummy_data, tmp_path):
    """Tests the initialization of both Image and MicroPolArrayImage"""
    dummy_data_16 = dummy_data(16)
    write_temp_image(tmp_path, dummy_data_16)
    for image_type in [ml.Image, ml.MicropolImage]:
        image = image_type(dummy_data_16)
        assert np.all(image.data == dummy_data_16)

        image = image_type(str(tmp_path / "sample_image.fits"))
        assert np.all(image.data == dummy_data_16)

        image = image_type(image)
        assert np.all(image.data == dummy_data_16)


def test_image_writing(dummy_data, tmp_path):
    """Tests the saving of both Image and MicroPolArrayImage"""
    dummy_data_16 = dummy_data(16)
    for image_type in [ml.Image, ml.MicropolImage]:
        image = image_type(dummy_data_16)
        image.save_as_fits(str(tmp_path / "image.fits"))


def test_demosaic(dummy_data, tmp_path):
    """Tests demosaic operation and demosaic writing"""
    dummy_data_16 = dummy_data(16)
    # test mean
    image = ml.MicropolImage(dummy_data_16, demosaic_mode="mean")
    image = image.demosaic()
    for idx, demo_image in enumerate(image.demosaiced_images):
        assert np.all(demo_image == np.full((16, 16), (idx + 1) / 4.0))

    # test adjacent
    image = ml.MicropolImage(dummy_data_16, demosaic_mode="adjacent")
    image = image.demosaic()
    for idx, demo_image in enumerate(image.demosaiced_images):
        assert np.all(demo_image == np.full((16, 16), (idx + 1)))

    # test writing
    image.save_demosaiced_images_as_fits(
        str(tmp_path / "demosaiced_images.fits")
    )


def test_rebinning(dummy_data):
    """Tests 2x2 and 4x4 binning (the other will be supposedly fine)"""
    dummy_data_16 = dummy_data(16)

    binned_image_2 = ml.MicropolImage(dummy_data_16).rebin(2)
    assert np.all(binned_image_2.data == dummy_data(8) * 4)

    binned_image_4 = ml.MicropolImage(dummy_data_16).rebin(4)
    assert np.all(binned_image_4.data == dummy_data(4) * 16)


def test_pol_parameters(dummy_data):
    """Test if polarization parameters are correcly computed"""

    def test_theo_stokes(image, I, Q, U):
        assert np.all(image.I.data == I)
        assert np.all(image.Q.data == Q)
        assert np.all(image.U.data == U)

        assert np.all(image.AoLP.data == 0.5 * np.arctan(U / Q) * half_ones)
        assert np.all(image.pB.data == np.sqrt(Q * Q + U * U) * half_ones)
        assert np.all(
            image.DoLP.data == np.sqrt(Q * Q + U * U) * half_ones / I
        )

    array_side = 16
    dummy_data_16 = dummy_data(array_side)
    half_ones = np.ones(shape=(int(array_side / 2), int(array_side / 2)))
    image = ml.MicropolImage(dummy_data_16)

    for i in range(4):
        assert np.all(image.single_pol_subimages[i] == i + 1)

    angles = [0, 45, -45, 90]
    numbers = [1.0, 2.0, 3.0, 4.0]
    for angle, n in zip(angles, numbers):
        assert np.all(image.single_pol_subimages[image.angle_dic[angle]] == n)

    I = 0.5 * (1 + 2 + 3 + 4)
    Q = 1.0 - 4.0
    U = 2.0 - 3.0
    test_theo_stokes(image, I, Q, U)

    new_angle_dic = {45: 0, 0: 1, 90: 2, -45: 3}
    image = ml.MicropolImage(dummy_data_16, angle_dic=new_angle_dic)
    assert image.angle_dic == new_angle_dic

    Q = 2.0 - 3.0
    U = 1.0 - 4.0
    test_theo_stokes(image, I, Q, U)


# TODO


def test_demodulation(dummy_data, tmp_path):
    """Create a dummy demodulation matrix, save it, read it then use it to demodulate. Check if demodulation is correctly done."""
    dummy_data_16 = dummy_data(16)
    angles = np.array([np.deg2rad(angle) for angle in [0, 45, -45, 90]])
    demo_matrix = np.array(
        [
            [0.5, 0.5, 0.5, 0.5],
            np.cos(2 * angles),
            np.sin(2 * angles),
        ]
    )
    for i in range(3):
        for j in range(4):
            image = fits.PrimaryHDU()
            image.data = np.ones_like(dummy_data_16) * demo_matrix[i, j]
            image.writeto(tmp_path / f"M{i}{j}.fits")

    image = ml.PolarcamImage(dummy_data_16)
    assert np.all(image.Q.data == (1 - 4))
    assert np.all(image.U.data == (2 - 3))
    assert np.all(image.I.data == (0.5 * (1 + 2 + 3 + 4)))

    demodulator = ml.Demodulator(str(tmp_path))

    demo_image = image.demodulate(demodulator=demodulator)
    assert np.all(np.round(demo_image.Q.data, 5) == (1.0 - 4.0))
    assert np.all(np.round(demo_image.U.data, 5) == (2.0 - 3.0))
    assert np.all(
        np.round(demo_image.I.data, 5) == (0.5 * (1.0 + 2.0 + 3.0 + 4.0))
    )
