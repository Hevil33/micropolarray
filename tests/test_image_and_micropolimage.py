import glob
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest
from astropy.io import fits
from scipy.optimize import curve_fit
from test_utils import generate_dummy_data, generate_polarized_data

import micropolarray as ml
from micropolarray.polarization_functions import AoLP, DoLP, pB
from micropolarray.processing.demodulation import Malus


class TestMicropolImage:
    def write_temp_image(self, tmp_path, data):
        """Writes images to the temp folder for testing"""
        image = fits.PrimaryHDU(data=data, do_not_scale_image_data=True, uint=False)
        image.header["FROMFILE"] = True
        image.writeto(tmp_path / "sample_image.fits")

    def test_image_initialization(self, generate_dummy_data, tmp_path):
        """Tests the initialization of both Image and MicroPolArrayImage"""
        dummy_data_16 = generate_dummy_data(16)
        self.write_temp_image(tmp_path, dummy_data_16)
        for ImageClass in [ml.Image, ml.MicropolImage]:
            image = ImageClass(dummy_data_16)
            assert np.all(image.data == dummy_data_16)

            image = ImageClass(str(tmp_path / "sample_image.fits"))
            assert np.all(image.data == dummy_data_16)
            assert image.header["FROMFILE"] == True

            image = ImageClass(image)
            assert np.all(image.data == dummy_data_16)

    def test_image_writing(self, generate_dummy_data, tmp_path):
        """Tests the saving of both Image and MicroPolArrayImage"""
        dummy_data_16 = generate_dummy_data(16)
        image = ml.Image(dummy_data_16)
        for input_path in ["image.fits", "test/image.fits"]:
            image.save_as_fits(tmp_path / input_path)
            read_image = ml.Image(str((tmp_path / input_path).with_suffix(".fits")))
            assert np.all(read_image.data == dummy_data_16)
            read_image.header["EXTEND"] = True
            assert read_image.header == image.header

        for image_type in [ml.Image, ml.MicropolImage]:
            image = image_type(dummy_data_16)
            image.save_as_fits(str(tmp_path / "image.fits"))

        image = ml.MicroPolarizerArrayImage(dummy_data_16)
        print(tmp_path / "image.fits")
        print((tmp_path / "image.fits").suffix)

        image.save_single_pol_images(tmp_path / "image.fits")

        for i in glob.glob(str(tmp_path / "image*")):
            print(i)
        return
        assert np.all(ml.Image(tmp_path / "image_POL0.fits").data == 1)

    def test_show(self, generate_dummy_data):
        dummy_data_16 = generate_dummy_data(16)
        for image_type in [ml.MicropolImage, ml.Image]:
            dummy_image = image_type(dummy_data_16)
            dummy_image.show()
            dummy_image.show_histogram()
        dummy_image = ml.MicropolImage(dummy_data_16)
        dummy_image.show_with_pol_params()
        dummy_image.show_single_pol_images()
        dummy_image.show_pol_param("DoLP")
        dummy_image.demosaic()
        dummy_image.show_demo_images()

    def test_dark_and_flat_correction(self, generate_dummy_data, tmp_path):
        # test dark
        dummy_data_16 = generate_dummy_data(16)
        dark_data = generate_dummy_data(16)
        dark_image = ml.MicropolImage(dark_data)
        dummy_image = ml.MicropolImage(dummy_data_16, dark=dark_image)
        assert np.all(dummy_image.data == 0.0)
        assert np.all(dummy_image.DoLP.data == 0.0)

        # test flat
        signal = 4.0
        dummy_data_16 = np.ones(shape=(16, 16)) * signal
        flat_image = ml.MicropolImage(dummy_data_16 * np.random.random(1))
        dummy_image = ml.MicropolImage(dummy_data_16, flat=flat_image)

        assert np.all(np.round(dummy_image.data, 2) * 1.0 == np.round(signal))

    def test_demosaic(self, generate_dummy_data, tmp_path):
        """Tests demosaic operation and demosaic writing"""
        dummy_data_16 = generate_dummy_data(16)
        # test mean
        image = ml.MicropolImage(dummy_data_16)
        assert image.data.shape == (16, 16)
        assert image.I.data.shape == (8, 8)
        image = image.demosaic(demosaic_mode="mean")

        for idx, demo_image in enumerate(image.demosaiced_images):
            assert np.all(demo_image == np.full((16, 16), (idx + 1) / 4.0))
        assert np.all(image.I.data == np.full((16, 16), 0.25 * 0.5 * (1 + 2 + 3 + 4)))

        # test adjacent
        image = ml.MicropolImage(dummy_data_16)
        image = image.demosaic(demosaic_mode="adjacent")
        for idx, demo_image in enumerate(image.demosaiced_images):
            assert np.all(demo_image == np.full((16, 16), (idx + 1)))

        # test writing
        image.save_demosaiced_images_as_fits(str(tmp_path / "demosaiced_images.fits"))

    def test_rebinning(self, generate_dummy_data):
        """Tests 2x2 and 4x4 binning (the other will be supposedly fine)"""
        dummy_data_16 = generate_dummy_data(16)

        binned_image_2 = ml.MicropolImage(dummy_data_16).rebin(2)
        assert np.all(binned_image_2.data == generate_dummy_data(8) * 4)

        binned_image_4 = ml.MicropolImage(dummy_data_16).rebin(4)
        assert np.all(binned_image_4.data == generate_dummy_data(4) * 16)

    def test_pol_parameters(self, generate_dummy_data):
        """Test if polarization parameters are correcly computed"""

        def test_theo_stokes(image, I, Q, U):
            assert np.all(image.I.data == I)
            assert np.all(image.Q.data == Q)
            assert np.all(image.U.data == U)

            assert np.all(image.AoLP.data == 0.5 * np.arctan2(U, Q) * half_ones)
            assert np.all(image.pB.data == np.sqrt(Q * Q + U * U) * half_ones)
            assert np.all(image.DoLP.data == np.sqrt(Q * Q + U * U) * half_ones / I)

        array_side = 16
        dummy_data_16 = generate_dummy_data(array_side)
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

    def test_congrid(self, generate_dummy_data):
        dummy_data_40 = generate_dummy_data(40)
        dummy_data_16 = generate_dummy_data(16)
        image = ml.MicropolImage(dummy_data_40)
        congridded_image = image.congrid(16, 16)
        assert np.all(congridded_image.data == dummy_data_16)
        assert np.all(congridded_image.I.data == 0.5 * (1 + 2 + 3 + 4))

    def test_operations(self, generate_dummy_data):
        data = np.ones(shape=(16, 16))
        for image_type in [ml.Image, ml.MicropolImage]:
            image1 = image_type(7 * data)
            image2 = image_type(2.0 * data)
            result = image1 + image2
            assert np.all(result.data == 9.0)
            result = image1 - image2
            assert np.all(result.data == 5.0)
            result = image1 * image2
            assert np.all(result.data == 14.0)
            result = image1 / image2
            assert np.all(result.data == 3.5)

        image1 = ml.MicropolImage(
            generate_polarized_data(shape=(6, 16), S=100, angle_rad=0)
        )
        image2 = ml.MicropolImage(50 * np.ones_like(image1.data))

        print(image1.AoLP)
