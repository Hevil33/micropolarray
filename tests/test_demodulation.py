import glob
import os
from logging import info, warn
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


class TestDemodulation:
    def test_demo_from_dummy(self, generate_dummy_data, tmp_path):
        """Create a dummy demodulation matrix, save it, read it then use it to demodulate. Check if demodulation is correctly done."""
        ml.set_default_angles(ml.PolarCam().angle_dic)
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
                image.data = np.ones(shape=(8, 8)) * demo_matrix[i, j]
                image.writeto(tmp_path / f"M{i}{j}.fits")

        image = ml.PolarcamImage(generate_dummy_data(16))
        assert np.all(image.Q.data == (1 - 4))
        assert np.all(image.U.data == (2 - 3))
        assert np.all(image.I.data == (0.5 * (1 + 2 + 3 + 4)))

        demodulator = ml.Demodulator(str(tmp_path))

        demo_image = image.demodulate(demodulator=demodulator)
        assert np.all(np.round(demo_image.Q.data, 5) == (1.0 - 4.0))
        assert np.all(np.round(demo_image.U.data, 5) == (2.0 - 3.0))
        assert np.all(np.round(demo_image.I.data, 5) == (0.5 * (1.0 + 2.0 + 3.0 + 4.0)))

    # TODO refactor calling angles with fixtures
    def test_demodulation_computation(self, tmp_path):
        # try demodulation
        angles = np.array([np.deg2rad(angle) for angle in [0, 45, -45, 90]])
        output_dir = tmp_path / "computed_matrix"
        output_str = str(output_dir)

        polarizations = np.arange(-45, 136, 15)
        pols_rad = np.deg2rad(polarizations)
        input_signal = 100
        t = 0.9
        eff = 0.7
        side = 20
        shape = (side, side)
        ones = np.ones(shape=shape)
        for pol, pol_rad in zip(polarizations, pols_rad):
            result_image = ml.MicropolImage(
                generate_polarized_data(
                    shape=shape,
                    S=input_signal,
                    angle_rad=pol_rad,
                    t=t,
                    eff=eff,
                )
            )
            result_image.save_as_fits(tmp_path / f"pol_{int(pol)}.fits")
        if False:  # check that fit will be ok
            for angle in angles:
                pars, pcov = curve_fit(
                    Malus,
                    pols_rad,
                    np.array([Malus(pol, t, eff, angle) for pol in pols_rad]),
                )
                print(f"t = {pars[0]}")
                print(f"eff = {pars[1] }")
                print(f"phi = {np.rad2deg(pars[2]) }")

        # read the files
        filenames = sorted(
            glob.glob(str(tmp_path / "pol*.fits")),
            key=lambda x: int(x.split(os.path.sep)[-1][4:].strip(".fits")),
        )
        # test normalization with number and with array
        for S_type in [
            input_signal,
            input_signal * np.ones(shape=(side, side)),
        ]:
            ml.calculate_demodulation_tensor(
                polarizer_orientations=polarizations,
                filenames_list=filenames,
                micropol_phases_previsions=np.rad2deg(angles),
                gain=2.75,
                output_dir=output_str,
                binning=1,
                occulter=False,
                procs_grid=[2, 2],
                dark_filename=None,
                flat_filename=None,
                normalizing_S=S_type,
                DEBUG=False,
            )

        # image polarized with phi=0 uniform, t=1, eff=1
        ideal_image = ml.MicropolImage(
            generate_polarized_data(
                shape=shape, S=input_signal, angle_rad=0, t=1, eff=1
            )
        )
        assert np.all(ideal_image.I.data == input_signal)
        assert np.all(ideal_image.Q.data == input_signal)
        assert np.all(ideal_image.U.data == 0)
        assert np.all(ideal_image.pB.data == input_signal)
        assert np.all(ideal_image.AoLP.data == 0)
        assert np.all(ideal_image.DoLP.data == 1)

        demodulator = ml.Demodulator(output_str)

        assert demodulator.eff.shape == ones.shape
        assert np.all(np.round(demodulator.eff, 2) == np.round(eff * ones, 2))
        assert np.all(np.round(demodulator.tk, 2) == np.round(t * ones, 2))
        assert np.all(
            np.round(demodulator.phi, 2)
            == np.round(
                ml.merge_polarizations(
                    np.array(
                        [
                            np.rad2deg(angles[i])
                            * np.ones(shape=(int(side / 2), int(side / 2)))
                            for i in range(4)
                        ]
                    )
                ),
                2,
            )
        )

        assert (
            demodulator.fit_found_flags.shape == demodulator.mij[0, 0].shape
        )  # after September 2023
        test_angle = np.deg2rad(30)
        example_image = ml.MicropolImage(
            generate_polarized_data(
                shape, S=input_signal, angle_rad=test_angle, t=t, eff=eff
            )
        )
        # Theoric values
        I = input_signal * (
            Malus(test_angle, 1, 1, 0) + Malus(test_angle, 1, 1, np.pi / 2)
        )
        Q = input_signal * (
            Malus(test_angle, 1, 1, 0) - Malus(test_angle, 1, 1, np.pi / 2)
        )
        U = input_signal * (
            Malus(test_angle, 1, 1, np.pi / 4) - Malus(test_angle, 1, 1, -np.pi / 4)
        )
        S = [I, Q, U]
        dolp = np.round(DoLP(S), 5)
        aolp = np.round(AoLP(S), 5)
        pb = np.round(pB(S), 5)

        # test demodulation without demosaicing
        no_demosaic_image = ml.MicropolImage(example_image)
        no_demosaic_image = no_demosaic_image.demodulate(demodulator, demosaicing=False)
        assert np.all(np.round(no_demosaic_image.pB.data, 5) == pb)
        assert (
            no_demosaic_image.pB.data.shape
            == no_demosaic_image.single_pol_subimages[0].shape
        )

        demo_image = example_image.demodulate(demodulator)
        if False:
            demodulator.show()
            example_image.show_with_pol_params()
            plt.show()

        assert np.all(np.round(demo_image.I.data, 5) == np.round(I, 5))
        assert np.all(np.round(demo_image.Q.data, 5) == np.round(Q, 5))
        assert np.all(np.round(demo_image.U.data, 5) == np.round(U, 5))
        assert np.all(np.round(demo_image.DoLP.data, 5) == dolp)
        assert np.all(np.round(demo_image.AoLP.data, 5) == aolp)
        assert np.all(np.round(demo_image.pB.data, 5) == pb)

        simples = []
        measureds = []
        theos = []

        def normalize(angle):
            while angle > np.pi / 2:
                angle -= np.pi
            while angle < -np.pi / 2:
                angle += np.pi
            return angle

        # for dummy_angle in np.arange(-np.pi / 2, np.pi / 2, 0.1):
        for dummy_angle in np.arange(-np.pi, np.pi, 0.1):
            polarized_image = ml.MicropolImage(
                generate_polarized_data(shape, input_signal, dummy_angle, t, eff)
            )

            simple = np.round(np.mean(polarized_image.AoLP.data), 1)
            simples.append(simple)

            polarized_image = polarized_image.demodulate(demodulator)

            measured = np.round(np.mean(polarized_image.AoLP.data), 1)
            theo = np.round(normalize(dummy_angle), 1)
            measureds.append(measured)
            theos.append(theo)

        # debug
        if False:
            for measured, theo, simple in zip(measureds, theos, simples):
                print(theo, simple, measured)
            fig, ax = plt.subplots()
            x = np.arange(-np.pi / 2, np.pi / 2, 0.1)
            x = np.arange(-np.pi, np.pi, 0.1)
            ax.plot(x, theos, label="theos")
            ax.plot(x, simples, label="simple")
            ax.plot(x, measureds, label="demodulated")
            ax.legend()
            plt.show()

        assert np.all(simple == theo)
        assert np.all(measured == theo)

    def test_demo_rebin(self, generate_dummy_data, tmp_path):
        dummy_data = generate_dummy_data(32)
