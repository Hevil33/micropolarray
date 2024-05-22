import numpy as np

from micropolarray.processing.nrgf import roi_from_polar

# TODO: aggiungere tutti i bei dati qua, rendere le camere personalizzabili (caricare dark da cartelle e cose cosi)


class Camera:
    def __init__(self):
        pass

    def occulter_roi(
        self, data: np.array, fill: float = 0.0, overoccult: int = 0
    ) -> np.array:
        """Returns the array in the polar ROI, else fill

        Args:
            data (np.array): Input array
            fill (float, optional): Value for filling. Defaults to 0.0.
            overoccult (int, optional): Pixels to overoccult. Defaults to 0.

        Returns:
            np.array: Array if in ROI, fill elsewhere
        """
        y, x, r = self.occulter_pos_last
        roidata = roi_from_polar(data, [y, x], [r + overoccult, 5000], fill=fill)

        return roidata

    def occulter_mask(self, overoccult: int = 0, rmax: int = None) -> np.array:
        """Returns an array of True inside the roi, False elsewhere. Useful for mean/std operations (where=occulter_mask).

        Args:
            overoccult (int, optional): Pixels to overoccult. Defaults to 15.
            rmax (int, optional): Maximum r of the ROI. Defaults to image nearest border.

        Returns:
            np.array: Boolean roi array
        """
        y, x, r = self.occulter_pos_last
        r = r + overoccult
        if rmax is None:
            rmax = np.min([y, x])
        occulter_mask = np.where(
            roi_from_polar(
                np.ones(shape=(self.h_image, self.w_image)), [y, x], [r, rmax]
            )
            != 0,
            True,
            False,
        )
        return occulter_mask


class Kasi(Camera):
    def __init__(self):
        # self.angle_dic = {-45: 0, 0: 1, 90: 2, 45: 3}  # old
        self.angle_dic = {0: 0, 45: 1, -45: 2, 90: 3}  # new
        self.linearity_range = [0.0, 2500.0]
        self.PTC = 2.64  # [e-/ADU]
        self.readout_noise = 10  # [e-]
        self.full_well = 10500  # [e-]
        self.h_image = 3000
        self.w_image = 4096


class PolarCam(Camera):
    def __init__(self):
        # self.angle_dic = {90: 0, 45: 1, -45: 2, 0: 3}  # My ref system
        self.angle_dic = {
            0: 0,
            45: 1,
            -45: 2,
            90: 3,
        }  # Ale ref system, 0 = vertical
        self.PolarCam_model = "U4"
        self.sensor_type = "CCD"
        self.h_image = 1952  # height [pixel]
        self.w_image = 1952  # width  [pixel]
        self.pixeldim_l1 = 7.4  # [micron]
        self.pixeldim_l2 = 7.4  # [micron]
        self.orientation00 = 0  # |  [degree]
        self.orientation01 = 45  # /  [degree]
        self.orientation11 = 90  # -- [degree]
        self.orientation10 = 135  # \  [degree]
        self.saturationCapacity = 44.0e3  # [e-]
        self.bitdepth = 12  # [bit]
        self.frameRate = 14  # [fps]
        self.Texp_min = 0.02  # [ms]
        self.quantumEff = 0.76  # @470 (nominal, by user manual)
        self.occulter_pos_last = [
            919,
            941,
            536,
        ]  # Occulter y, x, radius [px, px, px] updated January 2023
        self.occulter_pos_2021 = [
            925,
            934,
            532,
        ]  # updated march 27, 2021/2022 campaign
        # self.sun_dimension_pixels = 446  # from standard astropy atan(R_sun/AU)
        self.sun_dimension_pixels = 457  # L1_processing/from sun_occ_dim.py
        self.occulter_radius_sr = (
            self.occulter_pos_last[-1] / self.sun_dimension_pixels
        )  # occulter dimension in solar radii, L1_processing/from sun_occ_dim.py
        # self.occulter_radius_sr = 1.1901  # occulter dimension in solar radii

        self.gain = 9.28


class Antarticor:
    # AntarctiCor
    def __init__(self):
        self.aperture = 50  # [mm]
        self.effectiveFocalLength = 700  # [mm]
        self.fratio = self.effectiveFocalLength / self.aperture
        self.spectralRange = 591  # (591 +- 5) [nm]
        self.platescale = 4.3  # [arcsec/superpixel]
        self.FoV_degree = 0.6  # +/- 0.6 [degree]
        self.FoV_Rsun = 2.24  # +/- 2.24 [solar radii]

        # Other
        self.Topal = 0.27  # opal transmittance []
        self.const = 1.083 * 10 ** (-5)  # constant []
