from typing import Sequence
from fcat import WindModel
import numpy as np
from scipy.signal import lti, lsim
import math
from scipy.interpolate import interp1d
from enum import IntEnum

__all__ = ('DrydenGust',)


class Filter:
    def __init__(self, num: Sequence[float], den: Sequence[float]):
        """
        Wrapper for the scipy LTI system class.
        :param num: numerator of transfer function
        :param den: denominator of transfer function
        """
        self.filter = lti(num, den)
        self.x = None

    def simulate(self, u: np.ndarray, t: np.ndarray) -> np.ndarray:
        """
        Simulate filter
        :param u: filter input
        :param t: timeseries
        :return: filter output
        """
        _, y, x = lsim(self.filter, U=u, T=t, X0=self.x)
        self.x = x[-1]
        return y

    def reset(self):
        """
        Reset filter
        :return:
        """
        self.x = None


class TurbulenceIntensity(IntEnum):
    LIGHT = 0
    MODERATE = 1
    SEVERE = 2


class DrydenGust(WindModel):
    def __init__(self, wingspan: float, timeseries: np.ndarray, altitude: float = 100,
                 airspeed: float = 20.0,
                 intensity: TurbulenceIntensity = TurbulenceIntensity.MODERATE, seed: int = None):
        """
        Python realization of the continuous Dryden Turbulence Model (MIL-F-8785C).

        This implementation is adapted from https://github.com/eivindeb/pyfly

        :param wingspan: wingspan of aircraft
        :param altitude: Altitude of aircraft
        :param airspeed: Airspeed of aircraft
        :param intensity: Intensity of turbulence, see `TurbulenceIntensity`
        """

        b = wingspan
        h = altitude
        V_a = airspeed

        # Conversion factors
        # 1 meter = 3.281 feet
        meters2feet = 3.281
        feet2meters = 1 / meters2feet
        # 1 knot = 0.5144 m/s
        knots2mpers = 0.5144

        if intensity == TurbulenceIntensity.LIGHT:
            W_20 = 15 * knots2mpers
        elif intensity == TurbulenceIntensity.MODERATE:
            W_20 = 30 * knots2mpers
        elif intensity == TurbulenceIntensity.SEVERE:
            W_20 = 45 * knots2mpers
        else:
            raise Exception("Unsupported intensity type")

        # Convert meters to feet and follow MIL-F-8785C spec
        h = h * meters2feet
        b = b * meters2feet
        V_a = V_a * meters2feet
        W_20 = W_20 * meters2feet

        # Turbulence intensities
        sigma_w = 0.1 * W_20
        sigma_u = sigma_w / (0.177 + 0.000823 * h) ** 0.4
        sigma_v = sigma_u

        # Turbulence length scales
        L_u = h / (0.177 + 0.000823 * h) ** 1.2
        L_v = L_u
        L_w = h

        K_u = sigma_u * math.sqrt((2 * L_u) / (math.pi * V_a))
        K_v = sigma_v * math.sqrt((L_v) / (math.pi * V_a))
        K_w = sigma_w * math.sqrt((L_w) / (math.pi * V_a))

        T_u = L_u / V_a
        T_v1 = math.sqrt(3.0) * L_v / V_a
        T_v2 = L_v / V_a
        T_w1 = math.sqrt(3.0) * L_w / V_a
        T_w2 = L_w / V_a

        K_p = sigma_w * math.sqrt(0.8 / V_a) * ((math.pi / (4 * b)) ** (1 / 6)) / ((L_w) ** (1 / 3))
        K_q = 1 / V_a
        K_r = K_q

        T_p = 4 * b / (math.pi * V_a)
        T_q = T_p
        T_r = 3 * b / (math.pi * V_a)

        self.filters = {
            "H_u": Filter(feet2meters * K_u, [T_u, 1]),
            "H_v": Filter([feet2meters * K_v * T_v1, feet2meters * K_v],
                          [T_v2 ** 2, 2 * T_v2, 1]),
            "H_w": Filter([feet2meters * K_w * T_w1, feet2meters * K_w],
                          [T_w2 ** 2, 2 * T_w2, 1]),
            "H_p": Filter(K_p, [T_p, 1]),
            "H_q": Filter([-K_w * K_q * T_w1, -K_w * K_q, 0],
                          [T_q * T_w2 ** 2, T_w2 ** 2 + 2 * T_q * T_w2, T_q + 2 * T_w2, 1]),
            "H_r": Filter([K_v * K_r * T_v1, K_v * K_r, 0],
                          [T_r * T_v2 ** 2, T_v2 ** 2 + 2 * T_r * T_v2, T_r + 2 * T_v2, 1])}

        self.np_random = None
        self.seed(seed)
        self.time = timeseries
        self.wind_data = self._simulate(timeseries)
        self.interp_wind_data = self._interpolate_wind_data()

    def seed(self, seed: int = None) -> None:
        """
        Seed the random number generator.
        :param seed: (int) seed.
        :return:
        """
        self.np_random = np.random.RandomState(seed)

    def _interpolate_wind_data(self) -> np.ndarray:
        interpdata_list = []
        for i in range(6):
            interpdata_list.append(interp1d(self.time, self.wind_data[i, :]))
        return interpdata_list

    def _generate_noise(self, t: np.ndarray) -> np.ndarray:
        """
        Return a numpy array of normal distributed random variables of length 4
        """
        dt = t[1]-t[0]
        return np.sqrt(np.pi / dt) * self.np_random.standard_normal(size=(4, len(t)))

    def reset(self) -> None:
        for f in self.filters.values():
            f.reset()

    def _simulate(self, t: np.ndarray) -> np.ndarray:
        """
        Simulate turbulence by passing white white noise through the filters
        Return the wind vector of length 6. The first three are translational
        and the last three are rotational
        """

        noise = self._generate_noise(t)
        velocity = np.array([self.filters["H_u"].simulate(noise[0, :], t),
                             self.filters["H_v"].simulate(noise[1, :], t),
                             self.filters["H_w"].simulate(noise[2, :], t),

                             # Rotational part
                             self.filters["H_p"].simulate(noise[3, :], t),
                             self.filters["H_q"].simulate(noise[1, :], t),
                             self.filters["H_r"].simulate(noise[2, :], t)])
        assert velocity.shape == (6, len(t))
        return velocity

    def get(self, t: float) -> np.ndarray:
        wind_array = np.zeros((6,))
        for i in range(len(wind_array)):
            wind_array[i] = self.interp_wind_data[i](t)
        return wind_array
