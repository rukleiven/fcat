from typing import NamedTuple, Tuple
from fcat import AircraftProperties, ControlInput, State
import numpy as np
# (C_D_a_data, C_D_q_data, C_D_delta_e_data, C_L_a_data, C_L_q_data, C_L_delta_e_data)
from fcat.skywalkerX8_data import *
from fcat.utilities import calc_airspeed, calc_angle_of_attack, calc_angle_of_sideslip,\
     calc_rotational_airspeed

from scipy.interpolate import interp1d

__all__ = ('IcedSkywalkerX8Properties',)


class SkywalkerX8Constants(NamedTuple):
    wing_span: float = 2.1
    mean_chord: float = 0.3571
    wing_area: float = 0.75
    motor_constant: float = 40
    motor_efficiency_fact: float = 1
    mass: float = 3.3650
    I_xx: float = 0.340
    I_xy: float = 0.0
    I_xz: float = -0.031
    I_yy: float = 0.165
    I_yz: float = 0.0
    I_zx: float = -0.031
    I_zz: float = 0.400


class IcedSkywalkerX8Properties(AircraftProperties):
    """
    Properties for the SkywalkerX8 airplane. Parmaeter value are found
    in ...
    """

    def __init__(self, control_input: ControlInput, icing: float = 0.0):
        super().__init__(control_input)
        self.icing = icing
        self.constants = SkywalkerX8Constants()
        self.C_D_alpha = InterpolatedProperty(C_D_a_data)
        self.C_D_q = InterpolatedProperty(C_D_q_data, bounds_error=False, fill_value='extrapolate')
        self.C_D_delta_e = InterpolatedProperty(
            C_D_delta_e_data, bounds_error=False, fill_value='extrapolate')
        self.C_L_alpha = InterpolatedProperty(C_L_a_data)
        self.C_L_q = InterpolatedProperty(C_L_q_data, bounds_error=False, fill_value='extrapolate')
        self.C_L_delta_e = InterpolatedProperty(
            C_L_delta_e_data, bounds_error=False, fill_value='extrapolate')
        self.C_Y_beta = InterpolatedProperty(
            C_Y_beta_data, bounds_error=False, fill_value='extrapolate')
        self.C_Y_p = InterpolatedProperty(C_Y_p_data, bounds_error=False, fill_value='extrapolate')
        self.C_Y_r = InterpolatedProperty(C_Y_r_data, bounds_error=False, fill_value='extrapolate')
        self.C_Y_delta_a = InterpolatedProperty(
            C_Y_delta_a_data, bounds_error=False, fill_value='extrapolate')
        self.C_Y_delta_r = InterpolatedProperty(
            C_Y_delta_r_data, bounds_error=False, fill_value='extrapolate')
        self.C_m_alpha = InterpolatedProperty(C_m_data, bounds_error=False, fill_value='extrapolate')
        self.C_m_delta_e = InterpolatedProperty(
            C_m_delta_e_data, bounds_error=False, fill_value='extrapolate')
        self.C_m_q = InterpolatedProperty(C_m_q_data, bounds_error=False, fill_value='extrapolate')
        self.C_l_beta = InterpolatedProperty(
            C_l_beta_data, bounds_error=False, fill_value='extrapolate')
        self.C_l_p = InterpolatedProperty(C_l_p_data, bounds_error=False, fill_value='extrapolate')
        self.C_l_r = InterpolatedProperty(C_l_r_data, bounds_error=False, fill_value='extrapolate')
        self.C_l_delta_a = InterpolatedProperty(
            C_l_delta_a_data, bounds_error=False, fill_value='extrapolate')
        self.C_n_beta = InterpolatedProperty(
            C_n_beta_data, bounds_error=False, fill_value='extrapolate')
        self.C_n_p = InterpolatedProperty(C_n_p_data, bounds_error=False, fill_value='extrapolate')
        self.C_n_r = InterpolatedProperty(C_n_r_data, bounds_error=False, fill_value='extrapolate')
        self.C_n_delta_a = InterpolatedProperty(
            C_n_delta_a_data, bounds_error=False, fill_value='extrapolate')

    def mass(self):
        return self.constants.mass

    def inertia_matrix(self):
        return np.array([[self.constants.I_xx, self.constants.I_xy, self.constants.I_xz],
                         [self.constants.I_xy, self.constants.I_yy, self.constants.I_yz],
                         [self.constants.I_xz, self.constants.I_yz, self.constants.I_zz]])

    def drag_coeff(self, state: State, wind: np.ndarray) -> float:
        airspeed = np.sqrt(np.sum(calc_airspeed(state, wind)**2))
        alpha = calc_angle_of_attack(state, wind)*180.0/np.pi
        c = self.constants.mean_chord
        delta_e = self.control_input.elevator_deflection
        rot_airspeed = calc_rotational_airspeed(state, wind)
        pitch_dot = rot_airspeed[1]
        # TODO: If C_D_q becomes non-zero, should test term containing C_D_q
        return self.C_D_alpha(alpha, self.icing) + \
            self.C_D_q(alpha, self.icing) * c / (2*airspeed) * pitch_dot + \
            self.C_D_delta_e(alpha, self.icing)*np.abs(delta_e)

    def lift_coeff(self, state: State, wind: np.ndarray) -> float:
        airspeed = np.sqrt(np.sum(calc_airspeed(state, wind)**2))
        alpha = calc_angle_of_attack(state, wind)*180.0/np.pi
        c = self.constants.mean_chord
        delta_e = self.control_input.elevator_deflection
        rot_airspeed = calc_rotational_airspeed(state, wind)
        pitch_dot = rot_airspeed[1]
        return self.C_L_alpha(alpha, self.icing) + \
            (self.C_L_q(alpha, self.icing) * c / (2*airspeed)) * pitch_dot + \
            self.C_L_delta_e(alpha, self.icing)*delta_e

    def side_force_coeff(self, state: State, wind: np.ndarray) -> float:
        airspeed = np.sqrt(np.sum(calc_airspeed(state, wind)**2))
        beta = calc_angle_of_sideslip(state, wind)*180.0/np.pi
        b = self.constants.wing_span
        delta_a = self.control_input.aileron_deflection
        rot_airspeed = calc_rotational_airspeed(state, wind)
        roll_dot = rot_airspeed[0]
        yaw_dot = rot_airspeed[2]
        return self.C_Y_beta(beta, self.icing) + \
            (self.C_Y_p(beta, self.icing) * b / (2*airspeed)) * roll_dot + \
            (self.C_Y_r(beta, self.icing) * b / (2*airspeed)) * \
            yaw_dot + self.C_Y_delta_a(beta, self.icing)*delta_a

    def roll_moment_coeff(self, state: State, wind: np.ndarray) -> float:
        airspeed = np.sqrt(np.sum(calc_airspeed(state, wind)**2))
        beta = calc_angle_of_sideslip(state, wind)*180.0/np.pi
        b = self.constants.wing_span
        delta_a = self.control_input.aileron_deflection
        rot_airspeed = calc_rotational_airspeed(state, wind)
        roll_dot = rot_airspeed[0]
        yaw_dot = rot_airspeed[2]
        return self.C_l_beta(beta, self.icing) + \
            (self.C_l_p(beta, self.icing) * b / (2*airspeed)) * roll_dot + \
            (self.C_l_r(beta, self.icing) * b / (2*airspeed)) * \
            yaw_dot + self.C_l_delta_a(beta, self.icing)*delta_a

    def pitch_moment_coeff(self, state: State, wind: np.ndarray) -> float:
        airspeed = np.sqrt(np.sum(calc_airspeed(state, wind)**2))
        alpha = calc_angle_of_attack(state, wind)*180.0/np.pi
        c = self.constants.mean_chord
        delta_e = self.control_input.elevator_deflection
        rot_airspeed = calc_rotational_airspeed(state, wind)
        pitch_dot = rot_airspeed[1]
        return self.C_m_alpha(alpha, self.icing) + \
            (self.C_m_q(alpha, self.icing) * c / (2*airspeed)) * pitch_dot + \
            self.C_m_delta_e(alpha, self.icing)*delta_e

    def yaw_moment_coeff(self, state: State, wind: np.ndarray) -> float:
        airspeed = np.sqrt(np.sum(calc_airspeed(state, wind)**2))
        beta = calc_angle_of_sideslip(state, wind)*180.0/np.pi
        b = self.constants.wing_span
        delta_a = self.control_input.aileron_deflection
        rot_airspeed = calc_rotational_airspeed(state, wind)
        roll_dot = rot_airspeed[0]
        yaw_dot = rot_airspeed[2]
        return self.C_n_beta(beta, self.icing) + \
            (self.C_n_p(beta, self.icing) * b / (2*airspeed)) * roll_dot + \
            (self.C_n_r(beta, self.icing) * b / (2*airspeed)) * \
            yaw_dot + self.C_n_delta_a(beta, self.icing)*delta_a


def iced_clean_split(data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    iced_data = []
    clean_data = []
    for row in data:
        if row[1] == 1:
            iced_data.append([row[0], row[2]])
        else:
            clean_data.append([row[0], row[2]])
    return np.array(iced_data), np.array(clean_data)


class InterpolatedProperty:
    def __init__(self, data: np.ndarray, bounds_error: bool = True, fill_value: float = 0.0):
        iced_data, clean_data = iced_clean_split(data)
        self.iced_interp = interp1d(
            iced_data[:, 0], iced_data[:, 1], bounds_error=bounds_error, fill_value=fill_value)
        self.clean_interp = interp1d(
            clean_data[:, 0], clean_data[:, 1], bounds_error=bounds_error, fill_value=fill_value)

    def __call__(self, angle: float, ice_level: float) -> float:
        return ice_level*self.iced_interp(angle) + (1.0 - ice_level)*self.clean_interp(angle)
