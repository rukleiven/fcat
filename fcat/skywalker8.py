from typing import NamedTuple, Tuple
from fcat import AircraftProperties, ControlInput, State
import numpy as np
# (C_D_a_data, C_D_q_data, C_D_delta_e_data, C_L_a_data, C_L_q_data, C_L_delta_e_data)
from fcat.skywalkerX8_data import *
from fcat.utilities import calc_airspeed, calc_angle_of_attack, calc_angle_of_sideslip,\
    calc_rotational_airspeed, wind2body
from scipy.interpolate import interp1d

__all__ = ('IcedSkywalkerX8Properties', 'AsymetricIcedSkywalkerX8Properties', 'InterpolatedProperty')


class SkywalkerX8Constants(NamedTuple):
    wing_span: float = 2.1
    mean_chord: float = 0.3571
    wing_area: float = 0.75
    propeller_area: float = 0.1018
    motor_constant: float = 40
    motor_efficiency_fact: float = 1
    mass: float = 3.3650
    I_xx: float = 0.335
    I_xy: float = 0.0
    I_xz: float = -0.031
    I_yy: float = 0.140
    I_yz: float = 0.0
    I_zx: float = -0.029
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
        self.C_D_alpha = InterpolatedProperty(C_D_a_data, bounds_error=False,
                                              fill_value='extrapolate')
        self.C_D_q = InterpolatedProperty(C_D_q_data, bounds_error=False, fill_value='extrapolate')
        self.C_D_delta_e = InterpolatedProperty(
            C_D_delta_e_data, bounds_error=False, fill_value='extrapolate')
        self.C_L_alpha = InterpolatedProperty(C_L_a_data, bounds_error=False,
                                              fill_value='extrapolate')
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
        rot_airspeed_with_wind = calc_rotational_airspeed(state, wind)
        ang_rate_y_r = rot_airspeed_with_wind[1]
        # TODO: If C_D_q becomes non-zero, should test term containing C_D_q
        return self.C_D_alpha(alpha, self.icing) + \
            self.C_D_q(alpha, self.icing) * c / (2*airspeed) * ang_rate_y_r + \
            self.C_D_delta_e(alpha, self.icing)*np.abs(delta_e)

    def lift_coeff(self, state: State, wind: np.ndarray) -> float:
        airspeed = np.sqrt(np.sum(calc_airspeed(state, wind)**2))
        alpha = calc_angle_of_attack(state, wind)*180.0/np.pi
        c = self.constants.mean_chord
        delta_e = self.control_input.elevator_deflection
        rot_airspeed_with_wind = calc_rotational_airspeed(state, wind)
        ang_rate_y_r = rot_airspeed_with_wind[1]
        return self.C_L_alpha(alpha, self.icing) + \
            (self.C_L_q(alpha, self.icing) * c / (2*airspeed)) * ang_rate_y_r + \
            self.C_L_delta_e(alpha, self.icing)*delta_e

    def side_force_coeff(self, state: State, wind: np.ndarray) -> float:
        airspeed = np.sqrt(np.sum(calc_airspeed(state, wind)**2))
        beta = calc_angle_of_sideslip(state, wind)*180.0/np.pi
        b = self.constants.wing_span
        delta_a = self.control_input.aileron_deflection
        rot_airspeed_with_wind = calc_rotational_airspeed(state, wind)
        ang_rate_x_r = rot_airspeed_with_wind[0]
        ang_rate_z_r = rot_airspeed_with_wind[2]
        return self.C_Y_beta(beta, self.icing) + \
            (self.C_Y_p(beta, self.icing) * b / (2*airspeed)) * ang_rate_x_r + \
            (self.C_Y_r(beta, self.icing) * b / (2*airspeed)) * \
            ang_rate_z_r + self.C_Y_delta_a(beta, self.icing)*delta_a

    def roll_moment_coeff(self, state: State, wind: np.ndarray) -> float:
        airspeed = np.sqrt(np.sum(calc_airspeed(state, wind)**2))
        beta = calc_angle_of_sideslip(state, wind)*180.0/np.pi
        b = self.constants.wing_span
        delta_a = self.control_input.aileron_deflection
        rot_airspeed_with_wind = calc_rotational_airspeed(state, wind)
        ang_rate_x_r = rot_airspeed_with_wind[0]
        ang_rate_z_r = rot_airspeed_with_wind[2]
        return self.C_l_beta(beta, self.icing) + \
            (self.C_l_p(beta, self.icing) * b / (2*airspeed)) * ang_rate_x_r + \
            (self.C_l_r(beta, self.icing) * b / (2*airspeed)) * \
            ang_rate_z_r + self.C_l_delta_a(beta, self.icing)*delta_a

    def pitch_moment_coeff(self, state: State, wind: np.ndarray) -> float:
        airspeed = np.sqrt(np.sum(calc_airspeed(state, wind)**2))
        alpha = calc_angle_of_attack(state, wind)*180.0/np.pi
        c = self.constants.mean_chord
        delta_e = self.control_input.elevator_deflection
        rot_airspeed_with_wind = calc_rotational_airspeed(state, wind)
        ang_rate_y_r = rot_airspeed_with_wind[1]
        return self.C_m_alpha(alpha, self.icing) + \
            (self.C_m_q(alpha, self.icing) * c / (2*airspeed)) * ang_rate_y_r + \
            self.C_m_delta_e(alpha, self.icing)*delta_e

    def yaw_moment_coeff(self, state: State, wind: np.ndarray) -> float:
        airspeed = np.sqrt(np.sum(calc_airspeed(state, wind)**2))
        beta = calc_angle_of_sideslip(state, wind)*180.0/np.pi
        b = self.constants.wing_span
        delta_a = self.control_input.aileron_deflection
        rot_airspeed_with_wind = calc_rotational_airspeed(state, wind)
        ang_rate_x_r = rot_airspeed_with_wind[0]
        ang_rate_z_r = rot_airspeed_with_wind[2]
        return self.C_n_beta(beta, self.icing) + \
            (self.C_n_p(beta, self.icing) * b / (2*airspeed)) * ang_rate_x_r + \
            (self.C_n_r(beta, self.icing) * b / (2*airspeed)) * \
            ang_rate_z_r + self.C_n_delta_a(beta, self.icing)*delta_a

    def wing_span(self) -> float:
        return self.constants.wing_span

    def mean_chord(self) -> float:
        return self.constants.mean_chord

    def wing_area(self) -> float:
        return self.constants.wing_area

    def propeller_area(self) -> float:
        return self.constants.propeller_area

    def motor_constant(self) -> float:
        return self.constants.motor_constant

    def motor_efficiency_fact(self) -> float:
        return self.constants.motor_efficiency_fact

    def update_params(self, params: dict) -> None:
        self.icing = params.get('icing', self.icing)


class AsymetricIcedSkywalkerX8Properties(AircraftProperties):
    """
    Properties for the SkywalkerX8 airplane. Parmaeter value are found
    in ...
    """

    def __init__(self, control_input: ControlInput, icing_left_wing: float = 0.0,
                 icing_right_wing: float = 0.0):
        self.left_wing = IcedSkywalkerX8Properties(control_input, icing_left_wing)
        self.right_wing = IcedSkywalkerX8Properties(control_input, icing_right_wing)
        super().__init__(control_input)

        # Distance vectors from CoG to right wing center of pressure
        self.lfpoa_rw = np.array([0.0, 0.4, 0.0])  # lift force point of attack right wing
        self.dfpoa_rw = np.array([0.0, 0.25, 0.0])  # m
        self.sfpoa_rw = np.array([0.0, 0.2, 0.0])  # m

        # Distance vectors from CoG to right wing center of pressure
        self.lfpoa_lw = np.array([0.0, -0.4, 0.0])  # m
        self.dfpoa_lw = np.array([0.0, -0.25, 0.0])  # m
        self.sfpoa_lw = np.array([0.0, -0.2, 0.0])  # m

    @property
    def control_input(self):
        # Right wing is always equal to left wing
        return self.left_wing.control_input

    @control_input.setter
    def control_input(self, control_input: ControlInput):
        self.left_wing.control_input = control_input
        self.right_wing.control_input = control_input

    def drag_coeff(self, state: State, wind: np.ndarray) -> float:
        return 1/2*(self.left_wing.drag_coeff(state, wind) +
                    self.right_wing.drag_coeff(state, wind))

    def lift_coeff(self, state: State, wind: np.ndarray) -> float:
        return 1/2*(self.left_wing.lift_coeff(state, wind) +
                    self.right_wing.lift_coeff(state, wind))

    def side_force_coeff(self, state: State, wind: np.ndarray) -> float:
        return 1/2*(self.left_wing.side_force_coeff(state, wind) +
                    self.right_wing.side_force_coeff(state, wind))

    def roll_moment_coeff(self, state: State, wind: np.ndarray) -> float:
        asym_icing_moment = self.asymetric_moment_contribution(state, wind)
        asym_roll_moment = asym_icing_moment[0]
        return (1/2)*(self.left_wing.roll_moment_coeff(state, wind) +
                      self.right_wing.roll_moment_coeff(state, wind)) + \
            asym_roll_moment

    def pitch_moment_coeff(self, state: State, wind: np.ndarray) -> float:
        asym_icing_moment = self.asymetric_moment_contribution(state, wind)
        asym_pitch_moment = asym_icing_moment[1]
        return (1/2)*(self.left_wing.pitch_moment_coeff(state, wind) +
                      self.right_wing.pitch_moment_coeff(state, wind)) + \
            asym_pitch_moment

    def yaw_moment_coeff(self, state: State, wind: np.ndarray) -> float:
        asym_icing_moment = self.asymetric_moment_contribution(state, wind)
        asym_yaw_moment = asym_icing_moment[2]
        return (1/2)*(self.left_wing.yaw_moment_coeff(state, wind) +
                      self.right_wing.yaw_moment_coeff(state, wind)) + \
            asym_yaw_moment

    def update_params(self, params: dict) -> None:
        self.right_wing.update_params({'icing': params.get('icing_left_wing', self.left_wing.icing)})
        self.left_wing.update_params(
            {'icing': params.get('icing_right_wing', self.right_wing.icing)})

    def asymetric_moment_contribution(self, state: State, wind: np.ndarray) -> np.ndarray:
        f_rw_wind = (1/2)*np.array([-self.right_wing.drag_coeff(state, wind),
                                    self.right_wing.side_force_coeff(state, wind),
                                    -self.right_wing.lift_coeff(state, wind)])

        f_lw_wind = (1/2)*np.array([-self.left_wing.drag_coeff(state, wind),
                                    self.left_wing.side_force_coeff(state, wind),
                                    -self.left_wing.lift_coeff(state, wind)])

        drag_rw_body = wind2body(
            [f_rw_wind[0], 0, 0], state, wind)
        drag_lw_body = wind2body(
            [f_lw_wind[0], 0, 0], state, wind)

        side_rw_body = wind2body(
            [0, f_rw_wind[1], 0], state, wind)
        side_lw_body = wind2body(
            [0, f_lw_wind[1], 0], state, wind)

        lift_rw_body = wind2body(
            [0, 0, f_rw_wind[2]], state, wind)
        lift_lw_body = wind2body(
            [0, 0, f_lw_wind[2]], state, wind)

        drag_asym_moment_left_wing = np.cross(
            self.dfpoa_lw, drag_lw_body)
        side_force_asym_moment_left_wing = np.cross(
            self.sfpoa_lw, side_lw_body)
        lift_asym_moment_left_wing = np.cross(
            self.lfpoa_lw, lift_lw_body)
        asym_moment_left_wing = drag_asym_moment_left_wing + side_force_asym_moment_left_wing\
            + lift_asym_moment_left_wing
        drag_asym_moment_right_wing = np.cross(
            self.dfpoa_rw, drag_rw_body)
        side_force_asym_moment_right_wing = np.cross(
            self.dfpoa_rw, side_rw_body)
        lift_asym_moment_right_wing = np.cross(
            self.lfpoa_rw, lift_rw_body)
        asym_moment_right_wing = drag_asym_moment_right_wing + side_force_asym_moment_right_wing\
            + lift_asym_moment_right_wing
        asym_moment = asym_moment_left_wing + asym_moment_right_wing
        return asym_moment

    def wing_span(self) -> float:
        return self.left_wing.wing_span()

    def mean_chord(self) -> float:
        return self.left_wing.mean_chord()

    def wing_area(self) -> float:
        return self.left_wing.wing_area()

    def propeller_area(self) -> float:
        return self.left_wing.propeller_area()

    def motor_constant(self) -> float:
        return self.left_wing.motor_constant()

    def motor_efficiency_fact(self) -> float:
        return self.left_wing.motor_efficiency_fact()

    def mass(self):
        return self.left_wing.mass()

    def inertia_matrix(self):
        return self.left_wing.inertia_matrix()


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
