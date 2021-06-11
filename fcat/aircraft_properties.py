from abc import ABC, abstractmethod
import numpy as np
from fcat import State, ControlInput

__all__ = ('AircraftProperties',)


class AircraftProperties(ABC):
    """
    Class collecting fluid mechanical coefficients needed to describe dynamics of an aircraft.
    AircraftProperties represent an airplane where the control inputs are fixed.
    Thus, concrete implementations of this class, should return the fluid mechanical coefficients
    when the control variables are given. See py:class`IcedSkywalkerX8Properties` for an example.

    :param control_input: Control variables
    """

    def __init__(self, control_input: ControlInput):
        self.control_input = control_input

    @abstractmethod
    def drag_coeff(self, state: State, wind: np.ndarray) -> float:
        """
        :param state: State vector of the air-craft
        :param wind: Wind vector: Array of 6 elements, containing the wind contribution
            to aircraft translational and angular velocity in body frame on form
            [v_wx, v_wy, v_wz, ang_rate_w_x, ang_rate_w_y, ang_rate_w_z]

        The drag force is given by;
        F = 0.5*rho*wing_area*C_drag*v^2

        where rho is the air density and C_drag is the drag coefficient returned by this function
        """

    @abstractmethod
    def lift_coeff(self, state: State, wind: np.ndarray) -> float:
        """
        :param state: State vector of the air-craft
        :param wind: Wind vector: Array of 6 elements, containing the wind contribution
            to aircraft translational and angular velocity in body frame on form
            [v_wx, v_wy, v_wz, ang_rate_w_x, ang_rate_w_y, ang_rate_w_z]

        The lift force is given by

        F = 0.5*rho*wing_area*C_lift*v^2

        where rho is the air density and C_lift is the lift coefficient returned by this function
        """

    @abstractmethod
    def side_force_coeff(self, state: State, wind: np.ndarray) -> float:
        """
        :param state: State vector of the air-craft
        :param wind: Wind vector: Array of 6 elements, containing the wind contribution
            to aircraft translational and angular velocity in body frame on form
            [v_wx, v_wy, v_wz, ang_rate_w_x, ang_rate_w_y, ang_rate_w_z]

        The side force is given by

        F = 0.5*rho*wing_area*C_side_force*v^2

        where rho is the air density and C_side_force is the side force coefficient
        returned by this function
        """

    @abstractmethod
    def roll_moment_coeff(self, state: State, wind: np.ndarray) -> float:
        """
       :param state: State vector of the air-craft
       :param wind: Wind vector: Array of 6 elements, containing the wind contribution
            to aircraft translational and angular velocity in body frame on form
            [v_wx, v_wy, v_wz, ang_rate_w_x, ang_rate_w_y, ang_rate_w_z]

       The roll moment is given by

       F = 0.5*rho*wing_area*wing_span*C_l*v^2

       where rho is the air density and C_l is the rolling moment coefficient
       returned by this function
       """

    @abstractmethod
    def pitch_moment_coeff(self, state: State, wind: np.ndarray) -> float:
        """
        :param state: State vector of the air-craft
        :param wind: Wind vector: Array of 6 elements, containing the wind contribution
            to aircraft translational and angular velocity in body frame on form
            [v_wx, v_wy, v_wz, ang_rate_w_x, ang_rate_w_y, ang_rate_w_z]

        The pitch moment is given by

        F = 0.5*rho*wing_area*mean_chord*C_m*v^2

        where rho is the air density and C_m is the pitching moment coefficient
        returned by this function
        """

    @abstractmethod
    def yaw_moment_coeff(self, state: State, wind: np.ndarray) -> float:
        """
        :param state: State vector of the air-craft
        :param wind: Wind vector: Array of 6 elements, containing the wind contribution
            to aircraft translational and angular velocity in body frame on form
            [v_wx, v_wy, v_wz, ang_rate_w_x, ang_rate_w_y, ang_rate_w_z]

        The yaw moment is given by

        F = 0.5*rho*wing_area*wing_span*C_n*v^2

        where rho is the air density and C_n is the yawing moment coefficient
        returned by this function
        """

    @abstractmethod
    def mass(self) -> float:
        """
        Return the aircraft mass
        """

    def inv_inertia_matrix(self) -> np.ndarray:
        """
        Return the inverse inertia matrix of the aircraft, where the inertia
        matrix is of the form:
        [ I_xx  I_xy  I_xz
          I_yx  I_yy  I_yz
          I_zx  I_zy  I_zz
        ]
        """
        return np.linalg.inv(self.inertia_matrix())

    @abstractmethod
    def inertia_matrix(self) -> np.ndarray:
        """
        Return the inertia matrix of the aircraft on form:
        [ I_xx  I_xy  I_xz
          I_yx  I_yy  I_yz
          I_zx  I_zy  I_zz
        ]
        """

    @abstractmethod
    def wing_area(self) -> float:
        """
        Return wing area
        """

    @abstractmethod
    def mean_chord(self) -> float:
        """
        Return mean chord of the wing
        """

    @abstractmethod
    def wing_span(self) -> float:
        """
        Return wing span
        """
    @abstractmethod
    def propeller_area(self) -> float:
        """
        The propulsion force in body frame is given by;
        F_p = 0.5*rho*S_p*C_p*([(k_m*delta_t)^2-V_a^2, 0, 0])

        Where rho is air density, C_p is motor efficiency factor, k_m is motor constant,
        delta_t is throttle input, V_a is airspeed and
        S_p is the area swept by the propeller returned by this function
        """

    @abstractmethod
    def motor_constant(self) -> float:
        """
        The propulsion force in body frame is given by;
        F_p = 0.5*rho*S_p*C_p*([(k_m*delta_t)^2-V_a^2, 0, 0])

        Where rho is air density, C_p is motor efficiency factor,
        S_p is the area swept by the propeller, delta_t is throttle input, V_a is airspeed and
        k_m is the motor constant returned by this function
        """

    @abstractmethod
    def motor_efficiency_fact(self) -> float:
        """
        The propulsion force in body frame is given by;
        F_p = 0.5*rho*S_p*C_p*([(k_m*delta_t)^2-V_a^2, 0, 0])

        Where rho is air density, k_m is the motor constant,
        S_p is the area swept by the propeller, delta_t is throttle input, V_a is airspeed and
        C_p is motor efficiency factor returned by this function
        """

    def update_params(self, params: dict) -> None:
        """
        Update subclass specific types given as key-value pairs the passed dictionary

        :param params: Subclass specific parameters
        """
        pass

    def is_known_output(self, output: str) -> bool:
        """
        Return True if aircraft property can be returned

        :param output: output name
        """
        return False

    def get_output(self, output: str) -> float:
        """
        Return outåut value

        :param output: output name
        """
        return 0.0
