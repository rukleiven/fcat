from fcat import AircraftProperties
from fcat import ControlInput, State
import numpy as np

__all__ = ('FrictionlessBall',)


class FrictionlessBall(AircraftProperties):
    """
    FrictonlessBall represents a dummy aircraft where all fluid mechanical forces are zero.
    It is primariy used for testing purposes.
    """

    def __init__(self, control_input: ControlInput):
        super().__init__(control_input)

    def drag_coeff(self, state: State, wind: np.ndarray) -> float:
        """
        :param state: State vector of the air-craft
        :param wind: Wind speed vector


        The drag force is given by

        F = 0.5*rho*C_drag*v^2

        where rho is the air density and C_drag is the drag coefficient returned by this function
        """
        return 0.0

    def lift_coeff(self, state: State, wind: np.ndarray) -> float:
        """
        :param state: State vector of the air-craft
        :param wind: Wind speed vector

        The drag force is given by

        F = 0.5*rho*C_lift*v^2

        where rho is the air density and C_lift is the lift coefficient returned by this function
        """
        return 0.0

    def side_force_coeff(self, state: State, wind: np.ndarray) -> float:
        """
        :param state: State vector of the air-craft
        :param wind: Wind speed vector


        The drag force is given by

        F = 0.5*rho*C_side_force*v^2

        where rho is the air density and C_side_force is the side force coefficient
        returned by this function
        """
        return 0.0

    def roll_moment_coeff(self, state: State, wind: np.ndarray) -> float:
        """
       :param state: State vector of the air-craft
       :param wind: Wind speed vector

       The drag force is given by

       F = 0.5*rho*C_l*v^2

       where rho is the air density and C_l is the rolling moment coefficient
       returned by this function
       """
        return 0.0

    def pitch_moment_coeff(self, state: State, wind: np.ndarray) -> float:
        """
        :param state: State vector of the air-craft
        :param wind: Wind speed vector

        The drag force is given by

        F = 0.5*rho*C_m*v^2

        where rho is the air density and C_m is the pitching moment coefficient
        returned by this function
        """
        return 0.0

    def yaw_moment_coeff(self, state: State, wind: np.ndarray) -> float:
        """
        :param state: State vector of the air-craft
        :param wind: Wind speed vector

        The drag force is given by

        F = 0.5*rho*C_n*v^2

        where rho is the air density and C_n is the yawing moment coefficient
        returned by this function
        """
        return 0.0

    def mass(self) -> float:
        """
        Aircraft mass
        """
        return 2

    def inertia_matrix(self):
        return 2.0*np.eye(3)

    def wing_area(self) -> float:
        """
        Return wing area
        """
        return 2

    def mean_chord(self) -> float:
        """
        Return mean chord of the wing
        """
        return 3

    def wing_span(self) -> float:
        """
        Return wing span
        """
        return 4

    def propeller_area(self) -> float:
        """
        Return area swept by propeller
        """
        return 0.0

    def motor_constant(self) -> float:
        """
        Return motor constant
        """
        return 40

    def motor_efficiency_fact(self) -> float:
        """
        Return motor efficiency factor
        """
        return 1
