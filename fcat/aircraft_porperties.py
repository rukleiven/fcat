from abc import ABC, abstractmethod
import numpy as np
from fcat import State
from fcat import Control_input

__all__ = ('AircraftProperties', 'IcedSkywalkerX8Properties')


class AircraftProperties(ABC):
    """
    Class collecting all force functions required to describe the dynamics of an aircraft.
    """
    def __init__(self, control_input: Control_input):
        self.control_input = control_input

    @abstractmethod
    def drag_coeff(self, state: State, alpha: float) -> float:
        """
        :param state: State vector of the air-craft
        :param control_input: Control input vector of the aircraft
        :param alpha: Angle of attack


        The drag force is given by

        F = 0.5*rho*C_drag*v^2

        where rho is the air density and C_drag is the drag coefficient returned by this function
        """

    @abstractmethod
    def lift_coeff(self, state: State, control_input: Control_input, alpha: float) -> float:
        """
        :param state: State vector of the air-craft
        :param control_input: Control input vector of the aircraft
        :param alpha: Angle of attack

        The drag force is given by

        F = 0.5*rho*C_lift*v^2

        where rho is the air density and C_lift is the lift coefficient returned by this function
        """

    @abstractmethod    
    def side_force_coeff(self, state: State, control_input: Control_input, beta: float) -> float:
        """
        :param state: State vector of the air-craft
        :param control_input: Control input vector of the aircraft
        :param beta: Angle of sideslip


        The drag force is given by

        F = 0.5*rho*C_side_force*v^2

        where rho is the air density and C_side_force is the side force coefficient 
        returned by this function
        """


    @abstractmethod
    def roll_moment_coeff(self, state: State, control_input: Control_input, beta: float) -> float:
         """
        :param state: State vector of the air-craft
        :param control_input: Control input vector of the aircraft
        :param beta: Angle of sideslip

        The drag force is given by

        F = 0.5*rho*C_l*v^2

        where rho is the air density and C_l is the rolling moment coefficient 
        returned by this function
        """


    @abstractmethod
    def pitch_moment_coeff(self, state: State, control_input: Control_input, alpha: float) -> float:
        """
        :param state: State vector of the air-craft
        :param control_input: Control input vector of the aircraft
        :param alpha: Angle of attack

        The drag force is given by

        F = 0.5*rho*C_m*v^2

        where rho is the air density and C_m is the pitching moment coefficient returned by this function
        """


    @abstractmethod
    def yaw_moement_coeff(self, state: State, control_input: Control_input, beta: float) -> float:
        """
        :param state: State vector of the air-craft
        :param control_input: Control input vector of the aircraft
        :param beta: Angle of sideslip

        The drag force is given by

        F = 0.5*rho*C_n*v^2

        where rho is the air density and C_n is the yawing moment coefficient returned by this function
        """

    @abstractmethod
    def mass(self) -> float:
        """
        Aircraft mass 
        """
    
    @abstractmethod
    def inertia_matrix(self) -> np.ndarray:
        """
        Aircraft inertia matrix:
        [ I_xx  I_xy  I_xz
          I_yx  I_yy  I_yz
          I_zx  I_zy  I_zz
        ]
        """
        


class IcedSkywalkerX8Properties(AircraftProperties):
    """
    Properties for the SkywalkerX8 airplane. Parmaeter value are found
    in ...
    """
    def __init__(self, wind: np.ndarray, icing: float = 0.0, ):
        super().__init__(wind)
        self.icing = icing
        self.wing_span = 2.1
        self.mean_chord = 0.3571
        self.wing_area = 0.75
        self.motor_constant = 40
        self.motor_efficiency_fact = 1
        self.I_xx = 0.340
        self.I_xy = 0.0
        self.I_xz = -0.031
        self.I_yx = 0.0
        self.I_yy = 0.165
        self.I_yz = 0.0
        self.I_zx = -0.031
        self.I_zy = 0.0
        self.I_zz =0.400

    def mass(self):
        return 3.3650
    
    def inertia_matrix(self):
        return np.array([[self.I_xx, self.I_xy, self.I_xz], [self.I_yx, self.I_yy, self.I_yz], [self.I_zx, self.I_zy, self.I_zz]])
    
    def drag_coeff(self, state: State) -> float:
        return 1.0
