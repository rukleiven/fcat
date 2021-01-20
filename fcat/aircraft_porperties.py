from abc import ABC, abstractmethod
from typing import NamedTuple
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
        

class SkywalkerX8Constants(NamedTuple):
    wing_span: float
    mean_chord: float
    wing_area: float
    motor_constant: float
    motor_efficiency_fact: float
    mass: float
    I_xx: float
    I_xy: float
    I_xz: float
    I_yx: float
    I_yy: float
    I_yz: float
    I_yz: float
    I_yz: float
    I_zx: float
    I_zy: float
    I_zz: float

def default_skywalkerX8_constants() -> SkywalkerX8Constants:
    constants = SkywalkerX8Constants()
    constants.wing_span = 2.1
    constants.mean_chord = 0.3571
    constants.wing_area = 0.75
    constants.motor_constant = 40
    constants.motor_efficiency_fact = 1
    constants.mass = 3.3650
    constants.I_xx = 0.340
    constants.I_xy = 0.0
    constants.I_xz = -0.031
    constants.I_yx = 0.0
    constants.I_yy = 0.165
    constants.I_yz = 0.0
    constants.I_zx = -0.031
    constants.I_zy = 0.0
    constants.I_zz = 0.400
    return constants


class IcedSkywalkerX8Properties(AircraftProperties):
    """
    Properties for the SkywalkerX8 airplane. Parmaeter value are found
    in ...
    """
    def __init__(self, wind: np.ndarray, icing: float = 0.0):
        super().__init__(wind)
        self.icing = icing
        self.constants = default_skywalkerX8_constants()

    def mass(self):
        return self.constants.mass
    
    def inertia_matrix(self):
        return np.array([[self.constants.I_xx, self.constants.I_xy, self.constants.I_xz],
                         [self.constants.I_yx, self.constants.I_yy, self.constants.I_yz],
                         [self.constants.I_zx, self.constants.I_zy, self.constants.I_zz]])
    
    def drag_coeff(self, state: State) -> float:
        return 1.0
