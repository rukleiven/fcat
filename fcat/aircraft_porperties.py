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
    def lift_coeff(self, state: State, wind: np.ndarray) -> float:
        """
        :param state: State vector of the air-craft
        :param wind: wind vector

        The drag force is given by

        F = 0.5*rho*C_lift*v^2

        where rho is the air density and C_lift is the lift coefficient returned by this function
        """

    @abstractmethod    
    def side_force_coeff(self, state: State, wind: np.ndarray) -> float:
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
    def roll_moment_coeff(self, state: State, wind: np.ndarray) -> float:
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
    def pitch_moment_coeff(self, state: State, wind: np.ndarray) -> float:
        """
        :param state: State vector of the air-craft
        :param control_input: Control input vector of the aircraft
        :param alpha: Angle of attack

        The drag force is given by

        F = 0.5*rho*C_m*v^2

        where rho is the air density and C_m is the pitching moment coefficient returned by this function
        """


    @abstractmethod
    def yaw_moment_coeff(self, state: State, wind: np.ndarray) -> float:
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
    def __init__(self, control_input: Control_input, icing: float = 0.0):
        super().__init__(control_input)
        self.icing = icing
        self.constants = SkywalkerX8Constants()

    def mass(self):
        return self.constants.mass
    
    def inertia_matrix(self):
        return np.array([[self.constants.I_xx, self.constants.I_xy, self.constants.I_xz],
                         [self.constants.I_xy, self.constants.I_yy, self.constants.I_yz],
                         [self.constants.I_xz, self.constants.I_yz, self.constants.I_zz]])
    
    def drag_coeff(self, state: State, wind: np.ndarray) -> float:
        return 1.0
    
    def lift_coeff(self, state: State, wind: np.ndarray) -> float:
        return 1.0
    
    def side_force_coeff(self, state: State, wind: np.ndarray) -> float:
        return 1.0

    def roll_moment_coeff(self, state: State, wind: np.ndarray) -> float:
        return 1.0

    def pitch_moment_coeff(self, state: State, wind: np.ndarray) -> float:
        return 1.0
    
    def yaw_moment_coeff(self, state: State, wind: np.ndarray) -> float:
        return 1.0
