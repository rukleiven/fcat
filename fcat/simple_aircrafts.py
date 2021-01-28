from fcat import AircraftProperties
from fcat import ControlInput, State
import numpy as np
from fcat.simulation_constants import RHO
from fcat.utilities import calc_airspeed

__all__ = ('SimpleTestAircraftNoMoments', 'SimpleTestAircraftNoForces')


class SimpleTestAircraftNoMoments(AircraftProperties):
    """
    SimpleAircraft represents a dummy aircraft where all fluid mechanical forces
    and moments are given by simple equations of control-input;
    Drag_Force = abs(elevator_deflection)
    Lift Force = rudder_deflection
    Side Force = aileron_deflection
    Rolling moment = 0
    Pitch moment = 0
    Yawing moment = 0

    It is primariy used for testing purposes.
    """

    def __init__(self, control_input: ControlInput):
        super().__init__(control_input)

    def drag_coeff(self, state: State, wind: np.ndarray) -> float:
        """
        :param state: State vector of the air-craft
        :param wind: Wind speed vector


        The drag force is given by

        F = 0.5*rho*wing_area*C_drag*v^2
        where rho is the air density and C_drag is the drag coefficient returned by this function
        """
        V_a = np.sqrt(np.sum(calc_airspeed(state, wind)**2))
        # Test func: F_drag = abs(delta_e)-> C_drag = (2/(wing_area*RHO*V_a^2)* abs(delta_e)
        return 2/(self.wing_area()*RHO*V_a**2)*np.abs(self.control_input.elevator_deflection)

    def lift_coeff(self, state: State, wind: np.ndarray) -> float:
        """
        :param state: State vector of the air-craft
        :param wind: Wind speed vector

        The drag force is given by

        F = 0.5*rho*wing_area*C_lift*v^2

        where rho is the air density and C_lift is the lift coefficient returned by this function
        """
        V_a = np.sqrt(np.sum(calc_airspeed(state, wind)**2))
        # Test func: F_lift = delta_e -> C_lift = 2/(wing_area*RHO*V_a^2)
        return 2/(self.wing_area()*RHO*V_a**2)*self.control_input.aileron_deflection

    def side_force_coeff(self, state: State, wind: np.ndarray) -> float:
        """
        :param state: State vector of the air-craft
        :param wind: Wind speed vector


        The drag force is given by

        F = 0.5*rho*wing_area*C_side_force*v^2

        where rho is the air density and C_side_force is the side force coefficient
        returned by this function
        """
        V_a = np.sqrt(np.sum(calc_airspeed(state, wind)**2))
        return 2/(self.wing_area()*RHO*V_a**2)*self.control_input.rudder_deflection

    def roll_moment_coeff(self, state: State, wind: np.ndarray) -> float:
        """
       :param state: State vector of the air-craft
       :param wind: Wind speed vector

       The drag force is given by

       TAU = 0.5*rho*wing_area*wing_span*C_l*v^2

       where rho is the air density and C_l is the rolling moment coefficient
       returned by this function
       """
        return 0

    def pitch_moment_coeff(self, state: State, wind: np.ndarray) -> float:
        """
        :param state: State vector of the air-craft
        :param wind: Wind speed vector

        The drag force is given by

        TAU = 0.5*rho*wing_area*mean_chord*C_m*v^2

        where rho is the air density and C_m is the pitching moment coefficient
        returned by this function
        """
        return 0

    def yaw_moment_coeff(self, state: State, wind: np.ndarray) -> float:
        """
        :param state: State vector of the air-craft
        :param wind: Wind speed vector

        The drag force is given by

         TAU = 0.5*rho*wing_area*wing_span*C_n*v^2

        where rho is the air density and C_n is the yawing moment coefficient
        returned by this function
        """
        return 0

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
        return 1

    def motor_constant(self) -> float:
        """
        Return motor constant
        """

        return 2

    def motor_efficiency_fact(self) -> float:
        """
        Return motor efficiency factor
        """
        return 2/RHO


class SimpleTestAircraftNoForces(AircraftProperties):
    """
    SimpleAircraft represents a dummy aircraft where all fluid mechanical forces
    and moments are given by simple equations of control-input;
    Drag_Force = abs(elevator_deflection)
    Lift Force = rudder_deflection
    Side Force = aileron_deflection
    Rolling moment = 0
    Pitch moment = 0
    Yawing moment = 0

    It is primariy used for testing purposes.
    """

    def __init__(self, control_input: ControlInput):
        super().__init__(control_input)
        self.I_xx, self.I_yy, self.I_zz = 2, 2, 2
        self.I_xy = 0
        self.I_yz = 0
        self.I_xz = 1

    def drag_coeff(self, state: State, wind: np.ndarray) -> float:
        """
        :param state: State vector of the air-craft
        :param wind: Wind speed vector


        The drag force is given by

        F = 0.5*rho*wing_area*C_drag*v^2
        where rho is the air density and C_drag is the drag coefficient returned by this function
        """
        return 0

    def lift_coeff(self, state: State, wind: np.ndarray) -> float:
        """
        :param state: State vector of the air-craft
        :param wind: Wind speed vector

        The drag force is given by

        F = 0.5*rho*wing_area*C_lift*v^2

        where rho is the air density and C_lift is the lift coefficient returned by this function
        """
        return 0

    def side_force_coeff(self, state: State, wind: np.ndarray) -> float:
        """
        :param state: State vector of the air-craft
        :param wind: Wind speed vector


        The drag force is given by

        F = 0.5*rho*wing_area*C_side_force*v^2

        where rho is the air density and C_side_force is the side force coefficient
        returned by this function
        """
        return 0

    def roll_moment_coeff(self, state: State, wind: np.ndarray) -> float:
        """
       :param state: State vector of the air-craft
       :param wind: Wind speed vector

       The drag force is given by

       TAU = 0.5*rho*wing_area*wing_span*C_l*v^2

       where rho is the air density and C_l is the rolling moment coefficient
       returned by this function
       """
        V_a = np.sqrt(np.sum(calc_airspeed(state, wind)**2))
        # Test func: F_drag = abs(delta_e)-> C_drag = (2/(wing_area*RHO*V_a^2)*delta_e
        return 2/(self.wing_area()*self.wing_span()*RHO*V_a**2) *\
            self.control_input.elevator_deflection

    def pitch_moment_coeff(self, state: State, wind: np.ndarray) -> float:
        """
        :param state: State vector of the air-craft
        :param wind: Wind speed vector

        The drag force is given by

        TAU = 0.5*rho*wing_area*mean_chord*C_m*v^2

        where rho is the air density and C_m is the pitching moment coefficient
        returned by this function
        """
        V_a = np.sqrt(np.sum(calc_airspeed(state, wind)**2))
        # Test func: F_drag = abs(delta_e)-> C_drag = (2/(wing_area*RHO*V_a^2)*delta_e
        return 2/(self.wing_area()*self.mean_chord()*RHO*V_a**2) *\
            self.control_input.aileron_deflection

    def yaw_moment_coeff(self, state: State, wind: np.ndarray) -> float:
        """
        :param state: State vector of the air-craft
        :param wind: Wind speed vector

        The drag force is given by

         TAU = 0.5*rho*wing_area*wing_span*C_n*v^2

        where rho is the air density and C_n is the yawing moment coefficient
        returned by this function
        """
        V_a = np.sqrt(np.sum(calc_airspeed(state, wind)**2))
        # Test func: F_drag = delta_e -> C_drag = (2/(wing_area*RHO*V_a^2)* abs(delta_e)
        return 2/(self.wing_area()*self.wing_span()*RHO*V_a**2)*self.control_input.rudder_deflection

    def mass(self) -> float:
        """
        Aircraft mass
        """
        return 2

    def inertia_matrix(self):
        return np.array([[self.I_xx, self.I_xy, self.I_xz],
                         [self.I_xy, self.I_yy, self.I_yz],
                         [self.I_xz, self.I_yz, self.I_zz]])

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
        return 0  # Propulsion Force= 0

    def motor_constant(self) -> float:
        """
        Return motor constant
        """

        return 0

    def motor_efficiency_fact(self) -> float:
        """
        Return motor efficiency factor
        """
        return 2/RHO
