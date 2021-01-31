from fcat import AircraftProperties
from fcat import ControlInput, State
import numpy as np
from fcat.simulation_constants import AIR_DENSITY
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
        V_a = np.sqrt(np.sum(calc_airspeed(state, wind)**2))
        # Test func: F_drag = abs(delta_e)-> C_drag = (2/(wing_area*AIR_DENSITY*V_a^2)* abs(delta_e)
        return 2/(self.wing_area()*AIR_DENSITY*V_a**2)*np.abs(self.control_input.elevator_deflection)

    def lift_coeff(self, state: State, wind: np.ndarray) -> float:
        V_a = np.sqrt(np.sum(calc_airspeed(state, wind)**2))
        # Test func: F_lift = delta_a -> C_lift = 2/(wing_area*AIR_DENSITY*V_a^2)
        return 2/(self.wing_area()*AIR_DENSITY*V_a**2)*self.control_input.aileron_deflection

    def side_force_coeff(self, state: State, wind: np.ndarray) -> float:
        V_a = np.sqrt(np.sum(calc_airspeed(state, wind)**2))
        # Test func: F_side_force = delta_r -> C_lift = 2/(wing_area*AIR_DENSITY*V_a^2)
        return 2/(self.wing_area()*AIR_DENSITY*V_a**2)*self.control_input.rudder_deflection

    def roll_moment_coeff(self, state: State, wind: np.ndarray) -> float:
        return 0

    def pitch_moment_coeff(self, state: State, wind: np.ndarray) -> float:
        return 0

    def yaw_moment_coeff(self, state: State, wind: np.ndarray) -> float:
        return 0

    def mass(self) -> float:
        return 2

    def inertia_matrix(self):
        return 2.0*np.eye(3)

    def wing_area(self) -> float:
        return 2

    def mean_chord(self) -> float:
        return 3

    def wing_span(self) -> float:
        return 4

    def propeller_area(self) -> float:
        return 1

    def motor_constant(self) -> float:
        return 2

    def motor_efficiency_fact(self) -> float:
        return 2/AIR_DENSITY


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
        return 0

    def lift_coeff(self, state: State, wind: np.ndarray) -> float:
        return 0

    def side_force_coeff(self, state: State, wind: np.ndarray) -> float:
        return 0

    def roll_moment_coeff(self, state: State, wind: np.ndarray) -> float:
        V_a = np.sqrt(np.sum(calc_airspeed(state, wind)**2))
        # Test func: Roll_moment = delta_e
        # -> C_roll_moment = (2/(wing_area*wing_span*AIR_DENSITY*V_a^2)*delta_e
        return 2/(self.wing_area()*self.wing_span()*AIR_DENSITY*V_a**2) *\
            self.control_input.elevator_deflection

    def pitch_moment_coeff(self, state: State, wind: np.ndarray) -> float:
        V_a = np.sqrt(np.sum(calc_airspeed(state, wind)**2))
        # Test func: Pitch_moment = delta_a
        # -> C_pitch_moment = (2/(wing_area*mean_chord*AIR_DENSITY*V_a^2)*delta_a
        return 2/(self.wing_area()*self.mean_chord()*AIR_DENSITY*V_a**2) *\
            self.control_input.aileron_deflection

    def yaw_moment_coeff(self, state: State, wind: np.ndarray) -> float:
        V_a = np.sqrt(np.sum(calc_airspeed(state, wind)**2))
        # Test func: yaw_moment = delta_r
        # -> C_yaw_moment = (2/(wing_area*AIR_DENSITY*wing_span*V_a^2)*delta_r
        return 2/(self.wing_area()*self.wing_span()*AIR_DENSITY*V_a**2) *\
            self.control_input.rudder_deflection

    def mass(self) -> float:
        return 2

    def inertia_matrix(self):
        return np.array([[self.I_xx, self.I_xy, self.I_xz],
                         [self.I_xy, self.I_yy, self.I_yz],
                         [self.I_xz, self.I_yz, self.I_zz]])

    def wing_area(self) -> float:
        return 2

    def mean_chord(self) -> float:
        return 3

    def wing_span(self) -> float:
        return 4

    def propeller_area(self) -> float:
        return 0  # Propulsion Force= 0

    def motor_constant(self) -> float:
        return 0

    def motor_efficiency_fact(self) -> float:
        return 2/AIR_DENSITY
