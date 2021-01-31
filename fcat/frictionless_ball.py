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
        return 0.0

    def lift_coeff(self, state: State, wind: np.ndarray) -> float:
        return 0.0

    def side_force_coeff(self, state: State, wind: np.ndarray) -> float:
        return 0.0

    def roll_moment_coeff(self, state: State, wind: np.ndarray) -> float:
        return 0.0

    def pitch_moment_coeff(self, state: State, wind: np.ndarray) -> float:
        return 0.0

    def yaw_moment_coeff(self, state: State, wind: np.ndarray) -> float:
        return 0.0

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
        return 0.0

    def motor_constant(self) -> float:
        return 40

    def motor_efficiency_fact(self) -> float:
        return 1
