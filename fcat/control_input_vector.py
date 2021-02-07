import numpy as np
from fcat.utilities import flying_wing2ctrl_input_matrix
__all__ = ('ControlInput',)


class ControlInput:
    """
    Vector with control signals used to control the aircraft actuators. The control signals
    are throttle and angular deflections of elevator, aileron and rudder.

    :param init: Initial value of the control vector. If not given, all elements will be
        initialized to zero. The input array should be of length 4 and contain the elements
        [elevator deflection, aileron deflection, rudder deflection, throttle]
        The deflection angles are given in radians and throttle is an indicator variable
        (0 <= throttle <= 1) where 0 corresponds to no thrust and 1 corresponds maximum thrust.
    """

    def __init__(self, init: np.ndarray = None):
        self.control_input = init
        if init is None:
            self.control_input = np.zeros(4)

        if len(self.control_input) != 4:
            raise ValueError("Length of control_input vector must be 4")

    @property
    def elevator_deflection(self) -> float:
        return self.control_input[0]

    @elevator_deflection.setter
    def elevator_deflection(self, value: float):
        self.control_input[0] = value

    @property
    def aileron_deflection(self) -> float:
        return self.control_input[1]

    @aileron_deflection.setter
    def aileron_deflection(self, value: float):
        self.control_input[1] = value

    @property
    def rudder_deflection(self) -> float:
        return self.control_input[2]

    @rudder_deflection.setter
    def rudder_deflection(self, value: float):
        self.control_input[2] = value

    @property
    def throttle(self) -> float:
        return self.control_input[3]

    @throttle.setter
    def throttle(self, value: float):
        self.control_input[3] = value

    @property
    def elevon_right(self) -> float:
        elevon_vec = self.aileron_elevator2elevon(self.control_input[:2])
        return elevon_vec[0]

    @elevon_right.setter
    def elevon_right(self, value: float):
        elevon_vec = self.aileron_elevator2elevon(self.control_input[:2])
        elevon_vec[0] = value
        self.control_input[:2] = self.elevon2aileron_elevator(elevon_vec)

    @property
    def elevon_left(self) -> float:
        elevon_vec = self.aileron_elevator2elevon(self.control_input[:2])
        return elevon_vec[1]

    @elevon_left.setter
    def elevon_left(self, value: float):
        elevon_vec = self.aileron_elevator2elevon(self.control_input[:2])
        elevon_vec[1] = value
        self.control_input[:2] = self.elevon2aileron_elevator(elevon_vec)

    def elevon2aileron_elevator(self, elev: np.ndarray) -> np.ndarray:
        """
        Transform flying-wing elevoninput values to aileron-elevator values

        :param elev: elevon input on form; [delta_er, delta_ea]
            where delta_er is angular deflection on right elevon and
            delta_ea is angular deflection on right elevon

        return vector on form [delta_e, delta_a]
        Where delta_e is angular deflection on elevator,
        and delta_a is angular deflection on ailerons.
        """
        transform_matrix = flying_wing2ctrl_input_matrix()[:2, :2]
        return transform_matrix.dot(elev)

    def aileron_elevator2elevon(self, ail_elev: np.ndarray) -> np.ndarray:
        """
        Transform aileron-elevator input to flying-wing elevon values

        :param ail_elev: aileron-elevator input on form [delta_e, delta_a]
            Where delta_e is angular deflection on elevator,
            and delta_a is angular deflection on ailerons.

        return vector on form; [delta_er, delta_ea]
        where delta_er is angular deflection on right elevon and
        delta_ea is angular deflection on right elevon
        """
        transform_matrix = np.linalg.inv(flying_wing2ctrl_input_matrix()[:2, :2])
        return transform_matrix.dot(ail_elev)
