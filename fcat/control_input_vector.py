import numpy as np

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
