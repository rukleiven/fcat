import numpy as np

__all__ = ('ControlInput',)


class ControlInput:
    """
    Vector with control signals used to control the aircraft actuators. The control signals
    are throttle and angular deflections of elevator, aileron and rudder.
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
