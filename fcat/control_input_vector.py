import numpy as np

__all__ = ('Control_input')


class Control_input:
    def __init__(self, init: np.ndarray = None):
        self.control_input = init
        if init is None:
            self.control_input = np.zeros(4)
        
        if len(self.control_input) != 4:
            raise ValueError("Length of control_input vector must be 4")

    @property
    def delta_e(self) -> float:
        return self.control_input[0]

    @delta_e.setter
    def delta_e(self, value: float):
        self.control_input[0] = value

    @property
    def delta_a(self) -> float:
        return self.control_input[1]

    @delta_a.setter
    def delta_a(self, value: float):
        self.control_input[1] = value

    @property
    def delta_r(self) -> float:
        return self.control_input[2]

    @delta_r.setter
    def delta_r(self, value: float):
        self.control_input[2] = value

    @property
    def delta_t(self) -> float:
        return self.control_input[3]

    @delta_t.setter
    def delta_t(self, value: float):
        self.control_input[3] = value

