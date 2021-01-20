import numpy as np

__all__ = ('State',)


class State:
    def __init__(self, init: np.ndarray = None):
        self.state = init
        if init is None:
            self.state = np.zeros(12)
        
        if len(self.state) != 12:
            raise ValueError("Length of state vector must be 12")

    @property
    def x(self) -> float:
        return self.state[0]

    @x.setter
    def x(self, value: float):
        self.state[0] = value

    @property
    def y(self) -> float:
        return self.state[1]

    @y.setter
    def y(self, value: float):
        self.state[1] = value

    @property
    def z(self) -> float:
        return self.state[2]

    @z.setter
    def z(self, value: float):
        self.state[2] = value

    @property
    def roll(self) -> float:
        return self.state[3]

    @roll.setter
    def roll(self, value: float):
        self.state[3] = value

    @property
    def pitch(self) -> float:
        return self.state[4]

    @pitch.setter
    def pitch(self, value: float):
        self.state[4] = value

    @property
    def yaw(self) -> float:
        return self.state[5]

    @yaw.setter
    def yaw(self, value: float):
        self.state[5] = value

    @property
    def vx(self) -> float:
        return self.state[6]

    @vx.setter
    def vx(self, value: float):
        self.state[6] = value

    @property
    def vy(self) -> float:
        return self.state[7]

    @vy.setter
    def vy(self, value: float):
        self.state[7] = value

    @property
    def vz(self) -> float:
        return self.state[8]

    @vz.setter
    def vz(self, value: float):
        self.state[8] = value

    @property
    def roll_dot(self) -> float:
        return self.state[9]

    @roll_dot.setter
    def roll_dot(self, value: float):
        self.state[9] = value

    @property
    def pitch_dot(self) -> float:
        return self.state[10]

    @pitch_dot.setter
    def pitch_dot(self, value: float):
        self.state[10] = value

    @property
    def yaw_dot(self) -> float:
        return self.state[11]

    @yaw_dot.setter
    def yaw_dot(self, value: float):
        self.state[11] = value
