import numpy as np
from enum import IntEnum

__all__ = ('State',)


class StateVecIndices(IntEnum):
    """
    Enumerator that keeps track of which position the different control
    variables have in the underlying state vector
    """
    X = 0
    Y = 1
    Z = 2
    ROLL = 3
    PITCH = 4
    YAW = 5
    V_X = 6
    V_Y = 7
    V_Z = 8
    ROLL_DOT = 9
    PITCH_DOT = 10
    YAW_DOT = 11


class State:
    def __init__(self, init: np.ndarray = None):
        self.state = init
        if init is None:
            self.state = np.zeros(12)

        if len(self.state) != 12:
            raise ValueError("Length of state vector must be 12")

    @property
    def x(self) -> float:
        return self.state[StateVecIndices.X]

    @x.setter
    def x(self, value: float):
        self.state[StateVecIndices.X] = value

    @property
    def y(self) -> float:
        return self.state[StateVecIndices.Y]

    @y.setter
    def y(self, value: float):
        self.state[StateVecIndices.Y] = value

    @property
    def z(self) -> float:
        return self.state[StateVecIndices.Z]

    @z.setter
    def z(self, value: float):
        self.state[StateVecIndices.Z] = value

    @property
    def roll(self) -> float:
        return self.state[StateVecIndices.ROLL]

    @roll.setter
    def roll(self, value: float):
        self.state[StateVecIndices.ROLL] = value

    @property
    def pitch(self) -> float:
        return self.state[StateVecIndices.PITCH]

    @pitch.setter
    def pitch(self, value: float):
        self.state[StateVecIndices.PITCH] = value

    @property
    def yaw(self) -> float:
        return self.state[StateVecIndices.YAW]

    @yaw.setter
    def yaw(self, value: float):
        self.state[StateVecIndices.YAW] = value

    @property
    def vx(self) -> float:
        return self.state[StateVecIndices.V_X]

    @vx.setter
    def vx(self, value: float):
        self.state[StateVecIndices.V_X] = value

    @property
    def vy(self) -> float:
        return self.state[StateVecIndices.V_Y]

    @vy.setter
    def vy(self, value: float):
        self.state[StateVecIndices.V_Y] = value

    @property
    def vz(self) -> float:
        return self.state[StateVecIndices.V_Z]

    @vz.setter
    def vz(self, value: float):
        self.state[StateVecIndices.V_Z] = value

    @property
    def roll_dot(self) -> float:
        return self.state[StateVecIndices.ROLL_DOT]

    @roll_dot.setter
    def roll_dot(self, value: float):
        self.state[StateVecIndices.ROLL_DOT] = value

    @property
    def pitch_dot(self) -> float:
        return self.state[StateVecIndices.PITCH_DOT]

    @pitch_dot.setter
    def pitch_dot(self, value: float):
        self.state[StateVecIndices.PITCH_DOT] = value

    @property
    def yaw_dot(self) -> float:
        return self.state[StateVecIndices.YAW_DOT]

    @yaw_dot.setter
    def yaw_dot(self, value: float):
        self.state[StateVecIndices.YAW_DOT] = value
