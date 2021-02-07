from typing import Dict
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
    ANG_RATE_X = 9
    ANG_RATE_Y = 10
    ANG_RATE_Z = 11


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
    def ang_rate_x(self) -> float:
        return self.state[StateVecIndices.ANG_RATE_X]

    @ang_rate_x.setter
    def ang_rate_x(self, value: float):
        self.state[StateVecIndices.ANG_RATE_X] = value

    @property
    def ang_rate_y(self) -> float:
        return self.state[StateVecIndices.ANG_RATE_Y]

    @ang_rate_y.setter
    def ang_rate_y(self, value: float):
        self.state[StateVecIndices.ANG_RATE_Y] = value

    @property
    def ang_rate_z(self) -> float:
        return self.state[StateVecIndices.ANG_RATE_Z]

    @ang_rate_z.setter
    def ang_rate_z(self, value: float):
        self.state[StateVecIndices.ANG_RATE_Z] = value

    @property
    def velocity(self) -> float:
        return np.sqrt(self.vx**2 + self.vy**2 + self.vz**2)

    @staticmethod
    def from_dict(dct: Dict[str, float]):
        state = State()
        state.x = dct['x']
        state.y = dct['y']
        state.z = dct['z']
        state.roll = dct['roll']
        state.yaw = dct['yaw']
        state.pitch = dct['pitch']
        state.vx = dct['vx']
        state.vy = dct['vy']
        state.vz = dct['vz']
        state.ang_rate_x = dct['ang_rate_x']
        state.ang_rate_y = dct['ang_rate_y']
        state.ang_rate_z = dct['ang_rate_z']
        return state
