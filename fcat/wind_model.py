from abc import ABC, abstractmethod
import numpy as np

__all__ = ('WindModel', 'ConstantWind', 'no_wind')


class WindModel(ABC):
    """
    Class that represents a general wind model
    """
    @abstractmethod
    def get(self, t: float) -> np.ndarray:
        """
        Return the wind as an array of length 6 at the given time.
        The three first are translational wind speeds, and the three last
        are rotational speeds.

        :param t: Time
        """


class ConstantWind(WindModel):
    """
    Simple wind model where the wind is a constant vector
    """
    def __init__(self, wind: np.ndarray):
        self.wind = wind

    def get(self, t: float) -> np.ndarray:
        return self.wind


def no_wind() -> ConstantWind:
    return ConstantWind(np.zeros(6))
