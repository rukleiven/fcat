from typing import NamedTuple
from collections import namedtuple
import numpy as np

__all__ = ('StateSpaceMatrices', 'SaturatedStateSpaceController', 'SaturatedStateSpaceMatricesGS')


class StateSpaceMatrices(NamedTuple):
    A: np.ndarray
    B: np.ndarray
    C: np.ndarray
    D: np.ndarray


SaturatedStateSpaceController = namedtuple(
    'SaturatedStateSpaceController', StateSpaceMatrices._fields + ('lower', 'upper'))

SaturatedStateSpaceMatricesGS = namedtuple(
    'StateSpaceMatricesGS', SaturatedStateSpaceController._fields + ('switch_signal', ))
