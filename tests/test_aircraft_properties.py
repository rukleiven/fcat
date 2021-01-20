import pytest
import numpy as np
from fcat import State
from fcat import Control_input
from fcat import IcedSkywalkerX8Properties


@pytest.mark.parametrize('state, expect', [(State(), 1.0),
                                           (State(), 1.0)])
def test_drag_coeff(state, expect):
    control_input = Control_input()
    prop = IcedSkywalkerX8Properties(control_input)
    wind = np.array([0.0, 0.0, 0.0])
    assert prop.drag_coeff(state, wind) == pytest.approx(expect)