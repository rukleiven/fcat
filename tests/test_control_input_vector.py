import pytest
import numpy as np
from fcat import Control_input

def test_delta_e():
    control_input = Control_input()
    delta_e = 0.2
    control_input.delta_e = delta_e
    assert control_input.delta_e == pytest.approx(delta_e)
    assert np.allclose(control_input.control_input, [delta_e, 0.0, 0.0, 0.0])

def test_delta_a():
    control_input = Control_input()
    delta_a = 0.2
    control_input.delta_a = delta_a
    assert control_input.delta_a == pytest.approx(delta_a)
    assert np.allclose(control_input.control_input, [0.0, delta_a, 0.0, 0.0])

def test_delta_r():
    control_input = Control_input()
    delta_r = 0.2
    control_input.delta_r = delta_r
    assert control_input.delta_r == pytest.approx(delta_r)
    assert np.allclose(control_input.control_input, [0.0, 0.0, delta_r, 0.0])

def test_delta_t():
    control_input = Control_input()
    delta_t = 0.2
    control_input.delta_t = delta_t
    assert control_input.delta_t == pytest.approx(delta_t)
    assert np.allclose(control_input.control_input, [0.0, 0.0, 0.0, delta_t])