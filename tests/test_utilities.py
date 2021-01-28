import pytest
import numpy as np
from fcat import State
from fcat.utilities import body2wind, inertial2body

def body2wind_test_cases():
    state1 = State()
    vec1 = np.array([1.0, 0.0, 0.0])
    wind1 = np.zeros(6)
    expect1 = np.array([1.0, 0.0, 0.0])

    state2 = State()
    state2.vx = 10.0
    vec2 = np.array([1.0, 0.0, 0.0])
    wind2 = np.array([0.0, 10.0, 0.0])
    expect2 = np.array([np.sqrt(2.0)/2.0, -np.sqrt(2.0)/2.0, 0.0])

    state3 = State()
    state3.vx = 10.0
    vec3 = np.array([0.0, 1.0, 0.0])
    wind3 = np.array([0.0, 10.0, 0.0])
    expect3 = np.array([np.sqrt(2.0)/2.0, np.sqrt(2.0)/2.0, 0.0])

    state4 = State()
    state4.vx = 10.0
    state4.vz = 10.0
    vec4 = np.array([1.0, 0.0, 0.0])
    wind4 = np.array([0.0, 0.0, 0.0])
    expect4 = np.array([np.sqrt(2.0)/2.0, 0.0, -np.sqrt(2.0)/2.0])
    return [(state1, vec1, wind1, expect1),
            (state2, vec2, wind2, expect2),
            (state3, vec3, wind3, expect3),
            (state4, vec4, wind4, expect4)]

@pytest.mark.parametrize('state, vec, wind, expect', body2wind_test_cases())
def test_body2wind(state, vec, wind, expect):
    rotated = body2wind(vec, state, wind)
    assert np.allclose(rotated, expect)

def test_inertial2body_test_cases():
    state1 = State()
    state1.roll = np.pi/4.0
    vec1 = np.array([0.0, 0.0, 1.0])
    expect1 = np.array([0.0, np.sqrt(2.0)/2.0, np.sqrt(2.0)/2.0])

    state2 = State()
    state2.pitch = np.pi/4.0
    vec2 = np.array([0.0, 0.0, 1.0])
    expect2 = np.array([-np.sqrt(2.0)/2.0, 0.0, np.sqrt(2.0)/2.0])

    state3 = State()
    state3.yaw = np.pi/4.0
    vec3 = np.array([1.0, 0.0, 0.0])
    expect3 = np.array([np.sqrt(2.0)/2.0, -np.sqrt(2.0)/2.0, 0.0])
    return [(state1, vec1, expect1),
            (state2, vec2, expect2),
            (state3, vec3, expect3)]

@pytest.mark.parametrize('state, vec, expect', test_inertial2body_test_cases())
def test_inertial2body(state, vec, expect):
    rotated = inertial2body(vec, state)
    assert np.allclose(rotated, expect)