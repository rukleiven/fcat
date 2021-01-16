import pytest
import numpy as np
from fcat import State

def test_x():
    state = State()
    x = 0.2
    state.x = x
    assert state.x == pytest.approx(x)
    assert np.allclose(state.state, [x, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

def test_y():
    state = State()
    y = 0.2
    state.y = y
    assert state.y == pytest.approx(y)
    assert np.allclose(state.state, [0.0, y, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

def test_z():
    state = State()
    z = 0.2
    state.z = z
    assert state.z == pytest.approx(z)
    assert np.allclose(state.state, [0.0, 0.0, z, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

def test_roll():
    state = State()
    roll = 0.2
    state.roll = roll
    assert state.roll == pytest.approx(roll)
    assert np.allclose(state.state, [0.0, 0.0, 0.0, roll, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

def test_pitch():
    state = State()
    pitch = 0.2
    state.pitch = pitch
    assert state.pitch == pytest.approx(pitch)
    assert np.allclose(state.state, [0.0, 0.0, 0.0, 0.0, pitch, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

def test_yaw():
    state = State()
    yaw = 0.2
    state.yaw = yaw
    assert state.yaw == pytest.approx(yaw)
    assert np.allclose(state.state, [0.0, 0.0, 0.0, 0.0, 0.0, yaw, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])


def test_vx():
    state = State()
    vx = 0.2
    state.vx = vx
    assert state.vx == pytest.approx(vx)
    assert np.allclose(state.state, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, vx, 0.0, 0.0, 0.0, 0.0, 0.0])

def test_vy():
    state = State()
    vy = 0.2
    state.vy = vy
    assert state.vy == pytest.approx(vy)
    assert np.allclose(state.state, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, vy, 0.0, 0.0, 0.0, 0.0])

def test_vz():
    state = State()
    vz = 0.2
    state.vz = vz
    assert state.vz == pytest.approx(vz)
    assert np.allclose(state.state, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, vz, 0.0, 0.0, 0.0])

def test_roll_dot():
    state = State()
    roll_dot = 0.2
    state.roll_dot = roll_dot
    assert state.roll_dot == pytest.approx(roll_dot)
    assert np.allclose(state.state, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, roll_dot, 0.0, 0.0])

def test_pitch():
    state = State()
    pitch_dot = 0.2
    state.pitch_dot = pitch_dot
    assert state.pitch_dot == pytest.approx(pitch_dot)
    assert np.allclose(state.state, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, pitch_dot, 0.0])

def test_yaw():
    state = State()
    yaw_dot = 0.2
    state.yaw_dot = yaw_dot
    assert state.yaw_dot == pytest.approx(yaw_dot)
    assert np.allclose(state.state, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, yaw_dot])