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

def test_ang_rate_x():
    state = State()
    ang_rate_x = 0.2
    state.ang_rate_x = ang_rate_x
    assert state.ang_rate_x == pytest.approx(ang_rate_x)
    assert np.allclose(state.state, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ang_rate_x, 0.0, 0.0])

def test_pitch():
    state = State()
    ang_rate_y = 0.2
    state.ang_rate_y = ang_rate_y
    assert state.ang_rate_y == pytest.approx(ang_rate_y)
    assert np.allclose(state.state, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ang_rate_y, 0.0])

def test_yaw():
    state = State()
    ang_rate_z = 0.2
    state.ang_rate_z = ang_rate_z
    assert state.ang_rate_z == pytest.approx(ang_rate_z)
    assert np.allclose(state.state, [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ang_rate_z])


@pytest.mark.parametrize('vx, vy, vz, want', [
    (1.0, 0.0, 0.0, 1.0),
    (1.0, -1.0, 0.0, np.sqrt(2.0)),
    (1.0, -1.0, 1.0, np.sqrt(3.0)),
    (0.0, 0.0, 0.0, 0.0),
])
def test_velocity(vx, vy, vz, want):
    state = State()
    state.vx = vx
    state.vy = vy
    state.vz = vz
    assert state.velocity == pytest.approx(want)

def test_from_dict_state():
    dct = {
        'x': 0.5,
        'y': 0.2,
        'z': 0,
        'roll': 0.5,
        'pitch': 0.5,
        'yaw': 0.2,
        'vx': 0,
        'vy': 0.5,
        'vz': 0.5,
        'ang_rate_x': 0.2,
        'ang_rate_y': 0,
        'ang_rate_z': 0.5
    }
    state = State.from_dict(dct)
    cotnrol_input_expect = np.array([0.5, 0.2, 0, 0.5, 0.5, 0.2, 0, 0.5, 0.5, 0.2, 0, 0.5])
    assert np.allclose(state.state,cotnrol_input_expect)