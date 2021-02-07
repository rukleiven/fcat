import pytest
import numpy as np
from fcat import ControlInput


def test_elevator_deflection():
    control_input = ControlInput()
    elevator_deflection = 0.2
    control_input.elevator_deflection = elevator_deflection
    assert control_input.elevator_deflection == pytest.approx(elevator_deflection)
    assert np.allclose(control_input.control_input, [elevator_deflection, 0.0, 0.0, 0.0])

def test_aileron_deflection():
    control_input = ControlInput()
    aileron_deflection = 0.2
    control_input.aileron_deflection = aileron_deflection
    assert control_input.aileron_deflection == pytest.approx(aileron_deflection)
    assert np.allclose(control_input.control_input, [0.0, aileron_deflection, 0.0, 0.0])

def test_rudder_deflection():
    control_input = ControlInput()
    rudder_deflection = 0.2
    control_input.rudder_deflection = rudder_deflection
    assert control_input.rudder_deflection == pytest.approx(rudder_deflection)
    assert np.allclose(control_input.control_input, [0.0, 0.0, rudder_deflection, 0.0])

def test_throttle():
    control_input = ControlInput()
    throttle = 0.2
    control_input.throttle = throttle
    assert control_input.throttle == pytest.approx(throttle)
    assert np.allclose(control_input.control_input, [0.0, 0.0, 0.0, throttle])

def test_flying_wing_get():
    control_input = ControlInput()
    throttle = 0.2
    aileron = 0.2
    elevator = 0.2
    control_input.throttle = throttle
    control_input.aileron = aileron
    control_input.elevator = elevator
    expect_elevon = control_input.aileron_elevator2elevon(control_input.control_input[:2])
    expect_flying_wing_input = np.array([expect_elevon[0], expect_elevon[1], throttle])
    assert control_input.throttle == pytest.approx(throttle)
    assert np.allclose(np.array([control_input.elevon_right, control_input.elevon_left, control_input.throttle]), expect_flying_wing_input)

def test_flying_wing_setters():
    control_input = ControlInput()
    throttle = 1
    elevon_left = 2
    elevon_right = 1
    elevon_vec = np.array([elevon_right, elevon_left])
    control_input.elevon_left = elevon_left
    control_input.elevon_right = elevon_right
    control_input.throttle = throttle
    expect_throttle = 1
    expect_elev_ail = control_input.elevon2aileron_elevator(elevon_vec)
    expect_control_input = np.array([expect_elev_ail[0], expect_elev_ail[1], 0, expect_throttle])
    assert np.allclose(control_input.control_input, expect_control_input)

def test_aileron_elevator2elevon():
    control_input = ControlInput()
    test_vec1 = np.zeros(2)
    test_vec2 = np.ones(2)
    val1 = control_input.aileron_elevator2elevon(test_vec1)
    val2 = control_input.aileron_elevator2elevon(test_vec2)
    expect1 = np.zeros(2)
    expect2 = np.array([0, 2])
    assert np.allclose(val1,expect1)
    assert np.allclose(val2,expect2)

def test_elevon2aileron_elevator():
    control_input = ControlInput()
    test_vec1 = np.zeros(2)
    control_input.control_input[:2] = test_vec1
    val1 = control_input.elevon2aileron_elevator(test_vec1)
    test_vec2 = np.ones(2)
    control_input.control_input[:2] = test_vec2
    val2 = control_input.elevon2aileron_elevator(test_vec2)
    expect1 = np.zeros(2)
    expect2 = np.array([1, 0])
    assert np.allclose(val1,expect1)
    assert np.allclose(val2,expect2)

def test_from_dict_ctrl_vect():
    dct = {
        'elevator_deflection': 0.5,
        'aileron_deflection': 0.2,
        'rudder_deflection': 0,
        'throttle': 0.5
    }
    control_input = ControlInput.from_dict(dct)
    cotnrol_input_expect = np.array([0.5, 0.2, 0, 0.5])
    assert np.allclose(control_input.control_input,cotnrol_input_expect)