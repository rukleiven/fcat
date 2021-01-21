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