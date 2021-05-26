from fcat.utilities import flying_wing2ctrl_input_matrix
from fcat import ControlInput, build_flying_wing_actuator_system
from control import input_output_response
import numpy as np


def test_actuator_model():
    control_input = ControlInput()
    control_input.elevon_right = 2
    control_input.elevon_left = 3
    control_input.throttle = 0.5
    control_input.rudder = 0

    initial_value_elevator = 3
    initial_value_aileron = 2
    initial_value_throttle = 2
    initial_values = np.array(
        [initial_value_elevator, initial_value_aileron, 0, initial_value_throttle])

    initial_values_flying_wing = np.linalg.inv(flying_wing2ctrl_input_matrix()).dot(initial_values)
    initial_value_elevon_right = initial_values_flying_wing[0]
    initial_value_elevon_left = initial_values_flying_wing[1]
    initial_value_throttle = initial_values_flying_wing[3]

    elevon_time_constant = 0.3
    motor_time_constant = 0.2

    lin_model = build_flying_wing_actuator_system(elevon_time_constant, motor_time_constant)

    t = np.linspace(0.0, 10, 500, endpoint=True)
    u = np.array([control_input.control_input, ]*len(t)).transpose()
    T, yout = input_output_response(lin_model, t, U=u, X0=initial_values)

    yout = np.linalg.inv(flying_wing2ctrl_input_matrix()).dot(yout)
    expect_elevon_r = control_input.elevon_right + \
        (initial_value_elevon_right - control_input.elevon_right)*np.exp(-t/elevon_time_constant)
    expect_elevon_l = control_input.elevon_left + \
        (initial_value_elevon_left - control_input.elevon_left)*np.exp(-t/elevon_time_constant)
    expect_motor = control_input.throttle + \
        (initial_value_throttle - control_input.throttle)*np.exp(-t/motor_time_constant)
    assert np.allclose(expect_elevon_r, yout[0, :], atol=5e-3)
    assert np.allclose(expect_elevon_l, yout[1, :], atol=5e-3)
    assert np.allclose(expect_motor, yout[3, :], atol=5e-3)
