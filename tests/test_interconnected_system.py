from fcat.utilities import add_actuator
from fcat import (IcedSkywalkerX8Properties, no_wind, build_nonlin_sys, ControlInput,
                  build_flying_wing_actuator_system, State)
from control.iosys import input_output_response
import numpy as np

def test_interconnected_system():
    control_input = ControlInput()
    control_input.throttle = 0.5
    wind_model = no_wind()
    state = State()
    state.vx = 20
    prop = IcedSkywalkerX8Properties(control_input)
    aircraft_model = build_nonlin_sys(prop,wind_model)

    initial_control_input_state = ControlInput()
    initial_control_input_state.throttle = 0.4
    motor_time_constant = 0.001
    elevon_time_constant = 0.001
    actuator_model = build_flying_wing_actuator_system(elevon_time_constant, motor_time_constant)
    x0 = np.concatenate((initial_control_input_state.control_input, state.state))
    connected_system = add_actuator(actuator_model, aircraft_model)
    t = np.linspace(0.0, 0.5, 10, endpoint=True)
    u = np.array([control_input.control_input,]*len(t)).transpose()
    T, yout_without_actuator = input_output_response(aircraft_model, t, U=u, X0=state.state)
    T, yout_with_actuator = input_output_response(connected_system, t, U=u, X0=x0)
    assert np.allclose(yout_with_actuator[6,:], yout_without_actuator[6,:], atol = 5.e-3)
    assert np.allclose(yout_with_actuator[7,:], yout_without_actuator[7,:], atol = 5.e-3)
    assert np.allclose(yout_with_actuator[8,:], yout_without_actuator[8,:], atol = 5.e-3)
    assert np.allclose(yout_with_actuator[9,:], yout_without_actuator[9,:], atol = 5.e-3)
    assert np.allclose(yout_with_actuator[10,:], yout_without_actuator[10,:], atol = 5.e-3)
    assert np.allclose(yout_with_actuator[11,:], yout_without_actuator[11,:], atol = 5.e-3)




