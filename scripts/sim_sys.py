
from control import input_output_response
import numpy as np
from fcat import (
    aircraft_property_from_dct, actuator_from_dct, ControlInput, State,
    build_nonlin_sys, no_wind, PropUpdate, PropertyUpdater
)
from fcat.utilities import add_controllers
from fcat.inner_loop_controller import (pitch_hinf_controller, roll_hinf_controller,
                                        airspeed_pi_controller, get_state_space_from_file)
from yaml import load
from matplotlib import pyplot as plt


def plot_respons(t: np.ndarray, states: np.ndarray):
    fig = plt.figure()

    for i in range(states.shape[0]):
        ax = fig.add_subplot(3, 4, i+1)
        ax.plot(t, states[i, :])
        ax.set_xlabel("Time")
        ax.set_ylabel(f"State {i}")
    return fig


def sim_aircraft():
    """
    This is a temporary script for simulating the aircraft system.
    """
    # TODO: Write a more comprehensive simulation script
    # Get closed-loop blocks files:
    lateral_controller_filename = "examples/lateral_controller.json"
    longitudinal_controller_filename = 'examples/longitudinal_controller.json'

    A_lat, B_lat, C_lat, D_lat = get_state_space_from_file(lateral_controller_filename)
    A_lon, B_lon, C_lon, D_lon = get_state_space_from_file(longitudinal_controller_filename)

    lateral_controller_params = {
        "A": A_lat,
        "B": B_lat,
        "C": C_lat,
        "D": D_lat
    }

    longitudinal_controller_params = {
        "A": A_lon,
        "B": B_lon,
        "C": C_lon,
        "D": D_lon
    }

    airspeed_controller_params = {
        "kp": 0.123,
        "ki": 0.09,
        "kaw": 2,
        "throttle_trim": 0.58
    }

    lateral_controller = roll_hinf_controller(lateral_controller_params)
    longitudinal_controller = pitch_hinf_controller(longitudinal_controller_params)
    airspeed_controller = airspeed_pi_controller(airspeed_controller_params)

    config_filename = "examples/skywalkerx8_linearize.yml"
    with open(config_filename, 'r') as infile:
        data = load(infile)

    aircraft = aircraft_property_from_dct(data['aircraft'])
    ctrl = ControlInput.from_dict(data['init_control'])
    state = State.from_dict(data['init_state'])

    updates = {
        'icing': [PropUpdate(time=0.0, value=1.0),
                  PropUpdate(time=20.0, value=1.0)]
    }

    updater = PropertyUpdater(updates)
    sys = build_nonlin_sys(aircraft, no_wind(), updater)
    actuator = actuator_from_dct(data['actuator'])

    closed_loop_system = add_controllers(
        actuator, sys, longitudinal_controller, lateral_controller, airspeed_controller)

    init_vec = np.zeros((32,))
    init_vec[0:4] = ctrl.control_input
    init_vec[4:16] = state.state
    init_vec[16:31] = [6.11521563e-01,  8.68884187e-02, -1.02427841e-01,  4.95947106e-25,
                       7.47240388e-01, -8.21880405e+00,  4.23430310e-01, -1.13786607e-05,
                       7.69914385e-03, 5.62092015e-03, 5.89381582e-04, 4.89020637e-02,
                       2.36717523e-02, -2.71712656e-04, 2.17217400e-02]

    sim_time = 30
    t = np.linspace(0, sim_time, sim_time*5, endpoint=True)
    constant_input = np.array([21.401240221720634, 0.1369212836023969, -0.00033310605950459315])
    u_init = np.array([constant_input, ]*(50)).transpose()
    constant_input = np.array([21.401240221720634, 0.1369212836023969, -0.00033310605950459315])
    u_step = np.array([constant_input, ]*(100)).transpose()
    u = np.concatenate([u_init, u_step], axis=1)

    T, yout_non_lin, xout_non_lin = input_output_response(
        closed_loop_system, U=u, T=t, X0=init_vec, return_x=True, method='Radau')
    print(yout_non_lin[4, :])

    # fig = plot_respons(T, yout_non_lin[:, :])
    plt.show()


sim_aircraft()
