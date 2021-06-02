
from control import input_output_response
import numpy as np
from fcat import (
    aircraft_property_from_dct, actuator_from_dct, ControlInput, State,
    build_nonlin_sys, PropUpdate, PropertyUpdater, DrydenGust, ConstantWind, no_wind
)
from fcat.utilities import add_controllers, create_aircraft_output_fonction
from fcat.inner_loop_controller import (pitch_hinf_controller, roll_hinf_controller,
                                        airspeed_pi_controller, get_state_space_from_file,
                                        pitch_gain_scheduled_controller,
                                        roll_gain_scheduled_controller)
from yaml import load
from matplotlib import pyplot as plt
import json


def compare_simulations(filename1: str, filename2: str):
    with open(filename1, 'r') as f:
        data1 = json.load(f)
    with open(filename2, 'r') as f:
        data2 = json.load(f)

    t1 = data1.get('time')
    yout1 = data1.get('y_out')

    yout2 = data2.get('y_out')
    fig = plt.figure(figsize=(7, 3))
    # xout2 = data2.get('x_out')
    # xout1 = data1.get('x_out')

    # ax = fig.add_subplot(1, 2, 1)
    # ax.plot(t1, xout1[1])
    # ax.plot(t1, xout2[1])
    # ax.set_xlabel("Time(s)")
    # ax.set_ylabel("Aileron deflection (rad)")

    # ax = fig.add_subplot(1, 2, 2)
    # ax.plot(t1, yout1[3])
    # ax.plot(t1, yout2[3])
    # ax.set_xlabel("Time(s)")
    # ax.set_ylabel("Roll angle (rad)")

    for i in range(len(yout1)):
        ax = fig.add_subplot(3, 4, i+1)
        ax.plot(t1, yout1[i])
        ax.plot(t1, yout2[i])
        ax.set_xlabel("Time")
        ax.set_ylabel(f"State {i}")
        if i == 0:
            labels = ("gs", "robust")
            plt.legend(labels)
    plt.show()

    return fig


def plot_respons(t: np.ndarray, states: np.ndarray):
    fig = plt.figure()

    for i in range(states.shape[0]):
        ax = fig.add_subplot(3, 4, i+1)
        ax.plot(t, states[i, :])
        ax.set_xlabel("Time")
        ax.set_ylabel(f"State {i}")
    return fig


def sim_aircraft(controller_type: str = "gs"):
    """
    This is a temporary script for simulating the aircraft system.
    """

    # TODO: Write a more comprehensive simulation script
    # Get closed-loop blocks files:
    lateral_controller_filename = "examples/lat_test_ctrl.json"
    longitudinal_controller_filename = 'examples/long_test_ctrl.json'

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
    lateral_controllers_filename = './examples/lateral_controllergs.json'
    with open(lateral_controllers_filename, 'r') as f:
        lateral_controllers = json.load(f)
    longitudinal_controllers_filename = \
        './examples/longitudinal_controllergs.json'
    with open(longitudinal_controllers_filename, 'r') as f:
        longitudinal_controllers = json.load(f)
    longitudinalgs_controller_params = {'controllers': longitudinal_controllers}
    lateralgs_controller_params = {'controllers': lateral_controllers}
    longitudinal_controller = pitch_gain_scheduled_controller(longitudinalgs_controller_params)
    lateral_controller = roll_gain_scheduled_controller(lateralgs_controller_params)

    out_filename = None
    if(controller_type == "robust"):
        longitudinal_controller = pitch_hinf_controller(longitudinal_controller_params)
        lateral_controller = roll_hinf_controller(lateral_controller_params)
        out_filename = "./examples/sim_res_robust.json"
    elif (controller_type == "gs"):
        out_filename = "./examples/sim_res_gs.json"
    airspeed_controller = airspeed_pi_controller(airspeed_controller_params)
    config_filename = "examples/skywalkerx8_asymmetric_linearize.yml"
    with open(config_filename, 'r') as infile:
        data = load(infile)
    aircraft = aircraft_property_from_dct(data['aircraft'])
    ctrl = ControlInput.from_dict(data['init_control'])
    state = State.from_dict(data['init_state'])

    updates = {
        'icing_left_wing': [PropUpdate(time=0.0, value=1.0),
                            PropUpdate(time=16.0, value=0.0)],
        'icing_right_wing': [PropUpdate(time=0.0, value=1.0),
                             PropUpdate(time=5.0, value=0.5)],
        'icing': [PropUpdate(time=0.0, value=1.0),
                  PropUpdate(time=5.0, value=1.0),
                  PropUpdate(time=6.0, value=0.0)]
    }

    updater = PropertyUpdater(updates)
    _ = updater
    config_dict = {"outputs": ["x", "y", "z", "roll", "pitch", "yaw",
                               "vx", "vy", "vz", "ang_rate_x", "ang_rate_y",
                               "ang_rate_z", "airspeed", "icing"]
                   }
    sim_time = 15
    t = np.linspace(0, sim_time, sim_time*5, endpoint=True)
    out_function = create_aircraft_output_fonction(config_dict)

    _ = ConstantWind(np.array([5, 5, 0, 0.0, 0.0, -0.0]))
    _ = DrydenGust(2.1, t, intensity=0)
    wind = no_wind()
    sys = build_nonlin_sys(
        aircraft, wind,
        outputs=config_dict["outputs"], prop_updater=updater, out_func=out_function)
    actuator = actuator_from_dct(data['actuator'])

    closed_loop_system = add_controllers(
        actuator, sys, longitudinal_controller, lateral_controller, airspeed_controller)

    syslist = (actuator, sys, longitudinal_controller, lateral_controller, airspeed_controller)
    nstates = 0
    for system in syslist:
        nstates += system.nstates
    init_vec = np.zeros((nstates,))
    init_vec[0:4] = ctrl.control_input
    init_vec[4:16] = state.state
    # init_vec[16:32] = [6.11521563e-01,  8.68884187e-02, -1.02427841e-01,  4.95947106e-25,
    #                    7.47240388e-01, -8.21880405e+00,  4.23430310e-01, -1.13786607e-05,
    #                    7.69914385e-03, 5.62092015e-03, 5.89381582e-04, 4.89020637e-02,
    #                    2.36717523e-02, -2.71712656e-04, 2.17217400e-02, 0]

    constant_input = np.array([20, 0.04, -0.00033310605950459315])
    u_init = np.array([constant_input, ]*(25)).transpose()
    constant_input = np.array([20, 0.04, -0.00033310605950459315])
    u_step = np.array([constant_input, ]*(50)).transpose()
    u = np.concatenate([u_init, u_step], axis=1)

    T, yout_non_lin, xout_non_lin = input_output_response(
        closed_loop_system, U=u, T=t, X0=init_vec, return_x=True, method='BDF')

    plot_respons(T, yout_non_lin[:, :])
    plot_respons(T, xout_non_lin[:4, :])
    plt.show()

    sim_res = {
        'y_out': yout_non_lin.tolist(),
        'x_out': xout_non_lin.tolist(),
        'time': T.tolist()
    }

    with open(out_filename, 'w') as outfile:
        json.dump(sim_res, outfile, indent=2, sort_keys=True)
    print(f"Results written to {out_filename}")


sim_aircraft("robust")
# compare_simulations("examples/sim_res_gs.json", "examples/sim_res_robust.json")
