
from control import input_output_response, StateSpace
from control.iosys import LinearIOSystem, summing_junction, InterconnectedSystem
import numpy as np
from fcat import (
    aircraft_property_from_dct, actuator_from_dct, ControlInput, State,
    build_nonlin_sys, no_wind, PropUpdate, PropertyUpdater
)
from fcat.utilities import add_controllers
from fcat.inner_loop_controller import (pitch_hinf_controller, roll_hinf_controller,
                                        airspeed_pi_controller, get_state_space_from_file,
                                        get_lateral_state_space, get_longitudinal_state_space)
from yaml import load
from matplotlib import pyplot as plt
from fcat.lateral_linear_model_builder import build_lin_sys
from fcat.lon_linear_model_builder import build_lin_sys_lon


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
        "kp": 0.5,
        "ki": 0.02,
        "kaw": 0.8,
        "throttle_trim": 0.58
    }
    print(np.linalg.eigvals(A_lat))
    lateral_controller = roll_hinf_controller(lateral_controller_params)
    longitudinal_controller = pitch_hinf_controller(longitudinal_controller_params)
    airspeed_controller = airspeed_pi_controller(airspeed_controller_params)

    config_filename = "examples/skywalkerx8_asymetric_linearize.yml"
    with open(config_filename, 'r') as infile:
        data = load(infile)

    aircraft = aircraft_property_from_dct(data['aircraft'])
    ctrl = ControlInput.from_dict(data['init_control'])
    state = State.from_dict(data['init_state'])

    # updates = {
    #    'icing': [PropUpdate(time=0.0, value=1.0),
    #              PropUpdate(time=15.0, value=0.0)],
    # }

    updates = {
        'icing_right_wing': [PropUpdate(time=0.0, value=0.0),
                             PropUpdate(time=5.0, value=0.5),
                             PropUpdate(time=18.0, value=0.0)],
        'icing_left_wing': [PropUpdate(time=0.0, value=0.0),
                            PropUpdate(time=5.0, value=0.5),
                            PropUpdate(time=25.0, value=0.0)]
    }

    updater = PropertyUpdater(updates)
    sys = build_nonlin_sys(aircraft, no_wind(), updater)
    actuator = actuator_from_dct(data['actuator'])

    closed_loop_system = add_controllers(
        actuator, sys, longitudinal_controller, lateral_controller, airspeed_controller)

    init_vec = np.zeros((33,))
    init_vec[0:4] = ctrl.control_input
    init_vec[4:16] = state.state
    init_vec[16:32] = [6.11521563e-01,  8.68884187e-02, -1.02427841e-01,  4.95947106e-25,
                       7.47240388e-01, -8.21880405e+00,  4.23430310e-01, -1.13786607e-05,
                       7.69914385e-03, 5.62092015e-03, 5.89381582e-04, 4.89020637e-02,
                       2.36717523e-02, -2.71712656e-04, 2.17217400e-02, 0]

    sim_time = 40
    t = np.linspace(0, sim_time, sim_time*5, endpoint=True)
    constant_input = np.array([21.401240221720634, 0.1369212836023969, -0.0])
    u_init = np.array([constant_input, ]*(100)).transpose()
    constant_input = np.array([21.401240221720634, 0.1369212836023969, -0.0])
    u_step = np.array([constant_input, ]*(100)).transpose()
    u = np.concatenate([u_init, u_step], axis=1)

    T, yout_non_lin, xout_non_lin = input_output_response(
        closed_loop_system, U=u, T=t, X0=init_vec, return_x=True, method='Radau')
    plot_respons(T[90:], xout_non_lin[4:16, 90:])
    plot_respons(T[90:], xout_non_lin[:4, 90:])
    plt.show()
# sim_aircraft()


def sim_lateral_lin_sys():
    lateral_controller_filename = "fcat/inner_loop_controller/lateral_controller.json"
    A_lat, B_lat, C_lat, D_lat = get_state_space_from_file(lateral_controller_filename)
    lateral_controller_params = {
        "A": A_lat,
        "B": B_lat,
        "C": C_lat,
        "D": D_lat
    }
    lateral_controller = roll_hinf_controller(lateral_controller_params)
    A, B, C, D = get_lateral_state_space('examples/skywalkerx8_linmod.json')
    D = np.matrix(D)

    lin_sys = build_lin_sys(A, B, C, D)
    feedback_summing_junction_roll = summing_junction(
        inputs=['roll_command', 'roll'], outputs='roll_e', name='fb')
    in_list = 'fb.roll_command'
    connections = [('roll_hinf_controller.roll_error', 'fb.roll_e'),
                   ('lin_sys.aileron_cmd', 'roll_hinf_controller.aileron_deflection_command'),
                   ('fb.roll', '-lin_sys.roll_out')]
    sys_list = (feedback_summing_junction_roll, lateral_controller, lin_sys)
    outlist = ['lin_sys.roll_out', 'fb.roll_e', 'roll_hinf_controller.aileron_deflection_command']
    iosys = InterconnectedSystem(sys_list, connections=connections, outputs=(
        'roll_out', 'roll_e', 'ail_deflection'), inplist=in_list, inputs='roll_command',
        outlist=outlist)
    sim_time = 100
    t = np.linspace(0, sim_time, sim_time*5, endpoint=True)
    init_vec = np.zeros((13,))
    u_1 = np.array([0]*(250)).transpose()
    u_2 = np.array([0.4]*(250)).transpose()
    u = np.concatenate([u_1, u_2], axis=0)
    T, yout_non_lin, xout_non_lin = input_output_response(
        iosys, U=u, T=t, X0=init_vec, return_x=True, method='Radau')
    print(iosys)
    print(np.amax(xout_non_lin[-4, :]))
    plot_respons(T, yout_non_lin[:, :])
    plot_respons(T, xout_non_lin[-5:, :])
    plt.show()


def sim_linear_lon_sys():
    longitudinal_controller_filename = "fcat/inner_loop_controller/longitudinal_controller.json"
    A_lon, B_lon, C_lon, D_lon = get_state_space_from_file(longitudinal_controller_filename)
    longitudinal_controller_params = {
        "A": A_lon,
        "B": B_lon,
        "C": C_lon,
        "D": D_lon
    }
    longitudinal_controller = pitch_hinf_controller(longitudinal_controller_params)
    A, B, C, D = get_longitudinal_state_space('examples/skywalkerx8_linmod.json')
    D = np.matrix(D)
    lin_sys = build_lin_sys_lon(A, B, C, D)
    feedback_summing_junction_pitch = summing_junction(
        inputs=['pitch_command', 'pitch'], outputs='pitch_e', name='fb')
    in_list = 'fb.pitch_command'
    connections = [('pitch_hinf_controller.pitch_error', 'fb.pitch_e'),
                   ('lin_sys.elev_cmd', 'pitch_hinf_controller.elevator_deflection_command'),
                   ('fb.pitch', '-lin_sys.pitch_out')]
    sys_list = (feedback_summing_junction_pitch, longitudinal_controller, lin_sys)
    outlist = ['lin_sys.pitch_out', 'fb.pitch_e',
               'pitch_hinf_controller.elevator_deflection_command']
    iosys = InterconnectedSystem(sys_list, connections=connections, outputs=(
        'pitch_out', 'pitch_e', 'elevator_deflection_command'), inplist=in_list,
        inputs='pitch_command', outlist=outlist)
    sim_time = 100
    print(iosys)
    t = np.linspace(0, sim_time, sim_time*5, endpoint=True)
    init_vec = np.zeros((14,))
    u_1 = np.array([0]*(450)).transpose()
    u_2 = np.array([0]*(50)).transpose()
    u = np.concatenate([u_1, u_2], axis=0)
    T, yout_non_lin, xout_non_lin = input_output_response(
        iosys, U=u, T=t, X0=init_vec, return_x=True, method='Radau')
    print(iosys)
    plot_respons(T, yout_non_lin[:, :])
    plot_respons(T, xout_non_lin[-6:, :])
    plt.show()


def sim_linear_model(filename):
    with open(filename, 'r') as infile:
        data = load(infile)

    A = np.array(data['A'])
    B = np.array(data['B'])
    C = np.array(data['C'])
    D = np.array(data['D'])
    xeq = np.array(data['xeq'])
    ueq = np.array(data['ueq'])
    ss_model = StateSpace(A, B, C, D)
    lin_sys = LinearIOSystem(ss_model)
    print(lin_sys)
    init_vec = np.zeros((16,))
    init_vec[:4] = ueq
    init_vec[4:] = xeq
    sim_time = 40
    constant_input = ueq
    u_init = np.array([constant_input, ]*(50)).transpose()
    constant_input = ueq
    u_step = np.array([constant_input, ]*(150)).transpose()
    u = np.concatenate([u_init, u_step], axis=1)
    t = np.linspace(0, sim_time, sim_time*5, endpoint=True)

    T, yout = input_output_response(lin_sys, t, U=u, X0=init_vec)
    plot_respons(T, yout[:, :])
    plt.show()


# sim_linear_model("examples/skywalkerX8_linmod_asym_icing.json")
sim_aircraft()
