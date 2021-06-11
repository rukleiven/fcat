from fcat.inner_loop_controller.controller_utilities import lateral_controller, longitudinal_controller
from examples.skywalkerX8_analysis.skywalkerX8_eigenvalue_analysis import plot_respons
from fcat.inner_loop_controller import (init_robust_controller, SaturatedStateSpaceController,
                                        init_airspeed_controller, get_state_space_from_file)
import numpy as np
from fcat.constants import Direction
from fcat import(
    State, build_nonlin_sys, no_wind, aircraft_property_from_dct, ControlInput, actuator_from_dct,
    PropUpdate, PropertyUpdater
)
from fcat.utilities import build_closed_loop, get_init_vector
from control import input_output_response
from matplotlib import pyplot as plt
from yaml import load


def plot_respons(t: np.ndarray, states: np.ndarray):
    fig = plt.figure()
    for i in range(states.shape[0]):
        ax = fig.add_subplot(3, 4, i+1)
        ax.plot(t, states[i, :])
        ax.set_xlabel("Time")
        ax.set_ylabel(f"State {i}")
    return fig

def main():
    # Script showing how to simulate icedskywalkerX8 from config_file using controllers saved in file
    config_file = "examples/skywalkerX8_linearize.yml"
    lat_controller_file = "examples/inner_loop_controllers/single_robust_roll_ctrl.json"
    lon_controller_file = "examples/inner_loop_controllers/single_robust_pitch_ctrl.json"
    K_lat = get_state_space_from_file(lat_controller_file)
    K_lon = get_state_space_from_file(lon_controller_file)
    K_lat = SaturatedStateSpaceController(A=np.array(K_lat.A), B=np.array(K_lat.B),
                                          C=np.array(K_lat.C), D=np.array(K_lat.D),
                                          lower=-0.4, upper=0.4)
    K_lon = SaturatedStateSpaceController(A=np.array(K_lon.A), B=np.array(K_lon.B),
                                          C=np.array(K_lon.C), D=np.array(K_lon.D),
                                          lower=-0.4, upper=0.4)    

    # Using similar controllers for simplicity
    lat_controller = init_robust_controller(K_lat, Direction.LATERAL)
    lon_controller = init_robust_controller(K_lon, Direction.LONGITUDINAL)
    airspeed_pi_controller = init_airspeed_controller()
    with open(config_file, 'r') as infile:
        data = load(infile)
    aircraft = aircraft_property_from_dct(data['aircraft'])
    ctrl = ControlInput.from_dict(data['init_control'])
    state = State.from_dict(data['init_state'])
    
    updates = {
        'icing': [PropUpdate(0.5, 0.5),
                            PropUpdate(5.0, 1.0)],
    }

    updater = PropertyUpdater(updates)

    aircraft_model = build_nonlin_sys(aircraft, no_wind(), outputs=State.names+['icing', 'airspeed'], prop_updater=updater) 
    actuator = actuator_from_dct(data['actuator'])
    closed_loop = build_closed_loop(actuator, aircraft_model, lon_controller, lat_controller, airspeed_pi_controller)

    X0 = get_init_vector(closed_loop, ctrl, state)
    constant_input = np.array([20, 0.2, -0.2])
    sim_time = 15
    t = np.linspace(0, sim_time, sim_time*5, endpoint=True)
    u = np.array([constant_input, ]*(len(t))).transpose()
    T, yout_non_lin, _ = input_output_response(
        closed_loop, U=u, T=t, X0=X0, return_x=True, method='BDF')
    plot_respons(T, yout_non_lin)
    plt.show()
main()