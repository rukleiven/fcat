from examples.skywalkerX8_analysis.skywalkerX8_eigenvalue_analysis import plot_respons
from fcat.inner_loop_controller import (init_gs_controller, SaturatedStateSpaceMatricesGS,
                                        init_airspeed_controller)
import numpy as np
from fcat.constants import Direction
from fcat import(
    State, build_nonlin_sys, no_wind, IcedSkywalkerX8Properties, ControlInput, build_flying_wing_actuator_system
)
from fcat.utilities import build_closed_loop, get_init_vector
from control import input_output_response

def main():
    # Initialize gain scheduled controller
    # Using two simple P-controllers, see also fcat.longitudinal_controller for more complicated examples
    # Note that this is just an example on how to build and simulate an aircraft system
    # The controllers are simple not tuned to get desired reference tracking

    controllers = [
        SaturatedStateSpaceMatricesGS(
            A = -1*np.eye(1),
            B = np.zeros((1,1)),
            C = np.eye(1),
            D = np.zeros((1,1)),
            upper = 1,
            lower = -1,
            switch_signal = 0.3
        ), 
        SaturatedStateSpaceMatricesGS(
            A = -1*np.eye(1),
            B = np.zeros((1,1)),
            C = np.eye(1),
            D = np.zeros((1,1)),
            upper = 1,
            lower = -1,
            switch_signal = 0.6
        )
    ]
    # Using similar controllers for simplicity
    lat_gs_controller = init_gs_controller(controllers, Direction.LATERAL, 'icing')
    lon_gs_controller = init_gs_controller(controllers, Direction.LONGITUDINAL, 'icing')
    airspeed_pi_controller = init_airspeed_controller()
    control_input = ControlInput()
    control_input.throttle = 0.5
    state = State()
    state.vx = 20
    prop = IcedSkywalkerX8Properties(control_input)
    
    aircraft_model = build_nonlin_sys(prop, no_wind(), outputs=State.names+['icing', 'airspeed']) 
    motor_time_constant = 0.001
    elevon_time_constant = 0.001

    actuator_model = build_flying_wing_actuator_system(elevon_time_constant, motor_time_constant)
    closed_loop = build_closed_loop(actuator_model, aircraft_model, lon_gs_controller, lat_gs_controller, airspeed_pi_controller)

    X0 = get_init_vector(closed_loop, control_input, state)
    constant_input = np.array([20, 0.04, -0.00033310605950459315])
    sim_time = 15
    t = np.linspace(0, sim_time, sim_time*5, endpoint=True)
    u = np.array([constant_input, ]*(len(t))).transpose()
    T, yout_non_lin, _ = input_output_response(
        closed_loop, U=u, T=t, X0=X0, return_x=True, method='BDF')

main()

    

    
    
