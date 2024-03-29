from typing import Union, Sequence, Callable
import numpy as np
import math
from control.iosys import InputOutputSystem, InterconnectedSystem, summing_junction, interconnect
from fcat import State, ControlInput
from fcat.constants import (ROLL_ERROR, PITCH_ERROR, AIRSPEED_ERROR,
                            PITCH_COMMAND, ROLL_COMMAND, AIRSPEED_COMMAND)


def aicc(num_data: int, num_features: int, rmse: float) -> float:
    """
    Calculates the corrected Afaikes information criterion

    :param num_data: Number of data points
    :param rmse: Root mean square error
    :param num_features: Number of fitting parameters
    """
    return aic(num_data, rmse) + (2*num_features**2 + 2.0*num_features)/(num_data - num_features - 1)


def aic(num_features: int, rmse: float) -> float:
    """
    Calculates the Afaikes information criterion

    :param rmse: Root mean square error
    :param num_features: Number of fitting parameters
    """
    return 2.0*num_features + 2.0*np.log(rmse)


def calc_airspeed(state: State, wind: np.ndarray):
    """
    :param state: State vector of the aircraft
    :param wind: Wind vector

    Return the airspeed vector
    """
    # Calculate relative airspeed velocity vector components
    vx_r = state.vx - wind[0]
    vy_r = state.vy - wind[1]
    vz_r = state.vz - wind[2]

    airspeed_vec = np.array([vx_r, vy_r, vz_r])
    return airspeed_vec


def calc_rotational_airspeed(state: State, wind: np.ndarray):
    """
    :param state: State vector of the aircraft
    :param wind: Wind vector

    Return the airspeed rotational vector
    """

    # Calculate relative rotational airspeed velocity vector components
    ang_rate_x_r = state.ang_rate_x - wind[3]
    ang_rate_y_r = state.ang_rate_y - wind[4]
    ang_rate_z_r = state.ang_rate_z - wind[5]

    airspeed_rot_vec = np.array([ang_rate_x_r, ang_rate_y_r, ang_rate_z_r])
    return airspeed_rot_vec


def calc_angle_of_attack(state: State, wind: np.ndarray):
    """
    :param state: State vector of the aircraft
    :param wind: Wind vector

    Return the angle of attack in radians
    """
    airspeed_vec = calc_airspeed(state, wind)
    u_r = airspeed_vec[0]
    w_r = airspeed_vec[2]
    if u_r > 0.001:
        alpha = math.atan2(w_r, u_r)
    else:
        alpha = 0
    return alpha


def calc_angle_of_sideslip(state: State, wind: np.ndarray):
    """
    :param state: State vector of the aircraft
    :param wind: Wind vector

    Return the angle of sideslip in radians
    """
    airspeed_vec = calc_airspeed(state, wind)
    airspeed_magnitude = math.sqrt(airspeed_vec[0]**2 + airspeed_vec[1]**2 + airspeed_vec[2]**2)
    v_r = airspeed_vec[1]
    if airspeed_magnitude > 0.001:
        beta = math.asin(v_r/airspeed_magnitude)
    else:
        beta = 0
    return beta


def deg2rad(array: Union[np.ndarray, float]) -> Union[np.ndarray, float]:
    return array*np.pi/180.0


def wind2body_rot_matrix(state: State, wind: np.ndarray) -> np.ndarray:
    """
    Returns the rotation matrix that rotates a vector from the wind frame to the body frame

    :param state: State vector of the aircraft
    :param wind: Wind-vector
    """
    alpha = calc_angle_of_attack(state, wind)
    beta = calc_angle_of_sideslip(state, wind)
    return np.array([[np.cos(beta)*np.cos(alpha), np.sin(beta)*np.cos(alpha), -np.sin(alpha)],
                     [-np.sin(beta), np.cos(beta), 0],
                     [np.cos(beta)*np.sin(alpha), np.sin(alpha)*np.sin(beta), np.cos(alpha)]])


def body2wind_rot_matrix(state: State, wind: np.ndarray) -> np.ndarray:
    """
    Returns the rotation matrix that rotates a vector from the wind frame to the body frame

    :param state: State vector of the aircraft
    :param wind: Wind-vector
    """
    return wind2body_rot_matrix(state, wind).T.copy()


def body2wind(vec: np.ndarray, state: State, wind: np.ndarray) -> np.ndarray:
    """
    Rotate the vector vec from body frame to wind frame

    :param vec: Vector of length 3 to be rotated
    :param state: Current state vector of the air plane
    :param wind: Wind vector
    """
    return body2wind_rot_matrix(state, wind).dot(vec)


def wind2body(vec: np.ndarray, state: State, wind: np.ndarray) -> np.ndarray:
    """
    Rotate the vector vec from wind frame to body frame

    :param vec: Vector of length 3 to be rotated
    :param state: Current state vector of the air plane
    :param wind: Wind vector
    """
    return wind2body_rot_matrix(state, wind).dot(vec)


def inertial2body_rot_matrix(state: State) -> np.ndarray:
    """
    Rotate the vector vec from inertial frame to body frame

    :param vec: Vector of length 3 to be rotated
    :param state: Current state vector of the air plane
    :param wind: Wind vector
    """
    phi = state.roll
    theta = state.pitch
    psi = state.yaw

    s_phi = np.sin(phi)
    c_phi = np.cos(phi)
    s_theta = np.sin(theta)
    c_theta = np.cos(theta)
    s_psi = np.sin(psi)
    c_psi = np.cos(psi)

    return np.array([[c_theta*c_psi, c_theta*s_psi, -s_theta],
                     [s_phi*s_theta*c_psi-c_phi*s_psi, s_phi*s_theta*s_psi+c_phi*c_psi,
                      s_phi*c_theta],
                     [c_phi*s_theta*c_psi+s_phi*s_psi, c_phi*s_theta*s_psi-s_phi*c_psi,
                      c_phi*c_theta]])


def inertial2body(vec: np.ndarray, state: State) -> np.ndarray:
    """
    Rotate the vector vec from inertial frame to body frame

    :param vec: Vector of length 3 to be rotated
    :param state: Current state vector of the air plane
    :param wind: Wind vector
    """
    return inertial2body_rot_matrix(state).dot(vec)


def body2inertial(vec: np.ndarray, state: State) -> np.ndarray:
    """
    Rotate the vector vec from inertial frame to body frame

    :param vec: Vector of length 3 to be rotated
    :param state: Current state vector of the air plane
    :param wind: Wind vector
    """
    return inertial2body_rot_matrix(state).T.dot(vec)


def body2euler_angles_transform_matrix(state: State) -> np.ndarray:
    """
    Rotate the vector with angles in body frame to euler angles

    :param vec: Vector of length 3 to be rotated
    :param state: Current state vector of the air plane
    :param wind: Wind vector
    """
    phi = state.roll
    theta = state.pitch

    return np.array([[1, np.sin(phi)*np.tan(theta), np.cos(phi)*np.tan(theta)],
                     [0, np.cos(phi), -np.sin(phi)],
                     [0, np.sin(phi)/np.cos(theta), np.cos(phi)/np.cos(theta)]])


def body2euler(vec: np.ndarray, state: State) -> np.ndarray:
    """
    Rotate the vector vec with angles in body frame to euler angles

    :param vec: Vector of length 3 to be rotated
    :param state: Current state vector of the air plane
    :param wind: Wind vector
    """
    return body2euler_angles_transform_matrix(state).dot(vec)


def saturate(val: float, min: float, max: float) -> float:
    """
    :param val: value subject to saturation
    :param min: minimum value
    :param max: maximum value

    Returns the saturated value val
    """
    if val <= min:
        return min
    elif val >= max:
        return max
    else:
        return val


def flying_wing2ctrl_input_matrix():
    """
    Returns the matrix that transform from control vector with elevon deflections(flying-wing) to
    control vector with aileron and elevator deflection is given by.

    u = Tu_fw

    Where u is the control input vector with aileron and elevator deflections,
    u_fw is the control vector with elevon deflections and T is the matrix that is returned

    """
    return np.array([[0.5, 0.5, 0, 0], [-0.5, 0.5, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])


def collect_outputs(syslist: Sequence) -> list:
    """
    """
    outputs = []
    for system in syslist:
        outputs += list(system.output_index.keys())
    return outputs


def is_unique(data: Sequence) -> bool:
    """
    """
    return len(set(data)) == len(data)


def collect_inputs(syslist: Sequence) -> list:
    """
    """
    inputs = []
    for system in syslist:
        inputs += list(system.input_index.keys())
    return inputs


def create_connection_matrix(inputs: Sequence, outputs: Sequence) -> np.ndarray:

    if not is_unique(outputs):
        # TODO: Multiple
        return None
    connection_matrix = np.zeros((len(inputs), len(outputs)))
    for inp_i in range(len(inputs)):
        for outp_i in range(len(outputs)):
            if inputs[inp_i] == outputs[outp_i]:
                connection_matrix[inp_i, outp_i] = 1
    return connection_matrix


def create_input_map(external_inputs: Sequence, internal_inputs: Sequence):
    input_map = np.zeros((len(internal_inputs), len(external_inputs)))
    for ext_index in range(len(external_inputs)):
        for int_index in range(len(internal_inputs)):
            if external_inputs[ext_index] == internal_inputs[int_index]:
                input_map[int_index, ext_index] = 1
    return input_map


def add_actuator(actuator_model: InputOutputSystem,
                 aircraft_model: InputOutputSystem) -> InterconnectedSystem:
    inputs = ('elevator_deflection_command', 'aileron_deflection_command',
              'rudder_deflection_command', 'throttle_command')
    states_aircraft = ('x', 'y', 'z', 'roll', 'pitch', 'yaw', 'vx',
                       'vy', 'vz', 'ang_rate_x', 'ang_rate_y', 'ang_rate_z')
    outputs = ('x', 'y', 'z', 'roll', 'pitch', 'yaw', 'vx',
               'vy', 'vz', 'ang_rate_x', 'ang_rate_y', 'ang_rate_z')
    states_actuator = ('elevator_deflection', 'aileron_deflection',
                       'rudder_deflection', 'throttle')
    states = states_actuator + states_aircraft
    system_with_actuator = aircraft_model*actuator_model
    system_with_actuator.set_states(states)
    system_with_actuator.set_inputs(inputs)
    system_with_actuator.set_outputs(outputs)
    system_with_actuator.name = 'system_with_actuator'
    return system_with_actuator


def build_closed_loop(actuator_model: InputOutputSystem, aircraft_model: InputOutputSystem,
                      longitudinal_controller: InputOutputSystem,
                      lateral_controller: InputOutputSystem,
                      airspeed_controller: InputOutputSystem) -> InterconnectedSystem:
    """
    Convenience function to connect blocks according to the block diagram below
                  -------------       ----------       ----------
                 | Long. ctrl  |     |          |     |          |
    ===> (+) ==> | Lat. ctrl   | ==> | Actuator | ==> | Aircraft | =========>
         ^^      | Airsp. ctrl |     |          |     |          |   |
         | -1     -------------       ----------       ----------    |
         |                                                           |
         |                                                           |
         |===========================================================|

    """
    feedback_summing_junction_airspeed = summing_junction(
        inputs=[AIRSPEED_COMMAND, '-airspeed'], outputs=AIRSPEED_ERROR,
        name='feedback_summing_junction_airspeed')
    feedback_summing_junction_pitch = summing_junction(
        inputs=[PITCH_COMMAND, '-pitch'], outputs=PITCH_ERROR,
        name='feedback_summing_junction_pitch')
    feedback_summing_junction_roll = summing_junction(
        inputs=[ROLL_COMMAND, '-roll'], outputs=ROLL_ERROR,
        name='feedback_summing_junction_roll')
    syslist = (actuator_model, aircraft_model, longitudinal_controller, lateral_controller,
               airspeed_controller, feedback_summing_junction_airspeed,
               feedback_summing_junction_roll, feedback_summing_junction_pitch)
    inputs = (AIRSPEED_COMMAND, PITCH_COMMAND, ROLL_COMMAND)
    outputs = State.names

    interconnected_sys_cl = interconnect(syslist, inputs=inputs, outputs=outputs)
    connections = create_connection_matrix(collect_inputs(syslist), collect_outputs(syslist))
    input_map = create_input_map(inputs, collect_inputs(syslist))
    interconnected_sys_cl.set_connect_map(connections)
    interconnected_sys_cl.set_input_map(input_map)
    interconnected_sys_cl.name = 'feedback_loop'

    return interconnected_sys_cl


def is_state_variable(output: str) -> bool:
    return output in State.names


def create_aircraft_output_fonction(outputs: Sequence[str]) \
        -> Callable[[float, np.ndarray, np.ndarray, dict], np.ndarray]:
    def output_func(t: float, x: np.ndarray, u: np.ndarray, params: dict) -> np.ndarray:
        prop = params['prop']
        wind = params['wind'].get(t)

        out = np.zeros(len(outputs))
        for i, output in enumerate(outputs):
            if is_state_variable(output):
                out[i] = x[State.names.index(output)]
            # elif is_control_signal(output):
                # out[i] = x[ctrl_vecolInput.names.index[output]]
                # None
                # TODO: Add elevons_right nad elevons_left option
            elif output == "airspeed":
                state = State(x)
                out[i] = np.sqrt(np.sum(calc_airspeed(state, wind)**2))
            elif prop.is_known_output(output):
                out[i] = prop.get_output(output)
        return out
    return output_func


def get_init_vector(closed_loop: InputOutputSystem,
                    control_input: ControlInput, state: State) -> np.ndarray:
    init_vec = np.zeros(closed_loop.nstates)
    init_vec[0:4] = control_input.control_input
    init_vec[4:16] = state.state
    return init_vec
