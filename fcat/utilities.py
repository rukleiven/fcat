from typing import Union
import numpy as np
import math
from fcat import State


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
    roll_dot_r = state.roll_dot - wind[3]
    pitch_dot_r = state.pitch_dot - wind[4]
    yaw_dot_r = state.yaw_dot - wind[5]

    airspeed_rot_vec = np.array([roll_dot_r, pitch_dot_r, yaw_dot_r])
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


def euler_rot_matrix(state: State) -> np.ndarray:
    """
    Rotate the vector vec from body frame to euler angles

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
    Rotate the vector vec from inertial frame to body frame

    :param vec: Vector of length 3 to be rotated
    :param state: Current state vector of the air plane
    :param wind: Wind vector
    """
    return euler_rot_matrix(state).dot(vec)
