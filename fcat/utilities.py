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
