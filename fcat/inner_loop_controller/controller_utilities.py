from typing import Sequence
import json
from fcat.inner_loop_controller import hinfsyn
import numpy as np
from control import StateSpace, tf, augw, TransferFunction, tf2ss, ssdata, ss2tf
from scipy.optimize import minimize_scalar

__all__ = ('get_state_space_from_file', 'longitudinal_controller', 'lateral_controller')


def get_state_space_from_file(filename: str) -> Sequence:
    with open(filename, 'r') as f:
        data = json.load(f)
        A = data['A']
        B = data['B']
        C = data['C']
        D = data['D']
    return np.matrix(A), np.matrix(B), np.matrix(C), np.matrix(D)


def get_lateral_state_space(state_space_filename: str) -> Sequence:
    A, B, _, _ = get_state_space_from_file(state_space_filename)
    lateral_index = np.array([1, 7, 11, 13, 15])  # [delta_a, roll, vy, ang_rate_x, ang_rate_z]

    A_lat = A[lateral_index[:, None], lateral_index]
    B_lat = B[lateral_index[:, None], 1]

    # Output: roll
    C_lat = np.zeros((1, 5))
    C_lat[0, 1] = 1
    D_lat = 0
    return A_lat, B_lat, C_lat, D_lat


def get_longitudinal_state_space(state_space_filename: str) -> Sequence:
    A, B, _, _ = get_state_space_from_file(state_space_filename)
    A = np.matrix(A)
    B = np.matrix(B)
    # [delta_e, delta_t, pitch, vx, vz, ang_rate_y]
    longitudinal_index = np.array([0, 3,  8, 10, 12, 14])
    A_lon = A[longitudinal_index[:, None], longitudinal_index]
    B_lon = B[longitudinal_index[:, None], 0]

    # Output: Pitch
    C_lon = np.zeros((1, 6))
    C_lon[0, 2] = 1
    D_lon = 0
    return A_lon, B_lon, C_lon, D_lon


def lateral_controller(state_space_filename: str, controller_filename_out: str) -> Sequence:
    A_lat, B_lat, C_lat, D_lat = get_lateral_state_space(state_space_filename)
    lateral_statespace = StateSpace(A_lat, B_lat, C_lat, D_lat)
    # Filter design Variables:
    M = 2
    w_0 = 20.0
    A_fact = 0.001
    gam = 0
    while gam < 10 and w_0 < 20.2:
        print(w_0)
        print(gam)
        w_0 = w_0 + 0.1
        W_C = tf([2], [1])
        W_S = tf([1/M, w_0], [1, w_0*A_fact])
        W_T = tf([1, w_0/M], [A_fact, w_0])
        Plant = augw(lateral_statespace, W_S, W_C, W_T)
        K, CL, gam, rcond = hinfsyn(Plant, 1, 1, 0.01)

    # Write controller to file:
    lat_controller = {
        'A': (K.A).tolist(),
        'B': (K.B).tolist(),
        'C': (K.C).tolist(),
        'D': (K.D).tolist()
    }
    with open(controller_filename_out, 'w') as outfile:
        json.dump(lat_controller, outfile, indent=2, sort_keys=True)
    print(f"Lateral controller written to {controller_filename_out}")


def longitudinal_controller(state_space_filename: str, controller_filename_out: str) -> Sequence:
    A_lon, B_lon, C_lon, D_lon = get_longitudinal_state_space(state_space_filename)
    longitudinal_statespace = StateSpace(A_lon, B_lon, C_lon, D_lon)
    # Filter design:
    M = 2
    w_0 = 9.0
    A_fact = 0.001

    gam = 0
    while gam < 1/0.30935949 and w_0 < 13.2:
        print(gam)
        w_0 = w_0 + 0.1
        W_C = tf([0.05], [1])
        W_S = tf([1/M, w_0], [1, w_0*A_fact])
        W_T = tf([1, w_0/M], [A_fact, w_0])

        Plant = augw(longitudinal_statespace, W_S, W_C, W_T)
        K, CL, gam, rcond = hinfsyn(Plant, 1, 1, 0.01)
    # Write controller to file:
    lon_controller = {
        'A': (K.A).tolist(),
        'B': (K.B).tolist(),
        'C': (K.C).tolist(),
        'D': (K.D).tolist()
    }
    with open(controller_filename_out, 'w') as outfile:
        json.dump(lon_controller, outfile, indent=2, sort_keys=True)
    print(f"Longitudinal controller written to {controller_filename_out}")


def nu_gap(P_1: TransferFunction, P_2: TransferFunction, tol=1e-3) -> float:
    """
    Calculate nu_gap for SISO plants
    """
    P_1_ss = tf2ss(P_1)
    P_2_ss = tf2ss(P_2)
    A_1, B_1, C_1, D_1 = ssdata(P_1_ss)
    A_2, B_2, C_2, D_2 = ssdata(P_2_ss)

    if not B_1.shape[1] == B_2.shape[1] == 1 or not C_1.shape[0] == C_2.shape[0] == 1:
        # Check that iosizes are equal to 1
        return None
    # Define optimization function

    def f(x):
        tf_1 = P_1-P_2
        tf_2 = P_1**2
        tf_3 = P_2**2
        # Object to maximize:
        max_obj = abs(tf_1.evalfr(x))/np.sqrt((1+abs(tf_2.evalfr(x))) * (1+abs(tf_3.evalfr(x))))
        min_obj = -max_obj
        return min_obj

    nu_gap = minimize_scalar(f, method='brent')
    return (-f(nu_gap.x))


def find_icing_level_minimize_mu_gap():
    # Longitudinal icing level:
    A_1, B_1, C_1, D_1 = get_longitudinal_state_space('examples/skywalkerx8_linmod.json')
    P_1 = ss2tf(A_1, B_1, C_1, D_1)
    A_2, B_2, C_2, D_2 = get_longitudinal_state_space('examples/skywalkerx8_linmod_icing10.json')
    P_2 = ss2tf(A_2, B_2, C_2, D_2)
    nu_gap_min = nu_gap(P_1, P_2)
    longitudinal_icing_level = 1.0
    for i in range(1, 10):
        A_3, B_3, C_3, D_3 = get_longitudinal_state_space(
            'examples/skywalkerx8_linmod_icing0'+str(i)+'.json')
        P_3 = ss2tf(A_3, B_3, C_3, D_3)
        nu_gap_clean = nu_gap(P_1, P_3)
        nu_gap_iced = nu_gap(P_2, P_3)
        if max([nu_gap_clean, nu_gap_iced]) < nu_gap_min:
            nu_gap_min = max([nu_gap_clean, nu_gap_iced])
            longitudinal_icing_level = i*0.1
    print(nu_gap_min)
    # Lateral icing level
    A_1, B_1, C_1, D_1 = get_lateral_state_space('examples/skywalkerx8_linmod.json')
    P_1 = ss2tf(A_1, B_1, C_1, D_1)
    A_2, B_2, C_2, D_2 = get_lateral_state_space('examples/skywalkerx8_linmod_icing10.json')
    P_2 = ss2tf(A_2, B_2, C_2, D_2)
    nu_gap_min = nu_gap(P_1, P_2)
    lateral_icing_level = 1.0
    for i in range(1, 10):
        A_3, B_3, C_3, D_3 = get_lateral_state_space(
            'examples/skywalkerx8_linmod_icing0'+str(i)+'.json')
        P_3 = ss2tf(A_3, B_3, C_3, D_3)
        nu_gap_clean = nu_gap(P_1, P_3)
        nu_gap_iced = nu_gap(P_2, P_3)
        if max([nu_gap_clean, nu_gap_iced]) < nu_gap_min:
            nu_gap_min = max([nu_gap_clean, nu_gap_iced])
            lateral_icing_level = i*0.1

    return longitudinal_icing_level, lateral_icing_level
