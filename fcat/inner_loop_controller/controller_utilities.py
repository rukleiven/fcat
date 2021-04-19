from typing import Sequence
import json
from fcat.inner_loop_controller import hinfsyn
import numpy as np
from control import (StateSpace, tf, augw, TransferFunction, ss2tf, bode_plot, step_response,
                     feedback, evalfr, nyquist_plot, stability_margins)
from scipy.optimize import minimize_scalar
from matplotlib import pyplot as plt
# from collections.abc import Iterable
from copy import deepcopy

__all__ = ('get_state_space_from_file', 'lateral_controller', 'longitudinal_controller',
           'get_lateral_state_space', 'get_longitudinal_state_space', 'nu_gap', 'ncrs_margin',
           "winding_number_condition")


def plot_frequency_respons(sys: TransferFunction):
    mag, phase, omega = bode_plot(sys)


def get_state_space_from_file(filename: str) -> Sequence:
    with open(filename, 'r') as f:
        data = json.load(f)
        A = data['A']
        B = data['B']
        C = data['C']
        D = data['D']
    return np.matrix(A), np.matrix(B), np.matrix(C), np.matrix(D)


def get_lateral_state_space(*argv) -> Sequence:
    if len(argv) == 1:
        state_space_filename = argv[0]
        A, B, _, _ = get_state_space_from_file(state_space_filename)
    elif len(argv) == 4:
        A = argv[0]
        B = argv[1]
    lateral_index = np.array([1, 7, 11, 13, 15])  # [delta_a, roll, vy, ang_rate_x, ang_rate_z]
    A_lat = A[lateral_index[:, None], lateral_index]
    B_lat = B[lateral_index[:, None], 1]

    # Output: roll
    C_lat = np.zeros((1, 5))
    C_lat[0, 1] = 1
    D_lat = 0
    return A_lat, B_lat, C_lat, D_lat


def get_longitudinal_state_space(*argv) -> Sequence:
    if len(argv) == 1:
        state_space_filename = argv[0]
        A, B, _, _ = get_state_space_from_file(state_space_filename)
    elif len(argv) == 4:
        A = argv[0]
        B = argv[1]
    A = np.matrix(A)
    B = np.matrix(B)
    # [delta_e, delta_t, pitch, vx, vz, ang_rate_y]
    longitudinal_index = np.array([0, 3, 8, 10, 12, 14])
    A_lon = A[longitudinal_index[:, None], longitudinal_index]
    B_lon = B[longitudinal_index[:, None], 0]

    # Output: Pitch
    C_lon = np.zeros((1, 6))
    C_lon[0, 2] = 1
    D_lon = 0
    return A_lon, B_lon, C_lon, D_lon


def plot_respons(t: np.ndarray, states: np.ndarray):
    fig = plt.figure()

    for i in range(states.shape[0]):
        ax = fig.add_subplot(3, 4, i+1)
        ax.plot(t, states[i, :].transpose())
        ax.set_xlabel("Time")
        ax.set_ylabel(f"State {i}")
    return fig


def check_margins(controller: TransferFunction, nominal_plant: TransferFunction,
                  worst_case_upper: TransferFunction, worst_case_lower: TransferFunction,
                  phase_m: float = 0.45, gain_m: float = 2, stab_m=0.60):
    open_loops = []
    open_loops.append(worst_case_lower*controller)
    open_loops.append(worst_case_upper*controller)
    open_loops.append(nominal_plant*controller)

    for open_loop in open_loops:
        gm, pm, sm, _, _, _ = stability_margins(open_loop)
        if gm < gain_m or pm < phase_m or sm < stab_m:
            return False
    return True


def ncrs_margin(plant: TransferFunction, controller: TransferFunction) -> float:
    """
    Plant, Controllers are SISO
    """
    def get_matrix_at_freq(x: float):
        return np.array([[1/(1 + evalfr(plant*controller, 1j*x)),
                          evalfr(plant, 1j*x)/(1 + evalfr(plant*controller, 1j*x))],
                         [evalfr(controller, 1j*x)/(1 + evalfr(plant*controller, 1j*x)),
                          evalfr(plant*controller, 1j*x)/(1 + evalfr(plant*controller, 1j*x))]])

    def maximum_singular_value(x):
        u, s, vh = np.linalg.svd(get_matrix_at_freq(x))
        return 1/max(s)

    margin = minimize_scalar(maximum_singular_value, bounds=(0.0001, 2**12), method='bounded')
    return maximum_singular_value(margin.x)


def conjugate_plant(plant: TransferFunction) -> TransferFunction:
    num = deepcopy(plant.num[0][0])
    den = deepcopy(plant.den[0][0])
    num_poly_order = len(num) - 1
    den_poly_order = len(den) - 1
    for i in range(num_poly_order):
        if num_poly_order % 2 == 0:
            if i % 2 != 0:
                num[i] = -num[i]
        else:
            if i % 2 == 0:
                num[i] = -num[i]

    for i in range(den_poly_order):
        if den_poly_order % 2 == 0:
            if i % 2 != 0:
                den[i] = -den[i]
        else:
            if i % 2 == 0:
                den[i] = -den[i]

    c_plant = TransferFunction(num, den)
    return c_plant


def winding_number_condition(plant_1: TransferFunction, plant_2: TransferFunction) -> bool:
    plant_2_c = conjugate_plant(plant_2)

    def f(x):
        p = abs(evalfr(1+plant_2_c*plant_1, 1j*x))
        p = abs(10*x - 2*x**2 - 10)
        return p
    min_obj = minimize_scalar(f, method="golden")
    if f(min_obj.x) == 0:
        return False

    wno = nyquist_plot(1+plant_2_c*plant_1)
    eta_0 = 0
    eta_1 = 0
    eta_2 = 0
    plant_1_poles = plant_1.pole()
    plant_2_poles = plant_2.pole()
    for p in plant_1_poles:
        if np.real(p) > 0:
            eta_1 = eta_1 + 1
    for p in plant_2_poles:
        if np.real(p) > 0:
            eta_2 = eta_2 + 1
        if np.real(p) == 0:
            eta_0 = eta_0 + 1
    return wno + eta_1 - eta_2 - eta_0 == 0


# -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
def lateral_controller(controller_filename_out: str = None, *argv):
    if len(argv) == 1:
        state_space_filename = argv[0]
        A_lat, B_lat, C_lat, D_lat = get_lateral_state_space(state_space_filename)
    elif len(argv) == 4:
        A = argv[0]
        B = argv[1]
        C = argv[2]
        D = argv[3]
        A_lat, B_lat, C_lat, D_lat = get_lateral_state_space(A, B, C, D)
    lateral_statespace = StateSpace(A_lat, B_lat, C_lat, D_lat)

    wcl_filename = "./examples/skywalkerX8_analysis/SkywalkerX8_state_space_models\
                    /skywalkerx8_linmod.json"
    wcu_filename = "./examples/skywalkerX8_analysis/SkywalkerX8_state_space_models\
                    /skywalkerx8_linmod_icing10.json"
    A_wcl, B_wcl, C_wcl, D_wcl = get_lateral_state_space(wcl_filename)
    A_wcu, B_wcu, C_wcu, D_wcu = get_lateral_state_space(wcu_filename)

    Worst_case_lower_tf = ss2tf(A_wcl, B_wcl, C_wcl, D_wcl)
    Worst_case_upper_tf = ss2tf(A_wcu, B_wcu, C_wcu, D_wcu)
    lateral_tf = ss2tf(A_lat, B_lat, C_lat, D_lat)

    # Filter design Variables:
    M = 2.0
    w_0 = 5
    A_fact = 0.00001
    margins = True
    nu_gap_l = nu_gap(lateral_tf, Worst_case_lower_tf)
    nu_gap_u = nu_gap(lateral_tf, Worst_case_upper_tf)

    while w_0 < 100:
        w_0 = w_0 + 0.1
        W_C = tf([35], [1])
        # W_S1 = tf([1/M, w_0], [1, w_0*A_fact])
        W_S = tf([1/M, 2*w_0/np.sqrt(M), w_0**2], [1, 2*w_0*np.sqrt(A_fact), (w_0**2)*A_fact])
        W_T = tf([1, w_0/M], [A_fact, w_0])
        Plant = augw(lateral_statespace, W_S, W_C, W_T)
        K_tmp, CL_tmp, gam_tmp, rcond_tmp = hinfsyn(Plant, 1, 1, 0.001)
        controller = ss2tf(K_tmp.A, K_tmp.B, K_tmp.C, K_tmp.D)
        robust_margin = ncrs_margin(lateral_tf, controller)
        margins = check_margins(controller, lateral_statespace,
                                Worst_case_upper_tf, Worst_case_lower_tf)
        stability = nu_gap_l < robust_margin and nu_gap_u < robust_margin
        if stability and margins:
            K, CL, gam, _ = K_tmp, CL_tmp, gam_tmp, rcond_tmp
        else:
            break
    print(gam)
    controller = ss2tf(K.A, K.B, K.C, K.D)
    sim_time = 20
    OL = Worst_case_lower_tf
    CL = feedback(OL, 1)

    t = np.linspace(0, sim_time, sim_time*5, endpoint=True)
    T, yout = step_response(CL, T=t, X0=0)
    # plot_respons(T,yout)
    # plt.show()
    # stepsize = 1
    # print(yout)
    # overshoot = (max(yout)-stepsize)/stepsize
    # undershoot = (min(yout))/stepsize
    # #print(overshoot)
    # fig = plt.figure()
    # ax = fig.add_subplot(1,1,1)
    # ax.plot(t, yout)

    # plt.show()
    # plot_frequency_respons(W_S)
    # plot_frequency_respons(W_S1)
    # Write controller to file:
    if controller_filename_out is not None:
        lat_controller = {
            'A': (K.A).tolist(),
            'B': (K.B).tolist(),
            'C': (K.C).tolist(),
            'D': (K.D).tolist()
        }
        with open(controller_filename_out, 'w') as outfile:
            json.dump(lat_controller, outfile, indent=2, sort_keys=True)
        print(f"Lateral controller written to {controller_filename_out}")
    return np.array(K.A), np.array(K.B), np.array(K.C), np.array(K.D)


def longitudinal_controller(controller_filename_out: str = None, *argv):
    """
    """
    if len(argv) == 1:
        state_space_filename = argv[0]
        A_lon, B_lon, C_lon, D_lon = get_longitudinal_state_space(state_space_filename)
    elif len(argv) == 4:
        A = argv[0]
        B = argv[1]
        C = argv[2]
        D = argv[3]
        A_lon, B_lon, C_lon, D_lon = get_longitudinal_state_space(A, B, C, D)

    longitudinal_statespace = StateSpace(A_lon, B_lon, C_lon, D_lon)

    # Filter design:
    M = 2
    w_0 = 7
    A_fact = 0.001

    gam = 0
    while gam < 1/0.30935949 and w_0 < 10.2:
        w_0 = w_0 + 0.1
        W_C = tf([0.1], [1])
        W_S = tf([1/M, w_0], [1, w_0*A_fact])
        W_T = tf([1, w_0/M], [A_fact, w_0])

        Plant = augw(longitudinal_statespace, W_S, W_C, W_T)
        K, CL, gam, rcond = hinfsyn(Plant, 1, 1, 0.01)

    # Write controller to file:
    # plot_frequency_respons(W_S)
    # sim_time = 20
    # t = np.linspace(0, sim_time, sim_time*5, endpoint=True)
    # T, yout = step_response(CL, T=t, X0=0)
    # plot_respons(T,yout)
    # print(CL)
    # plt.show()
    # print(controller_filename_out)
    if controller_filename_out is not None:
        lon_controller = {
            'A': (K.A).tolist(),
            'B': (K.B).tolist(),
            'C': (K.C).tolist(),
            'D': (K.D).tolist()
        }
        with open(controller_filename_out, 'w') as outfile:
            json.dump(lon_controller, outfile, indent=2, sort_keys=True)

        print(f"Longitudinal controller written to {controller_filename_out}")

    return np.array(K.A), np.array(K.B), np.array(K.C), np.array(K.D)


def nu_gap(P_1: TransferFunction, P_2: TransferFunction, tol=1e-3) -> float:
    """
    Calculate nu_gap for SISO plants
    """
    # Define optimization function
    def f(x):
        tf_1 = P_1
        tf_2 = P_2
        # Object to maximize:
        max_obj = abs((evalfr(tf_1, 1j*x) - evalfr(tf_2, 1j*x))/np.sqrt((1+evalfr(tf_1, 1j*x)
                      * evalfr(tf_1, -1j*x)) * (1+evalfr(tf_2, 1j*x)*evalfr(tf_2, -1j*x))))
        min_obj = -max_obj
        return min_obj

    nu_gap = minimize_scalar(f, bounds=(0.0001, 10**12), method='bounded')
    return (-f(nu_gap.x))


def find_icing_level_minimize_mu_gap():
    # Longitudinal icing level:
    A_1, B_1, C_1, D_1 = get_longitudinal_state_space(
        './examples/skywalkerX8_analysis/SkywalkerX8_state_space_models/skywalkerx8_linmod.json')
    P_1 = ss2tf(A_1, B_1, C_1, D_1)
    A_2, B_2, C_2, D_2 = get_longitudinal_state_space('./examples/skywalkerX8_analysis\
        /SkywalkerX8_state_space_models/skywalkerx8_linmod_icing10.json')
    P_2 = ss2tf(A_2, B_2, C_2, D_2)
    nu_gap_min = nu_gap(P_1, P_2)
    longitudinal_icing_level = 1.0
    for i in range(1, 10):
        A_3, B_3, C_3, D_3 = get_longitudinal_state_space('./examples/skywalkerX8_analysis\
            /SkywalkerX8_state_space_models/skywalkerx8_linmod_icing0'+str(i)+'.json')
        P_3 = ss2tf(A_3, B_3, C_3, D_3)
        nu_gap_clean = nu_gap(P_1, P_3)
        nu_gap_iced = nu_gap(P_2, P_3)
        if max([nu_gap_clean, nu_gap_iced]) < nu_gap_min:
            nu_gap_min = max([nu_gap_clean, nu_gap_iced])
            longitudinal_icing_level = i*0.1
    # Lateral icing level
    A_1, B_1, C_1, D_1 = get_lateral_state_space(
        './examples/skywalkerX8_analysis/SkywalkerX8_state_space_models/skywalkerx8_linmod.json')
    P_1 = ss2tf(A_1, B_1, C_1, D_1)
    A_2, B_2, C_2, D_2 = get_lateral_state_space('./examples/skywalkerX8_analysis\
        /SkywalkerX8_state_space_models/skywalkerx8_linmod_icing10.json')
    P_2 = ss2tf(A_2, B_2, C_2, D_2)
    nu_gap_min = nu_gap(P_1, P_2)
    lateral_icing_level = 1.0
    for i in range(1, 10):
        A_3, B_3, C_3, D_3 = get_lateral_state_space('./examples/skywalkerX8_analysis\
            /SkywalkerX8_state_space_models/skywalkerx8_linmod_icing0'+str(i)+'.json')
        P_3 = ss2tf(A_3, B_3, C_3, D_3)
        nu_gap_clean = nu_gap(P_1, P_3)
        nu_gap_iced = nu_gap(P_2, P_3)
        if max([nu_gap_clean, nu_gap_iced]) < nu_gap_min:
            nu_gap_min = max([nu_gap_clean, nu_gap_iced])
            lateral_icing_level = i*0.1

    return longitudinal_icing_level, lateral_icing_level

# find_icing_level_minimize_mu_gap()
