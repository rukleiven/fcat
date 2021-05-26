from typing import Sequence
import json

from control.iosys import LinearIOSystem, input_output_response
from fcat.inner_loop_controller import hinfsyn
import numpy as np
from control import (StateSpace, tf, augw, TransferFunction, ss2tf, bode_plot,
                     feedback, evalfr, nyquist_plot, stability_margins, tf2ss)
from scipy.optimize import minimize_scalar
from matplotlib import pyplot as plt
# from collections.abc import Iterable
from copy import deepcopy
import matplotlib as mpl
mpl.rcParams.update({"font.family": "serif", "font.size": 11})
plt.style.use("seaborn-deep")


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
                  phase_m: float = 0.45, gain_m: float = 2, stab_m=0.50):
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
        :param plant: Transferfunction of linearized SISO plant
        :param controller: Transferfunction of feedback controller

        returns: gap-metric stability margin
    """

    def get_matrix_at_freq(x: float):
        return np.array([[1/(1 + evalfr(plant*controller, 1j*x)),
                          evalfr(plant, 1j*x)/(1 + evalfr(plant*controller, 1j*x))],
                         [evalfr(controller, 1j*x)/(1 + evalfr(plant*controller, 1j*x)),
                          evalfr(plant*controller, 1j*x)/(1 + evalfr(plant*controller, 1j*x))]])

    def maximum_singular_value(x):
        u, s, vh = np.linalg.svd(get_matrix_at_freq(x))
        if x < 0.0001:
            return 1
        return 1/max(s)

    margin = minimize_scalar(maximum_singular_value, method='brent')
    return maximum_singular_value(margin.x)


def conjugate_plant(plant: TransferFunction) -> TransferFunction:
    """
        :param plant: Transferfunction of linearized SISO plant

        returns: the conjugated plant, i.e. P(-s)
    """

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
    """
        :param plant_1: Transferfunction of linearized SISO plant
        :param plant_2: Transferfunction of linearized SISO plant

        returns: True if plant_1 and plant_2 satisfy the winding number condition. (False if not)
    """

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


def performance_requirements(controller: TransferFunction, plant: TransferFunction,
                             overshoot_threshold: float, step_size: float = 0.5,
                             sim_time: float = 10) -> bool:
    OL = controller*plant
    CL = feedback(OL, 1)
    linsys = LinearIOSystem(tf2ss(CL))
    t = np.linspace(0, sim_time, sim_time*5, endpoint=True)
    u = np.array([step_size, ]*(len(t))).transpose()
    t, yout = input_output_response(linsys, U=u, T=t, method="BDF")
    overshoot = (max(yout)-step_size)/step_size
    return overshoot < overshoot_threshold

# -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:


def lateral_controller(ss_nom_filename: str, ss_wc_lower_fname: str, ss_wc_upper_fname: str,
                       controller_filename_out: str = None) -> tuple:
    """
        :param controller_filename_out: Write to filename
        :param ss_nom_filename: nominal plant filename
        :param ss_wc_lower_fname: worst case lower filename
        :param ss_wc_upper_fname: worst case upper filename

        returns: A, B, C, D state space matrices of the synthesized controller
    """

    A_lat, B_lat, C_lat, D_lat = get_lateral_state_space(ss_nom_filename)
    lateral_statespace = StateSpace(A_lat, B_lat, C_lat, D_lat)

    A_wcl, B_wcl, C_wcl, D_wcl = get_lateral_state_space(ss_wc_lower_fname)
    A_wcu, B_wcu, C_wcu, D_wcu = get_lateral_state_space(ss_wc_upper_fname)

    worst_case_lower_tf = ss2tf(A_wcl, B_wcl, C_wcl, D_wcl)
    worst_case_upper_tf = ss2tf(A_wcu, B_wcu, C_wcu, D_wcu)
    lateral_tf = ss2tf(A_lat, B_lat, C_lat, D_lat)

    # Filter design Variables:
    M = 2.0
    w_0 = 1  # min_freq
    A_fact = 0.0001
    margins = True
    nu_gap_l = nu_gap(lateral_tf, worst_case_lower_tf)
    nu_gap_u = nu_gap(lateral_tf, worst_case_upper_tf)
    omega_step = 0.1
    while w_0 < 10:
        w_0 = w_0 + omega_step
        W_C = tf([1.2], [1])
        # W_S = tf([1/M, w_0], [1, w_0*A_fact])
        W_S = tf([1/M, 2*w_0/np.sqrt(M), w_0**2], [1, 2*w_0*np.sqrt(A_fact), (w_0**2)*A_fact])
        W_T = tf([1, w_0/M], [A_fact, w_0])
        Plant = augw(lateral_statespace, W_S, W_C, W_T)
        K_tmp, CL_tmp, gam_tmp, rcond_tmp = hinfsyn(Plant, 1, 1, 0.01)
        controller = ss2tf(K_tmp.A, K_tmp.B, K_tmp.C, K_tmp.D)

        # Stability-requirements
        robust_margin = ncrs_margin(lateral_tf, controller)
        stability = nu_gap_l < robust_margin and nu_gap_u < robust_margin

        # Robustness-requirements
        margins = check_margins(controller, lateral_statespace,
                                worst_case_upper_tf, worst_case_lower_tf, stab_m=0.6)

        # Performance-requirements
        perf_req = performance_requirements(controller, lateral_tf, 0.27)

        if stability and margins and perf_req:
            K, _, gam, _ = K_tmp, CL_tmp, gam_tmp, rcond_tmp
        else:
            break
    print(f"Lateral performance value (gamma) {gam}")
    print(f"Latereal controller crossover frequency (w_0) {w_0-omega_step}")

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


def longitudinal_controller(ss_nom_filename: str, ss_wc_lower_fname: str, ss_wc_upper_fname: str,
                            controller_filename_out: str = None) -> tuple:
    """
        :param controller_filename_out: Write to filename
        :param ss_nom_filename: nominal plant filename
        :param ss_wc_lower_fname: worst case lower filename
        :param ss_wc_upper_fname: worst case upper filename

        returns: A, B, C, D state space matrices of the synthesized controller
    """

    A_lon, B_lon, C_lon, D_lon = get_longitudinal_state_space(ss_nom_filename)
    longitudinal_ss = StateSpace(A_lon, B_lon, C_lon, D_lon)

    A_wcl, B_wcl, C_wcl, D_wcl = get_longitudinal_state_space(ss_wc_lower_fname)
    A_wcu, B_wcu, C_wcu, D_wcu = get_longitudinal_state_space(ss_wc_upper_fname)

    worst_case_lower_tf = ss2tf(A_wcl, B_wcl, C_wcl, D_wcl)
    worst_case_upper_tf = ss2tf(A_wcu, B_wcu, C_wcu, D_wcu)
    long_tf = ss2tf(longitudinal_ss)
    # Initial filter design:
    M = 2
    w_0 = 5
    A_fact = 0.001

    nu_gap_l = nu_gap(long_tf, worst_case_lower_tf)
    nu_gap_u = nu_gap(long_tf, worst_case_upper_tf)
    omega_step = 0.2
    while w_0 < 15:
        w_0 = w_0 + omega_step
        W_C = tf([0.1], [1])
        W_S = tf([1/M, w_0], [1, w_0*A_fact])
        W_T = tf([1, w_0/M], [A_fact, w_0])

        Plant = augw(longitudinal_ss, W_S, W_C, W_T)
        K_tmp, CL_tmp, gam_tmp, rcond_tmp = hinfsyn(Plant, 1, 1, 0.01)
        controller = ss2tf(K_tmp.A, K_tmp.B, K_tmp.C, K_tmp.D)

        # Stability-requirements
        robust_margin = ncrs_margin(long_tf, controller)
        stability = nu_gap_l < robust_margin and nu_gap_u < robust_margin

        # Robustness-requirements
        margins = check_margins(controller, long_tf,
                                worst_case_upper_tf, worst_case_lower_tf, stab_m=0.5)

        # Performance-requirements
        perf_req = performance_requirements(controller, long_tf, 0.1)
        if stability and margins and perf_req:
            K, _, gam, _ = K_tmp, CL_tmp, gam_tmp, rcond_tmp
        else:
            break

    print(f"Longitudinal performance value (gamma) {gam}")
    print(f"Longitudinal controller crossover frequency (w_0) {w_0-omega_step}")
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
        :param P_1: Transferfunction of linearized SISO plant
        :param P_2: Transferfunction of linearized SISO plant

        returns: the nu-gap metric between P_1 and P_2
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
