from typing import Sequence, Tuple
import json
from fcat.constants import Direction
from control.iosys import NonlinearIOSystem
from fcat.inner_loop_controller import (hinfsyn, roll_gain_scheduled_controller,
                                        pitch_hinf_controller,
                                        roll_hinf_controller, pitch_gain_scheduled_controller,
                                        StateSpaceMatrices, SaturatedStateSpaceController,
                                        SaturatedStateSpaceMatricesGS, airspeed_pi_controller)

import numpy as np
from control import (StateSpace, tf, augw, TransferFunction, ss2tf,
                     evalfr, nyquist_plot, stability_margins)
from scipy.optimize import minimize_scalar
from matplotlib import pyplot as plt
# from collections.abc import Iterable
from copy import deepcopy
import matplotlib as mpl
from typing import Union
mpl.rcParams.update({"font.family": "serif", "font.size": 11})
plt.style.use("seaborn-deep")


__all__ = ('get_state_space_from_file', 'lateral_controller', 'longitudinal_controller',
           'get_lateral_state_space', 'get_longitudinal_state_space', 'nu_gap', 'ncrs_margin',
           "winding_number_condition", 'init_airspeed_controller',
           'init_gs_controller', 'init_robust_controller')


def get_state_space_from_file(filename: str) -> Sequence:
    try:
        with open(filename, 'r') as f:
            data = json.load(f)
            A = data['A']
            B = data['B']
            C = data['C']
            D = data['D']
        return StateSpaceMatrices(A, B, C, D)
    except FileNotFoundError:
        raise(FileNotFoundError)


def get_lateral_state_space(ss_model: Union[str, StateSpaceMatrices]) -> Sequence:
    """
    Get lateral state-space from full state-space modell.
    :param ss_model: filename or StateSpaceMatrices representation of state space model

    """
    if isinstance(ss_model, str):
        A, B, _, _ = get_state_space_from_file(ss_model)
    elif isinstance(ss_model, StateSpaceMatrices):
        A = np.array(ss_model.A)
        B = np.array(ss_model.B)
    lateral_index = np.array([1, 7, 11, 13, 15])  # [delta_a, roll, vy, ang_rate_x, ang_rate_z]
    A_lat = A[lateral_index[:, None], lateral_index]
    B_lat = B[lateral_index[:, None], 1]

    # Output: roll
    C_lat = np.zeros((1, 5))
    C_lat[0, 1] = 1
    D_lat = 0
    return A_lat, B_lat, C_lat, D_lat


def get_longitudinal_state_space(ss_model: Union[str, StateSpaceMatrices]) -> Sequence:
    """
    Get longitudinal state-space from full state-space modell.
    :param ss_model: filename or StateSpaceMatrices representation of state space model

    """
    if isinstance(ss_model, str):
        A, B, _, _ = get_state_space_from_file(ss_model)
    elif isinstance(ss_model, StateSpaceMatrices):
        A = ss_model.A
        B = ss_model.B
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


def check_margins(controller: TransferFunction, nominal_plant: TransferFunction,
                  worst_case_upper: TransferFunction, worst_case_lower: TransferFunction,
                  phase_m: float = 0.45, gain_m: float = 2, stab_m=0.50) -> bool:
    """
    Check gain, phase and stability margins for worst-case and nominal plants.
    :param controller: Transfer function object of linear controller
    :param nominal plant: Transfer function object of nominal plant
    :param worst_case_upper: Transfer function object of Worst-case upper plant
    :param worst_case_lower: Transfer function object of Worst-case lower plant
    :param phase_m: Minimum acceptable phase margin
    :param gain_m:  Minimum acceptable gain margin
    :param stab_m:  Minimum acceptable stability margin

    returns: True if all margins are ok. False if not.
    """
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


def lateral_controller(nominal_ss: StateSpaceMatrices,
                       boundary_ss=Tuple[StateSpaceMatrices]) -> StateSpaceMatrices:
    """
        :param controller_filename_out: Write-to filename
        :param ss_nom_filename: nominal plant filename
        :param ss_wc_lower_fname: worst case lower filename
        :param ss_wc_upper_fname: worst case upper filename

        returns: A, B, C, D state space matrices of the synthesized controller
    """

    A_lat, B_lat, C_lat, D_lat = get_lateral_state_space(nominal_ss)
    lateral_statespace = StateSpace(A_lat, B_lat, C_lat, D_lat)

    A_wcl, B_wcl, C_wcl, D_wcl = get_lateral_state_space(boundary_ss[0])
    A_wcu, B_wcu, C_wcu, D_wcu = get_lateral_state_space(boundary_ss[1])

    worst_case_lower_tf = ss2tf(A_wcl, B_wcl, C_wcl, D_wcl)
    worst_case_upper_tf = ss2tf(A_wcu, B_wcu, C_wcu, D_wcu)
    lateral_tf = ss2tf(A_lat, B_lat, C_lat, D_lat)

    # Filter design Variables:
    M = 2.0
    w_0 = 1  # min_freq
    A_fact = 0.0002
    nu_gap_l = nu_gap(lateral_tf, worst_case_lower_tf)
    nu_gap_u = nu_gap(lateral_tf, worst_case_upper_tf)
    omega_step = 0.1

    freq_max = 10
    while w_0 < freq_max:
        w_0 = w_0 + omega_step
        W_S = tf([1/M, 2*w_0/np.sqrt(M), w_0**2], [1, 2*w_0*np.sqrt(A_fact), (w_0**2)*A_fact])
        W_C = tf([1], [1])
        W_T = tf([1, w_0/M], [A_fact, w_0])
        Plant = augw(lateral_statespace, W_S, W_C, W_T)
        K_tmp, CL_tmp, gam_tmp, rcond_tmp = hinfsyn(Plant, 1, 1, 0.01)

        controller = ss2tf(K_tmp.A, K_tmp.B, K_tmp.C, K_tmp.D)

        # Stability-requirements
        robust_margin = 1/gam_tmp
        stability = nu_gap_l < robust_margin and nu_gap_u < robust_margin

        # Robustness-requirements
        margins = check_margins(controller, lateral_statespace,
                                worst_case_upper_tf, worst_case_lower_tf, stab_m=0.7)

        if stability and margins:
            K, _, gam, _ = K_tmp, CL_tmp, gam_tmp, rcond_tmp
        else:
            break

    print(f"Lateral performance value (gamma) {gam}")
    print(f"Latereal controller crossover frequency (w_0) {w_0-omega_step}")

    return StateSpaceMatrices(K.A, K.B, K.C, K.D)


def longitudinal_controller(nominal_ss: StateSpaceMatrices,
                            boundary_ss=Tuple[StateSpaceMatrices]) -> StateSpaceMatrices:
    """
        :param nominal_ss: nomnial plant used in hinf-synthesis
        :param boundary_ss: tuple of worst case plant used for gap-metric robustness requirements

        returns: StateSpaceMatrices pf controller
    """

    A_lon, B_lon, C_lon, D_lon = get_longitudinal_state_space(nominal_ss)
    longitudinal_ss = StateSpace(A_lon, B_lon, C_lon, D_lon)

    A_wcl, B_wcl, C_wcl, D_wcl = get_longitudinal_state_space(boundary_ss[0])
    A_wcu, B_wcu, C_wcu, D_wcu = get_longitudinal_state_space(boundary_ss[1])

    worst_case_lower_tf = ss2tf(A_wcl, B_wcl, C_wcl, D_wcl)
    worst_case_upper_tf = ss2tf(A_wcu, B_wcu, C_wcu, D_wcu)
    long_tf = ss2tf(longitudinal_ss)

    # Initial filter design:
    M = 2
    w_0 = 1
    A_fact = 0.001

    nu_gap_l = nu_gap(long_tf, worst_case_lower_tf)
    nu_gap_u = nu_gap(long_tf, worst_case_upper_tf)
    omega_step = 0.1
    max_freq = 13.8
    while w_0 < max_freq:
        w_0 = w_0 + omega_step
        W_C = tf([1], [1])
        W_S = tf([1/M, w_0], [1, w_0*A_fact])
        W_T = tf([1, w_0/M], [A_fact, w_0])

        Plant = augw(longitudinal_ss, W_S, W_C, W_T)

        K_tmp, CL_tmp, gam_tmp, _ = hinfsyn(Plant, 1, 1, 0.001)
        controller = ss2tf(K_tmp.A, K_tmp.B, K_tmp.C, K_tmp.D)

        # Stability-requirements
        robust_margin = 1/gam_tmp
        stability = nu_gap_l < robust_margin and nu_gap_u < robust_margin

        # Robustness-requirements
        margins = check_margins(controller, long_tf,
                                worst_case_upper_tf, worst_case_lower_tf, stab_m=0.5)

        if stability and margins:
            K, _, gam = K_tmp, CL_tmp, gam_tmp
        else:
            break

    print(f"Longitudinal performance value (gamma) {gam}")
    print(f"Longitudinal controller crossover frequency (w_0) {w_0-omega_step}")

    return StateSpaceMatrices(A=K.A, B=K.B, C=K.C, D=K.D)


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

    nu_gap = minimize_scalar(f, bounds=(0.0001, 2**12), method='bounded')
    return (-f(nu_gap.x))


def init_robust_controller(controller: SaturatedStateSpaceController,
                           dir: Direction) -> NonlinearIOSystem:
    if dir == Direction.LONGITUDINAL:
        return pitch_hinf_controller(controller)
    elif dir == Direction.LATERAL:
        return roll_hinf_controller(controller)


def init_gs_controller(controllers: Sequence[SaturatedStateSpaceMatricesGS], dir: Direction,
                       switch_signal: str) -> NonlinearIOSystem:
    if dir == Direction.LONGITUDINAL:
        return pitch_gain_scheduled_controller(controllers, switch_signal)
    elif dir == Direction.LATERAL:
        return roll_gain_scheduled_controller(controllers, switch_signal)


def init_airspeed_controller(kp: float = 0.123, ki: float = 0.09, kaw: float = 2,
                             throttle_t: float = 0.58) -> NonlinearIOSystem:
    airspeed_controller_params = {
        "kp": kp,
        "ki": ki,
        "kaw": kaw,
        "throttle_trim": throttle_t
    }
    return airspeed_pi_controller(airspeed_controller_params)
