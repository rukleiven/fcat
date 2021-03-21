import numpy as np
from control.iosys import NonlinearIOSystem


def update_func(t: float, x: np.ndarray, u: np.ndarray, params: dict) -> np.ndarray:
    A = params.get('A')
    B = params.get('B')
    elev_cmd = u[0]
    x_dot = x_dot = A.dot(x).transpose() + np.multiply(B, elev_cmd)
    return x_dot


def output_func(t: float, x: np.ndarray, u: np.ndarray, params: dict):
    C = params.get('C')
    D = params.get('D')

    elev_cmd = u[0]
    y = C.dot(x) + D.dot(elev_cmd)
    return y


def build_lin_sys_lon(A, B, C, D) -> NonlinearIOSystem:
    """
    Construct a linear IO system as NonlinearIOSystem from passed input

    :param A, B, C, D: state space matrices
    """
    inputs = 'elev_cmd'
    outputs = 'pitch_out'
    name = 'lin_sys'
    states = ('delta_e', 'delta_t', 'pitch', 'vx', 'vz', 'ang_rate_y')
    system = NonlinearIOSystem(
        update_func, inputs=inputs, name=name, outputs=outputs, outfcn=output_func,
        params={'A': A, 'B': B, 'C': C, 'D': D}, states=states
    )
    return system
