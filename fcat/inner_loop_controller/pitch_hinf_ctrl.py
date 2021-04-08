import numpy as np
from fcat.utilities import saturate
from control import NonlinearIOSystem

__all__ = ('pitch_hinf_controller',)


def pitch_hinf_update(t, x, u, params={}):
    # Get state-space matricies
    A = params.get('A')
    B = params.get('B')

    pitch_error = u[0]
    x_dot = A.dot(x).transpose() + np.multiply(B, pitch_error)
    return x_dot


def pitch_hinf_output(t, x, u, params={}):
    # Get state-space matricies
    C = params.get('C')
    D = params.get('D')

    elevator_deflection_min = params.get('elevator_deflection_min', -0.4)
    elevator_deflection_max = params.get('elevator_deflection_max', 0.4)
    pitch_error = u[0]

    y = C.dot(x) + D.dot(pitch_error)
    elevator_deflection_command = saturate(y, elevator_deflection_min, elevator_deflection_max)
    return elevator_deflection_command


def pitch_hinf_controller(params={}) -> NonlinearIOSystem:
    """
    Returns the pitchhinf controller as a nonlinearIOsystem

    :param params: Dictionary containing controller parameters A,B,C,D,elevator_deflection_max,
                   elevator_deflection_min
        - A,B,C,D: state space matrices of controller
        - aileron_deflection_max: maximum elevator deflection (rad)
        - aileron_deflection_min: minimum elevator deflection (rad)
    """
    name = 'pitch_hinf_controller'
    inputs = 'pitch_error'
    outputs = 'elavtor_deflection_command_command'

    # Find number of controller states:
    A = params.get('A')
    nstates = len(A)
    states = []
    for i in range(nstates):
        states.append("longitudinal_controller["+str(i)+"]")

    pitch_hinf_controller = NonlinearIOSystem(updfcn=pitch_hinf_update, outfcn=pitch_hinf_output,
                                              inputs=inputs, outputs=outputs, params=params,
                                              name=name, states=states)
    return pitch_hinf_controller
