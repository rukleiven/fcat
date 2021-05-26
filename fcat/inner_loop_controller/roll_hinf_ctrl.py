import numpy as np
from fcat.utilities import saturate
from control import NonlinearIOSystem

__all__ = ('roll_hinf_controller',)


def roll_hinf_update(t, x, u, params={}):
    # Get state-space matricies
    A = params.get('A')
    B = params.get('B')

    roll_error = u[0]
    x_dot = A.dot(x.reshape(len(x), 1)) + np.multiply(B, roll_error)
    return x_dot


def roll_hinf_output(t, x, u, params={}):
    # Get state-space matricies
    C = params.get('C')
    D = params.get('D')

    aileron_deflection_min = params.get('aileron_deflection_min', -0.6)
    aileron_deflection_max = params.get('aileron_deflection_max', 0.6)
    roll_error = u[0]
    y = C.dot(x) + D.dot(roll_error)
    # print("roll_error:")
    # print(roll_error)
    # print("output")
    # print(y)
    aileron_deflection_command = saturate(y, aileron_deflection_min, aileron_deflection_max)
    return aileron_deflection_command


def roll_hinf_controller(params={}) -> NonlinearIOSystem:
    """
    Returns the rollhinf controller as a nonlinearIOsystem

    :param params: Dictionary containing controller parameters A,B,C,D,aileron_deflection_max,
                   aileron_deflection_min
        - A,B,C,D: state space matrices of controller
        - aileron_deflection_max: maximum aileron deflection (rad)
        - aileron_deflection_min: minimum aileron deflection (rad)
    """
    name = 'roll_hinf_controller'
    inputs = 'roll_error'
    outputs = 'aileron_deflection_command'

    # Find number of controller states:
    A = params.get('A')
    nstates = len(A)
    states = []
    for i in range(nstates):
        states.append("lateral_controller["+str(i)+"]")

    roll_hinf_controller = NonlinearIOSystem(updfcn=roll_hinf_update, outfcn=roll_hinf_output,
                                             inputs=inputs, outputs=outputs, params=params,
                                             name=name, states=states)
    return roll_hinf_controller
