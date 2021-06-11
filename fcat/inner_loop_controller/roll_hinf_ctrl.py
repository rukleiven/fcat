import numpy as np
from fcat.utilities import saturate
from control import NonlinearIOSystem
from fcat.constants import (ROLL_ERROR, ROLL_HINF_CONTROLLER, AILERON_DEFLECTION_COMMAND)
from fcat.inner_loop_controller import SaturatedStateSpaceController
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

    aileron_deflection_min = params.get('lower', -0.4)
    aileron_deflection_max = params.get('upper', 0.4)
    roll_error = u[0]
    y = C.dot(x) + D.dot(roll_error)
    aileron_deflection_command = saturate(y, aileron_deflection_min, aileron_deflection_max)
    return aileron_deflection_command


def roll_hinf_controller(controller: SaturatedStateSpaceController) -> NonlinearIOSystem:
    """
    Returns the rollhinf controller as a NonlinearIOsystem

    :params controller:
    """
    name = ROLL_HINF_CONTROLLER
    inputs = ROLL_ERROR
    outputs = AILERON_DEFLECTION_COMMAND

    # Find number of controller states:
    A = controller.A
    nstates = A.shape[0]
    states = []
    for i in range(nstates):
        states.append("lateral_controller["+str(i)+"]")

    params = {'A': controller.A,
              'B': controller.B,
              'C': controller.C,
              'D': controller.D,
              'upper': controller.upper,
              'lower': controller.lower
              }

    roll_hinf_controller = NonlinearIOSystem(updfcn=roll_hinf_update, outfcn=roll_hinf_output,
                                             inputs=inputs, outputs=outputs, params=params,
                                             name=name, states=states)
    return roll_hinf_controller
