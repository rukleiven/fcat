import numpy as np
from fcat.utilities import saturate
from control import NonlinearIOSystem
from fcat.constants import (PITCH_ERROR, PITCH_HINF_CONTOLLER, ELEVATOR_DEFLECTION_COMMAND)
from fcat.inner_loop_controller import SaturatedStateSpaceController

__all__ = ('pitch_hinf_controller',)


def pitch_hinf_update(t, x, u, params={}):
    # Get state-space matricies
    A = params.get('A')
    B = params.get('B')

    pitch_error = u[0]
    x_dot = A.dot(x.reshape(len(x), 1)) + np.multiply(B, pitch_error)
    return x_dot


def pitch_hinf_output(t, x, u, params={}):
    # Get state-space matricies
    C = params.get('C')
    D = params.get('D')
    elevator_deflection_min = params.get('lower', -0.4)
    elevator_deflection_max = params.get('upper', 0.4)
    pitch_error = u[0]

    y = C.dot(x) + D.dot(pitch_error)
    elevator_deflection_command = saturate(y, elevator_deflection_min, elevator_deflection_max)
    return elevator_deflection_command


def pitch_hinf_controller(controller: SaturatedStateSpaceController) -> NonlinearIOSystem:
    """
    Returns the pitchhinf controller as a nonlinearIOsystem

    :param controller: Instant of saturated controller
    """
    name = PITCH_HINF_CONTOLLER
    inputs = PITCH_ERROR
    outputs = ELEVATOR_DEFLECTION_COMMAND

    # Find number of controller states:
    A = controller.A
    nstates = A.shape[0]
    states = []
    for i in range(nstates):
        states.append("longitudinal_controller["+str(i)+"]")

    params = {'A': controller.A,
              'B': controller.B,
              'C': controller.C,
              'D': controller.D,
              'upper': controller.upper,
              'lower': controller.lower
              }

    pitch_hinf_controller = NonlinearIOSystem(updfcn=pitch_hinf_update, outfcn=pitch_hinf_output,
                                              inputs=inputs, outputs=outputs, params=params,
                                              name=name, states=states)
    return pitch_hinf_controller
