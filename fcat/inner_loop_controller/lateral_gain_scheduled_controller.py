import numpy as np
from fcat.utilities import saturate
from control import NonlinearIOSystem

__all__ = ('roll_gain_scheduled_controller',)


def roll_gs_update(t, x, u, params={}):
    controllers = params.get('controllers')
    roll_error = u[0]
    icing_level = 0
    if (len(u) == 2):
        icing_level = u[1]
    dict_keys = list(controllers.keys())
    icing_levels = []
    for key in dict_keys:
        icing_levels.append(controllers[key]['icing_level'])
    controller_index = icing_levels.index(min(icing_levels, key=lambda x: abs(x-icing_level)))
    A = np.matrix(controllers[dict_keys[controller_index]]['A'])
    B = np.matrix(controllers[dict_keys[controller_index]]['B'])
    x_dot = A.dot(x).transpose() + np.multiply(B, roll_error)
    return x_dot


def roll_gs_output(t, x, u, params={}):
    roll_error = u[0]
    icing_level = 0
    if (len(u) == 2):
        icing_level = u[1]
    controllers = params['controllers']
    dict_keys = list(controllers.keys())
    icing_levels = []
    for key in dict_keys:
        icing_levels.append(controllers[key]['icing_level'])
    controller_index = icing_levels.index(min(icing_levels, key=lambda x: abs(x-icing_level)))
    C = np.matrix(controllers[dict_keys[controller_index]]['C'])
    D = np.matrix(controllers[dict_keys[controller_index]]['D'])

    y = C.dot(x) + D.dot(roll_error)

    aileron_deflection_min = params.get('aileron_deflection_min', -0.4)
    aileron_deflection_max = params.get('aileron_deflection_max', 0.4)

    aileron_deflection_command = saturate(y, aileron_deflection_min, aileron_deflection_max)
    return aileron_deflection_command


def roll_gain_scheduled_controller(params={}) -> NonlinearIOSystem:
    """
    Returns the roll gain scheduled controller as a nonlinearIOsystem

    :param params: Dictionary containing controller parameters A,B,C,D,aileron_deflection_max,
                   aileron_deflection_min
        - A,B,C,D: state space matrices of controller
        - aileron_deflection_max: maximum aileron deflection (rad)
        - aileron_deflection_min: minimum aileron deflection (rad)
    """
    name = 'roll_gs_controller'
    inputs = 'roll_error', 'icing'
    outputs = 'aileron_deflection_command'

    controllers = params['controllers']
    # Find number of controller states:
    A = np.matrix(controllers.get(list(controllers.keys())[0]).get('A'))
    nstates = A.shape[0]
    states = []
    for i in range(nstates):
        states.append("lateral_controller["+str(i)+"]")

    roll_gs_controller = NonlinearIOSystem(updfcn=roll_gs_update, outfcn=roll_gs_output,
                                           inputs=inputs, outputs=outputs, params=params,
                                           name=name, states=states)
    return roll_gs_controller
