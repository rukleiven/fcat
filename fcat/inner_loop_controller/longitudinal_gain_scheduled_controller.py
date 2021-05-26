import numpy as np
from fcat.utilities import saturate
from control import NonlinearIOSystem

__all__ = ('pitch_gain_scheduled_controller',)


def pitch_gs_update(t, x, u, params={}):
    controllers = params.get('controllers')
    pitch_error = u[0]
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

    x_dot = A.dot(x).transpose() + np.multiply(B, pitch_error)
    return x_dot


def pitch_gs_output(t, x, u, params={}):
    # Get state-space matricies
    pitch_error = u[0]
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

    elevator_deflection_min = params.get('elevator_deflection_min', -0.4)
    elevator_deflection_max = params.get('elevator_deflection_max', 0.4)
    pitch_error = u[0]
    y = C.dot(x) + D.dot(pitch_error)
    elevator_deflection_command = saturate(y, elevator_deflection_min, elevator_deflection_max)

    return elevator_deflection_command


def pitch_gain_scheduled_controller(params={}) -> NonlinearIOSystem:
    """
    Returns the pitch gain scheduled controller as a nonlinearIOsystem

    :param params: Dictionary containing controller parameters A,B,C,D,elevator_deflection_max,
                   elevator_deflection_min
        - A,B,C,D: state space matrices of controller
        - aileron_deflection_max: maximum elevator deflection (rad)
        - aileron_deflection_min: minimum elevator deflection (rad)
    """
    name = 'pitch_gs_controller'
    inputs = 'pitch_error', 'icing'
    outputs = 'elevator_deflection_command'
    controllers = params['controllers']
    # Find number of controller states:
    A = np.matrix(controllers.get(list(controllers.keys())[0]).get('A'))
    nstates = A.shape[0]
    states = []
    for i in range(nstates):
        states.append("longitudinal_controller["+str(i)+"]")

    pitch_gs_controller = NonlinearIOSystem(updfcn=pitch_gs_update, outfcn=pitch_gs_output,
                                            inputs=inputs, outputs=outputs, params=params,
                                            name=name, states=states)
    return pitch_gs_controller
