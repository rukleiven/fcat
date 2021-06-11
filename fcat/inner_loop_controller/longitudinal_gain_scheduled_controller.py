import numpy as np
from fcat.utilities import saturate
from control import NonlinearIOSystem
from fcat.constants import (PITCH_ERROR, PITCH_GS_CONTROLLER, ELEVATOR_DEFLECTION_COMMAND)
from fcat.inner_loop_controller import SaturatedStateSpaceMatricesGS
from typing import Sequence
__all__ = ('pitch_gain_scheduled_controller',)


def pitch_gs_update(t, x, u, params={}):
    controllers = params.get('controllers')
    pitch_error = u[0]
    switch_value = 0
    if (len(u) == 2):
        switch_value = u[1]
    controllers = params['controllers']
    switch_values = [c.switch_signal for c in controllers]
    controller_index = switch_values.index(min(switch_values, key=lambda x: abs(x-switch_value)))
    active_controller = controllers[controller_index]
    A = active_controller.A
    B = active_controller.B
    x_dot = A.dot(x).transpose() + np.multiply(B, pitch_error)
    return x_dot


def pitch_gs_output(t, x, u, params={}):
    # Get state-space matricies
    pitch_error = u[0]
    switch_value = 0
    if (len(u) == 2):
        switch_value = u[1]
    controllers = params['controllers']
    switch_values = [c.switch_signal for c in controllers]
    controller_index = switch_values.index(min(switch_values, key=lambda x: abs(x-switch_value)))
    active_controller = controllers[controller_index]
    C = active_controller.C
    D = active_controller.D

    y = C.dot(x) + D.dot(pitch_error)

    aileron_deflection_min = active_controller.lower
    aileron_deflection_max = active_controller.upper

    aileron_deflection_command = saturate(y, aileron_deflection_min, aileron_deflection_max)
    return aileron_deflection_command


def pitch_gain_scheduled_controller(controllers: Sequence[SaturatedStateSpaceMatricesGS],
                                    switch_value: str) -> NonlinearIOSystem:
    """
    Returns the pitch gain scheduled controller as a nonlinearIOsystem

    """
    name = PITCH_GS_CONTROLLER
    inputs = (PITCH_ERROR, switch_value)
    outputs = ELEVATOR_DEFLECTION_COMMAND

    # Find number of controller states:
    A = controllers[0].A
    nstates = A.shape[0]
    states = []
    states = [f"longitudinal[{i}]" for i in range(nstates)]

    # Create params dict
    # params = {f"controller{i}": dict(c) for i, c in enumerate(controllers)}
    params = {"controllers": controllers}
    pitch_gs_controller = NonlinearIOSystem(updfcn=pitch_gs_update, outfcn=pitch_gs_output,
                                            inputs=inputs, outputs=outputs, params=params,
                                            name=name, states=states)
    return pitch_gs_controller
