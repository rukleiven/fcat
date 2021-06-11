from fcat.inner_loop_controller import SaturatedStateSpaceMatricesGS
from typing import Sequence
import numpy as np
from fcat.utilities import saturate
from control import NonlinearIOSystem
from fcat.constants import (ROLL_ERROR, ROLL_GS_CONTROLLER, AILERON_DEFLECTION_COMMAND)

__all__ = ('roll_gain_scheduled_controller',)


def roll_gs_update(t, x, u, params={}):
    controllers = params.get('controllers')
    roll_error = u[0]
    switch_value = 0
    if (len(u) == 2):
        switch_value = u[1]
    controllers = params['controllers']
    switch_values = [c.switch_signal for c in controllers]
    controller_index = switch_values.index(min(switch_values, key=lambda x: abs(x-switch_value)))
    active_controller = controllers[controller_index]
    A = active_controller.A
    B = active_controller.B
    x_dot = A.dot(x).transpose() + np.multiply(B, roll_error)
    return x_dot


def roll_gs_output(t, x, u, params={}):
    roll_error = u[0]
    switch_value = 0
    if (len(u) == 2):
        switch_value = u[1]
    controllers = params['controllers']
    switch_values = [c.switch_signal for c in controllers]
    controller_index = switch_values.index(min(switch_values, key=lambda x: abs(x-switch_value)))
    active_controller = controllers[controller_index]
    C = active_controller.C
    D = active_controller.D

    y = C.dot(x) + D.dot(roll_error)

    aileron_deflection_min = active_controller.lower
    aileron_deflection_max = active_controller.upper

    aileron_deflection_command = saturate(y, aileron_deflection_min, aileron_deflection_max)
    return aileron_deflection_command


def roll_gain_scheduled_controller(controllers: Sequence[SaturatedStateSpaceMatricesGS],
                                   switch_value: str) -> NonlinearIOSystem:
    """
    Returns the roll gain scheduled controller as a nonlinearIOsystem

    :param params:
    """
    name = ROLL_GS_CONTROLLER
    inputs = (ROLL_ERROR, switch_value)
    outputs = AILERON_DEFLECTION_COMMAND

    # Find number of controller states:
    A = controllers[0].A
    nstates = A.shape[0]

    states = [f"lateral[{i}]" for i in range(nstates)]

    # Create params dict
    params = {"controllers": controllers}
    roll_gs_controller = NonlinearIOSystem(updfcn=roll_gs_update, outfcn=roll_gs_output,
                                           inputs=inputs, outputs=outputs, params=params,
                                           name=name, states=states)
    return roll_gs_controller
