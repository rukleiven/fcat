import numpy as np
from fcat.utilities import saturate
from control import NonlinearIOSystem

__all__ = ('roll_pi_controller',)


def pi_update(t, x, u, params={}):
    ki = params.get('ki', 0.005)
    kaw = params.get('kaw', 0.01)  # anti-windup gain

    roll_error = u[0]

    # Compute nominal controller output
    aileron_nom = pi_output(t, x, roll_error, params)

    # Compute anti-windup compensation (scale by ki to account for structure)
    u_aw = kaw/ki * (np.clip(aileron_nom, 0, 1) - aileron_nom) if ki != 0 else 0

    return roll_error + u_aw


def pi_output(t, x, u, params={}):
    # Get the controller parameters that we need
    kp = params.get('kp', 0.05)
    ki = params.get('ki', 0.005)
    aileron_min = params.get('throttle_min', -0.3)
    aileron_max = params.get('throttle_max', 0.3)
    roll_error = u
    integrated_error = x[0]
    aileron = kp*roll_error + ki*integrated_error
    return saturate(aileron, aileron_min, aileron_max)


def roll_pi_controller(params={}) -> NonlinearIOSystem:
    """
    Returns the PI controller as a nonlinear systems

    :param params: Dictionary containing controller parameters ki, kp, kaw, throttle_min,
                   throttle_max
        - kp: proportional controller gain
        - ki: integrator controller gain
        - kaw: anti-windup controller gain
        - aileron_min: minimum throttle output value
        - aileron_max: maximum throttle output value
    """
    name = 'roll_pi_controller'
    inputs = 'roll_error'
    outputs = 'aileron_deflection_command'
    states = 'integrated_error'

    pi_controller = NonlinearIOSystem(updfcn=pi_update, outfcn=pi_output, inputs=inputs,
                                      outputs=outputs, states=states, params=params, name=name)
    return pi_controller
