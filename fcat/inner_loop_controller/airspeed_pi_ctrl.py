import numpy as np
from fcat.utilities import saturate
from control import NonlinearIOSystem

__all__ = ('airspeed_pi_controller',)


def pi_update(t, x, u, params={}):
    ki = params.get('ki', 0.1)
    kaw = params.get('kaw', 2)  # anti-windup gain

    airspeed_error = u[0]

    # Compute nominal controller output
    throttle_nom = pi_output(t, x, u, params)

    # Compute anti-windup compensation (scale by ki to account for structure)
    u_aw = kaw/ki * (np.clip(throttle_nom, 0, 1) - throttle_nom) if ki != 0 else 0

    return airspeed_error + u_aw


def pi_output(t, x, u, params={}):
    # Get the controller parameters that we need
    kp = params.get('kp', 0.5)
    ki = params.get('ki', 0.1)
    throttle_min = params.get('throttle_min', 0.0)
    throttle_max = params.get('throttle_max', 1.0)
    throttle_trim = params.get('throttle_trim', 0.5)
    airspeed_error = u[0]
    integrated_error = x[0]
    throttle = kp*airspeed_error + ki*integrated_error

    return saturate(throttle + throttle_trim, throttle_min, throttle_max)


def airspeed_pi_controller(params={}) -> NonlinearIOSystem:
    """
    Returns the PI controller as a nonlinear systems

    :param params: Dictionary containing controller parameters ki, kp, kaw, throttle_min,
                   throttle_max
        - kp: proportional controller gain
        - ki: integrator controller gain
        - kaw: anti-windup controller gain
        - throttle_min: minimum throttle output value
        - trhottle_max: maximum throttle output value
        - throttle_trim: throttle trim-point
    """
    name = 'airspeed_controller'
    inputs = 'airspeed_error'
    outputs = 'throttle_command'
    states = 'integrated_error'

    pi_controller = NonlinearIOSystem(updfcn=pi_update, outfcn=pi_output, inputs=inputs,
                                      outputs=outputs, states=states, params=params, name=name)
    return pi_controller
