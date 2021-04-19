import pytest
import numpy as np
from fcat import State
from fcat import ControlInput
from fcat import IcedSkywalkerX8Properties
from fcat.skywalker8 import SkywalkerX8Constants
from fcat.simulation_constants import AIR_DENSITY
from fcat.model_builder import dynamics_kinetmatics_update
from fcat import no_wind


def test_skywalkerX8_force_x_dir():
    control_input = ControlInput()
    control_input.throttle = 0.0
    t = 0
    state = State()
    state.vx = 20.0

    for j in range(3):
        state.pitch = state.pitch + 0.05
        state.roll = state.roll + 0.05
        state.yaw = state.yaw + 0.05
        x_update = np.zeros(11)
        constants = SkywalkerX8Constants()
        for i in range(0, 11):
            control_input.throttle = i*0.1
            prop = IcedSkywalkerX8Properties(control_input)
            params = {
                "prop": prop,
                "wind": no_wind()
            }
            update = dynamics_kinetmatics_update(
                t, x=state.state, u=control_input.control_input, params=params)
            x_update[i] = update[6]

        S_p = constants.propeller_area
        C_p = constants.motor_efficiency_fact
        k_m = constants.motor_constant
        m = constants.mass
        K = 2*m/(AIR_DENSITY*S_p*C_p*k_m**2)
        for i in range(0, 10):
            throttle_0 = i*0.1
            throttle_1 = (i+1)*0.1
            assert np.allclose(K*(x_update[i+1]-x_update[i]), throttle_1**2 - throttle_0**2)
