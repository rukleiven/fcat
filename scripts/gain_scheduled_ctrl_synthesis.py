import click
from yaml import dump, load
from fcat.inner_loop_controller import (longitudinal_controller, lateral_controller,
                                        SaturatedStateSpaceMatricesGS, StateSpaceMatrices)
from fcat import (
    aircraft_property_from_dct, actuator_from_dct, ControlInput, State,
    build_nonlin_sys, no_wind)
from fcat.utilities import add_actuator
from typing import Sequence
from control import ssdata
import numpy as np


def update_icing_level(infile: str, icing_level: float):
    with open(infile) as f:
        data = load(f)
    data['aircraft']['icing'] = icing_level
    with open(infile, 'w') as f:
        dump(data, f)


def linearize_system(infile: str) -> str:
    with open(infile) as f:
        data = load(f)
    aircraft = aircraft_property_from_dct(data['aircraft'])
    ctrl = ControlInput.from_dict(data['init_control'])
    state = State.from_dict(data['init_state'])
    config_dict = {"outputs": ["x", "y", "z", "roll", "pitch", "yaw",
                               "vx", "vy", "vz", "ang_rate_x", "ang_rate_y",
                               "ang_rate_z"]
                   }
    sys = build_nonlin_sys(aircraft, no_wind(), config_dict['outputs'], None)
    actuator = actuator_from_dct(data['actuator'])
    xeq = state.state
    ueq = ctrl.control_input
    aircraft_with_actuator = add_actuator(actuator, sys)
    states_lin = np.concatenate((ueq, xeq))
    linearized_sys = aircraft_with_actuator.linearize(states_lin, ueq)
    A, B, C, D = ssdata(linearized_sys)
    linsys = StateSpaceMatrices(A, B, C, D)
    return linsys


def upper_lower_icing_levels(icing_levels: Sequence) -> list:
    lower_upper = []
    for i in range(len(icing_levels)):
        if i == 0:
            lower = 0
        else:
            lower = round(icing_levels[i] - (icing_levels[i]-icing_levels[i-1])/2, 2)
        if i == len(icing_levels) - 1:
            upper = 1
        else:
            upper = round(icing_levels[i] + (icing_levels[i+1] - icing_levels[i])/2, 2)
        lower_upper.append((lower, upper))
    return lower_upper


@click.command()
@click.option('--infile', help="yml file containing aircraft configurations")
@click.option('--longs_outfile', type=str, default=None, help="JSON file where the "
              "linearized state space model will be written")
@click.option('--latgs_outfile', type=str, default=None, help="JSON file where the "
              "linearized state space model will be written")
@click.option('--icing_levels', '-i', type=float, default=[], multiple=True,
              help="List of icing levels")
def gscs(infile: str, longs_outfile: str, latgs_outfile: str, icing_levels: Sequence):
    """
    Run controller synthesis

    param:
    """
    icing_levels = list(set(icing_levels))
    icing_levels.sort()
    ss_models = []

    for i in range(len(icing_levels)):
        update_icing_level(infile, icing_levels[i])
        ss_mod = linearize_system(infile)
        ss_models.append(ss_mod)

    lower_upper_icing = upper_lower_icing_levels(icing_levels)
    lower_upper_ss_models = []
    for lower_upper in lower_upper_icing:
        update_icing_level(infile, lower_upper[0])
        linsys_lower = linearize_system(infile)
        update_icing_level(infile, lower_upper[1])
        linsys_upper = linearize_system(infile)
        lower_upper_ss_models.append((linsys_lower, linsys_upper))

    longitudinal_controllers = []
    lateral_controllers = []
    for i in range(len(ss_models)):
        assert len(ss_models) == len(lower_upper_ss_models)
        l_u_ssmod_tuple = lower_upper_ss_models[i]
        # Longitudinal controller synthesis
        K = longitudinal_controller(ss_models[i], l_u_ssmod_tuple)

        longitudinal_controllers.append(SaturatedStateSpaceMatricesGS(A=K.A, B=K.B,
                                                                      C=K.C, D=K.D,
                                                                      lower=-0.4, upper=0.4,
                                                                      switch_signal=icing_levels[i]))
        # Lateral controller synthesis
        K = lateral_controller(ss_models[i], l_u_ssmod_tuple)
        lateral_controllers.append(SaturatedStateSpaceMatricesGS(A=K.A, B=K.B,
                                                                 C=K.C, D=K.D,
                                                                 lower=-0.4, upper=0.4,
                                                                 switch_signal=icing_levels[i]))
