import click
from yaml import dump, load
import json
from fcat.inner_loop_controller import longitudinal_controller, lateral_controller
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


def linearize_system(infile: str, index: int) -> str:
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
    linsys = {
        'A': np.array(A).tolist(),
        'B': np.array(B).tolist(),
        'C': np.array(C).tolist(),
        'D': np.array(D).tolist(),
        'xeq': xeq.tolist(),
        'ueq': ueq.tolist()
    }

    f_name = "examples/skywalkerX8_analysis/ss_mod_gs/"
    f_name += "skywalkerX8_linmod_icing" + str(index) + ".json"
    with open(f_name, 'w') as outfile:
        json.dump(linsys, outfile, indent=2, sort_keys=True)
    print(f"Linear model written to {f_name}")
    return f_name


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
    """
    longitudinal_controllers = {}
    lateral_controllers = {}
    icing_levels = list(set(icing_levels))
    icing_levels.sort()
    ss_models = []
    for i in range(len(icing_levels)):
        update_icing_level(infile, icing_levels[i])
        f_name = linearize_system(infile, i)
        ss_models.append(f_name)

    lower_upper_icing = upper_lower_icing_levels(icing_levels)
    lower_upper_fnames = []
    name_indx = len(icing_levels)
    for lower_upper in lower_upper_icing:
        update_icing_level(infile, lower_upper[0])
        f_name_lower = linearize_system(infile, name_indx)
        name_indx += 1
        update_icing_level(infile, lower_upper[1])
        f_name_upper = linearize_system(infile, name_indx)
        name_indx += 1
        lower_upper_fnames.append((f_name_lower, f_name_upper))

    for i in range(len(ss_models)):
        assert len(ss_models) == len(lower_upper_fnames)
        l_u_fname_tuple = lower_upper_fnames[i]
        lower_fname = l_u_fname_tuple[0]
        upper_fname = l_u_fname_tuple[1]
        # Longitudinal controller synthesis
        A_lon, B_lon, C_lon, D_lon = longitudinal_controller(ss_models[i], lower_fname, upper_fname)

        controller = {
            'icing_level': icing_levels[i],
            'A': A_lon.tolist(),
            'B': B_lon.tolist(),
            'C': C_lon.tolist(),
            'D': D_lon.tolist()
        }

        longitudinal_controllers['lon_contoller' +
                                 str(len(longitudinal_controllers.keys()))] = controller

        # Lateral controller synthesis
        A_lat, B_lat, C_lat, D_lat = lateral_controller(ss_models[i], lower_fname, upper_fname)
        controller = {
            'icing_level': icing_levels[i],
            'A': A_lat.tolist(),
            'B': B_lat.tolist(),
            'C': C_lat.tolist(),
            'D': D_lat.tolist()
        }
        lateral_controllers['lat_controller'+str(len(lateral_controllers.keys()))] = controller

    if longs_outfile is not None:
        with open(longs_outfile, 'w') as outfile:
            json.dump(longitudinal_controllers, outfile, indent=2, sort_keys=True)
    if latgs_outfile is not None:
        with open(latgs_outfile, 'w') as outfile:
            json.dump(lateral_controllers, outfile, indent=2, sort_keys=True)
