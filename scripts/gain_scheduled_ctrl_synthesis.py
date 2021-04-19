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


def linearize_system(infile: str):  # -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
    return np.array(A), np.array(B), np.array(C), np.array(D)


@click.command()
@click.option('--infile', help="JSON file containing linearized state space model of aircraft")
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

    for icing_level in icing_levels:
        update_icing_level(infile, icing_level)
        A, B, C, D = linearize_system(infile)

        # Longitudinal controller synthesis

        A_lon, B_lon, C_lon, D_lon = longitudinal_controller(None, A, B, C, D)

        controller = {
            'icing_level': icing_level,
            'A': A_lon.tolist(),
            'B': B_lon.tolist(),
            'C': C_lon.tolist(),
            'D': D_lon.tolist()
        }

        longitudinal_controllers['lon_contoller' +
                                 str(len(longitudinal_controllers.keys()))] = controller

        # Lateral controller synthesis
        A_lat, B_lat, C_lat, D_lat = lateral_controller(None, A, B, C, D)
        controller = {
            'icing_level': icing_level,
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
