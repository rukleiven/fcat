import click
from yaml import dump, load
from control.iosys import find_eqpt
from control import ssdata
import numpy as np
from fcat import (
    aircraft_property_from_dct, actuator_from_dct, ControlInput, State,
    build_nonlin_sys, no_wind
)
import json

TEMPLATE_FILE = "linearize_template.yml"


def rad2deg(val: float) -> float:
    return val*180/np.pi


def linearize_input_template() -> dict:
    template_input = {
        'init_state': {
            'x': 0.0,
            'y': 0.0,
            'z': 0.0,
            'roll': 0.0,
            'pitch': 0.0,
            'yaw': 0.0,
            'vx': 20.0,
            'vy': 0.0,
            'vz': 1.0,
            'ang_rate_x': 0.0,
            'ang_rate_y': 0.0,
            'ang_rate_z': 0.0
        },
        'init_control': {
            'elevator_deflection': 0.0,
            'aileron_deflection': 0.0,
            'rudder_deflection': 0.0,
            'throttle': 0.5,
        },
        'aircraft': {
            'type': 'skywalkerX8',
            'icing': 0.0,
        },
        'actuator': {
            'type': 'none'
        }
    }
    return template_input


def generate_linearize_template():
    input_dict = linearize_input_template()
    with open(TEMPLATE_FILE, 'w') as out:
        dump(input_dict, out, default_flow_style=False)
    print(f"Template for the input yml file written to {TEMPLATE_FILE}")


@click.command()
@click.option('--config', help="YML file describing the system")
@click.option('--out', default=None, help="JSON file where the linearized state space model will "
                                          "be written. If not given, the model will not be stored")
@click.option('--template', is_flag=True, help="If given, a template for the input file will "
                                               "be generated")
@click.option('--trim', is_flag=True, help="If given, finding equilibrium with initial guess given "
              "in yml file, otherwise linearize at the values "
              "in yml file")
def linearize(config: str, out: str, template: bool = False, trim: bool = False):
    """
    Linearize the model specified via the config file
    """
    if template:
        generate_linearize_template()
        return

    with open(config, 'r') as infile:
        data = load(infile)

    aircraft = aircraft_property_from_dct(data['aircraft'])
    # TODO: Fix this!
    actuator = actuator_from_dct(data['actuator'])
    _ = actuator

    ctrl = ControlInput.from_dict(data['init_control'])
    state = State.from_dict(data['init_state'])
    iu = [2, 3]
    sys = build_nonlin_sys(aircraft, no_wind(), None)

    idx = [2, 6, 7, 8, 9, 10, 11]
    y0 = state.state
    iy = [0, 1, 2, 5, 9, 10, 11]

    xeq = state.state
    ueq = ctrl.control_input

    if trim:
        print("Finding equillibrium point...")
        xeq, ueq = find_eqpt(sys, state.state, u0=ctrl.control_input, idx=idx, y0=y0, iy=iy, iu=iu)
        print("Equillibrium point found")
        print()
        print("Equilibrium state vector")
        print(f"x: {xeq[0]: .2e} m, y: {xeq[1]: .2e} m, z: {xeq[2]: .2e} m")
        print(f"roll: {rad2deg(xeq[3]): .1f} deg, pitch: {rad2deg(xeq[4]): .1f} deg"
              f", yaw: {rad2deg(xeq[5]): .1f} deg")
        print(f"vx: {xeq[6]: .2e} m/s, vy: {xeq[7]: .2e} m/s, vz: {xeq[8]: .2e} m/s")
        print(f"Ang.rates: x: {rad2deg(xeq[9]): .1f} deg/s, y: {rad2deg(xeq[10]): .1f} deg/s"
              f", z: {rad2deg(xeq[11]): .1f} deg/s")
        print()
        print("Equilibrium input control vector")
        print(f"elevator: {rad2deg(ueq[0]): .1f} deg, aileron: {rad2deg(ueq[1]): .1f} deg"
              f", rudder: {rad2deg(ueq[2]): .1f} deg, throttle: {ueq[3]: .1f}")
        print()

    linearized = sys.linearize(xeq, ueq)
    print("Linearization finished")
    A, B, C, D = ssdata(linearized)

    print("Found linear state space model on the form")
    print()
    print("  dx  ")
    print(" ---- = Ax + Bu")
    print("  dt ")
    print()
    print("With observarion equation")
    print()
    print("y = Cx + Du")
    print()

    print("Eigen values of A:")
    eig = np.linalg.eigvals(A)
    print('\n'.join(f'{x:.2e}' for x in eig))

    linsys = {
        'A': A.tolist(),
        'B': B.tolist(),
        'C': C.tolist(),
        'D': D.tolist(),
        'xeq': xeq.tolist(),
        'ueq': ueq.tolist()
    }

    if out is not None:
        with open(out, 'w') as outfile:
            json.dump(linsys, outfile, indent=2, sort_keys=True)
        print(f"Linear model written to {out}")
