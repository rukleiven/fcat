import click
from fcat.inner_loop_controller import lateral_controller, get_state_space_from_file
import json


@click.command()
@click.option('--infile', help="JSON file containing linearized state space model of aircraft")
@click.option('--outfile', default=None, help="JSON file where the "
                                              "linearized state space model will be written")
def latcs(infile: str, outfile: str, lower_ss_fname: str, upper_ss_fname: str):
    """
    Run controller synthesis for skywalkerX8

    """
    # TODO: Take in boundary filenames as additional arguments?
    boundary_ss = (get_state_space_from_file(lower_ss_fname),
                   get_state_space_from_file(upper_ss_fname))
    nom_ss = get_state_space_from_file(infile)
    K = lateral_controller(nom_ss, boundary_ss)

    # Write controller to file:

    if outfile is not None:
        lat_controller = {
            'A': (K.A).tolist(),
            'B': (K.B).tolist(),
            'C': (K.C).tolist(),
            'D': (K.D).tolist()
        }

        with open(outfile, 'w') as outfile:
            json.dump(lat_controller, outfile, indent=2, sort_keys=True)
        print(f"Lateral controller written to {outfile}")
