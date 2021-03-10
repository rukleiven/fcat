import click
from fcat.inner_loop_controller import lateral_controller


@click.command()
@click.option('--infile', help="JSON file containing linearized state space model of aircraft")
@click.option('--outfile', default=None, help="JSON file where the "
                                              "linearized state space model will be written")
def latcs(infile: str, outfile: str):
    """
    Run controller synthesis
    """
    lateral_controller(infile, outfile)
