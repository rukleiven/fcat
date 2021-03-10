import click
from fcat.inner_loop_controller import longitudinal_controller


@click.command()
@click.option('--infile', help="JSON file containing linearized state space model of aircraft")
@click.option('--outfile', default=None, help="JSON file where the "
                                              "linearized state space model will be written")
def longcs(infile: str, outfile: str):
    """
    Run controller synthesis
    """
    longitudinal_controller(infile, outfile)
