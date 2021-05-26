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
    wcl_filename = \
        "./examples/skywalkerX8_analysis/SkywalkerX8_state_space_models/skywalkerx8_linmod.json"
    wcu_filename = \
        "./examples/skywalkerX8_analysis/SkywalkerX8_state_space_models/"
    wcu_filename += "skywalkerx8_linmod_icing10.json"

    lateral_controller(infile, wcl_filename, wcu_filename, outfile)
