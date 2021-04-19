#!/usr/bin/env python
import click
from linearize import linearize
from linstep import linstep
from lateral_controller_synthesis import latcs
from longitudinal_controller_synthesis import loncs
from gain_scheduled_ctrl_synthesis import gscs


@click.group()
def entry_point():
    pass


entry_point.add_command(linearize)
entry_point.add_command(linstep)
entry_point.add_command(loncs)
entry_point.add_command(latcs)
entry_point.add_command(gscs)

if __name__ == '__main__':
    entry_point()
