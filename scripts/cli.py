#!/usr/bin/env python
import click
from linearize import linearize
from linstep import linstep
from lateral_controller_synthesis import latcs
from longitudinal_controller_synthesis import longcs


@click.group()
def entry_point():
    pass


entry_point.add_command(linearize)
entry_point.add_command(linstep)
entry_point.add_command(longcs)
entry_point.add_command(latcs)

if __name__ == '__main__':
    entry_point()
