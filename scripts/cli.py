#!/usr/bin/env python
import click
from linearize import linearize
from linstep import linstep


@click.group()
def entry_point():
    pass


entry_point.add_command(linearize)
entry_point.add_command(linstep)

if __name__ == '__main__':
    entry_point()
