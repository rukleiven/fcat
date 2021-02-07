import click
import json
import numpy as np
from control import StateSpace, step_response
from matplotlib import pyplot as plt


def plot_respons(t: np.ndarray, states: np.ndarray):
    fig = plt.figure()

    for i in range(states.shape[0]):
        ax = fig.add_subplot(3, 4, i+1)
        ax.plot(t, states[i, :])
        ax.set_xlabel("Time")
        ax.set_ylabel(f"State {i}")
    return fig


@click.command()
@click.option("--mod", help="JSON file with a linear model. For example output from linearize.")
@click.option("--step_var", default=0, type=int, help="Index of the"
              "input where step should be applied")
@click.option("--out", default="", help="CSV file where output will be stored. If not given or"
              "empty string, no output will be written")
@click.option("--show", is_flag=True, help="If given, plots of the output will be generated")
@click.option("--tmax", type=float, default=10.0, help="Simulation time")
@click.option("--nt", type=int, default=100, help="Number of time steps between 0 and tmax.")
def linstep(mod: str, step_var: int, out: str, show: bool, tmax: float, nt: int):
    """
    Calculates the step response of a linear model
    """
    with open(mod, 'r') as infile:
        data = json.load(infile)

    ss = StateSpace(data['A'], data['B'], data['C'], data['D'])
    t = np.linspace(0.0, tmax, nt)
    T, yout = step_response(ss, T=t, input=step_var)

    if out != "":
        all_data = np.vstack((T, yout))
        np.savetxt(out, all_data.T, delimiter=",")
        print(f"Step result response written to {out}")

    if show:
        plot_respons(T, yout)
        plt.show()
