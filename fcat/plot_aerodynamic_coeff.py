from matplotlib import pyplot as plt
from typing import Sequence
from fcat import AircraftProperties, State, StateVecIndices
from fcat.utilities import calc_angle_of_sideslip
import numpy as np


__all__ = ('plot_aero_coeff',)


def plot_aero_coeff(prop: Sequence[AircraftProperties], states: Sequence[State],
                    wind: np.ndarray, index: StateVecIndices) -> plt.Figure:
    fig = plt.figure()
    ax_drag = fig.add_subplot(2, 3, 1)
    ax_side_force = fig.add_subplot(2, 3, 2)
    ax_lift = fig.add_subplot(2, 3, 3)
    ax_roll_moment = fig.add_subplot(2, 3, 4)
    ax_pitch_moment = fig.add_subplot(2, 3, 5)
    ax_yaw_moment = fig.add_subplot(2, 3, 6)

    x = [s.state[index] for s in states]
    x = [calc_angle_of_sideslip(s, wind.wind)*180/np.pi for s in states]
    xlabel = "AOS"
    for p in prop:
        drag = [p.drag_coeff(s, wind.wind) for s in states]
        side_force = [p.side_force_coeff(s, wind.wind) for s in states]
        lift = [p.lift_coeff(s, wind.wind) for s in states]
        roll = [p.roll_moment_coeff(s, wind.wind) for s in states]
        pitch = [p.pitch_moment_coeff(s, wind.wind) for s in states]
        yaw = [p.yaw_moment_coeff(s, wind.wind) for s in states]

        ax_drag.plot(x, drag, label=p.__class__)
        ax_side_force.plot(x, side_force, label=p.__class__)
        ax_lift.plot(x, lift, label=p.__class__)
        ax_roll_moment.plot(x, roll, label=p.__class__)
        ax_pitch_moment.plot(x, pitch, label=p.__class__)
        ax_yaw_moment.plot(x, yaw, label=p.__class__)

    ax_drag.set_xlabel(xlabel)
    ax_drag.set_ylabel('Drag coeff.')

    ax_side_force.set_xlabel(xlabel)
    ax_side_force.set_ylabel('Side force coeff.')

    ax_lift.set_xlabel(xlabel)
    ax_lift.set_ylabel('Lift coeff.')

    ax_roll_moment.set_xlabel(xlabel)
    ax_roll_moment.set_ylabel('Roll momment coeff.')

    ax_pitch_moment.set_xlabel(xlabel)
    ax_pitch_moment.set_ylabel('Pitch moment coeff.')

    ax_yaw_moment.set_xlabel(xlabel)
    ax_yaw_moment.set_ylabel('Yaw moment coeff')

    ax_drag.legend(["No ice", "Iced", "Asymetric"])
    return fig
