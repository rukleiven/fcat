from control.iosys import LinearIOSystem
from control import StateSpace
from fcat.utilities import flying_wing2ctrl_input_matrix
from typing import Sequence
import numpy as np


def get_MIMO_state_space(elevon_time_constant: float, motor_time_constant: float) -> Sequence:
    A = np.array([[-1/elevon_time_constant, 0, 0, 0],
                  [0, -1/elevon_time_constant, 0, 0],
                  [0, 0, -1, 0],
                  [0, 0, 0, -1/motor_time_constant]])
    B = -A.copy()
    C = np.eye(4)
    C[2, 2] = 0
    D = np.zeros_like(B)
    return [A, B, C, D]


def build_flying_wing_actuator_system(elevon_time_constant: float,
                                      motor_time_constat: float) -> LinearIOSystem:
    A_flying_wing, B_flying_wing, C_flying_wing, D_flying_wing = get_MIMO_state_space(
        elevon_time_constant, motor_time_constat)
    print(A_flying_wing)
    print(B_flying_wing)
    transform_matrix = flying_wing2ctrl_input_matrix()
    inv_transform_matrix = np.linalg.inv(transform_matrix)
    A = transform_matrix.dot(A_flying_wing).dot(inv_transform_matrix)
    B = transform_matrix.dot(B_flying_wing).dot(inv_transform_matrix)
    C = transform_matrix.dot(C_flying_wing).dot(inv_transform_matrix)
    D = transform_matrix.dot(D_flying_wing).dot(inv_transform_matrix)
    lin_sys = StateSpace(A, B, C, D)
    inputs = ('elevator_deflection_command', 'aileron_deflection_command',
              'rudder_deflection_command', 'throttle_command')
    states = ('elevator_deflection', 'aileron_deflection', 'rudder_deflection', 'throttle')
    outputs = ('elevator_deflection', 'aileron_deflection', 'rudder_deflection', 'throttle')
    name = 'actuator_model'
    return LinearIOSystem(lin_sys, inputs=inputs, outputs=outputs, states=states, name=name)
