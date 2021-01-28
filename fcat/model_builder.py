import numpy as np
from fcat import AircraftProperties, ControlInput, State
from fcat.state_vector import StateVecIndices
from fcat.utilities import (calc_airspeed, wind2body, inertial2body,
                            body2euler, body2inertial)
from control.iosys import NonlinearIOSystem
from fcat.simulation_constants import RHO, GRAVITY_CONST

__all__ = ('build_nonlin_sys',)


def dynamics_kinetmatics_update(t: float, x: np.ndarray, u: np.ndarray, params: dict) -> np.ndarray:
    prop = params['prop']
    wind = params['wind']
    state = State(init=x)

    # Update the control inputs
    prop.control_input = ControlInput(init=u)

    update = np.zeros_like(x)

    V_a = np.sqrt(np.sum(calc_airspeed(state, wind)**2))
    b = prop.wing_span()
    S = prop.wing_area()
    c = prop.mean_chord()
    S_prop = prop.propeller_area()
    C_prop = prop.motor_efficiency_fact()
    k_motor = prop.motor_constant()
    qS = 0.5*RHO*S*V_a**2
    coeff_wind_frame = np.array([-prop.drag_coeff(state, wind),
                                 prop.side_force_coeff(state, wind),
                                 -prop.lift_coeff(state, wind)])
    force_body_frame = qS*wind2body(coeff_wind_frame, state, wind)

    moment_coeff_vec = np.array([b*prop.roll_moment_coeff(state, wind),
                                 c*prop.pitch_moment_coeff(state, wind),
                                 b*prop.yaw_moment_coeff(state, wind)])
    moment_vec = qS*moment_coeff_vec
    omega = np.array([state.roll_dot, state.pitch_dot, state.yaw_dot]) - wind[3:]
    # omega = body2inertial(np.array([state.roll_dot, state.pitch_dot, state.yaw_dot])
    #  - wind[3:], state)
    velocity = np.array([state.vx, state.vy, state.vz]) - wind[:3]

    gravity_body_frame = inertial2body([0.0, 0.0, prop.mass()*GRAVITY_CONST], state)
    F_propulsion = 0.5*RHO*S_prop*C_prop * \
        np.array([(k_motor*prop.control_input.throttle)**2-V_a**2, 0, 0])
    v_dot = (force_body_frame + gravity_body_frame + F_propulsion) / \
        prop.mass() - np.cross(omega, velocity)

    # Velocity update
    update[StateVecIndices.V_X:StateVecIndices.V_Z+1] = v_dot
    # Momentum equations
    omega_dot = \
        prop.inv_inertia_matrix().dot(moment_vec - np.cross(omega, prop.inertia_matrix().dot(omega)))
    update[StateVecIndices.ROLL_DOT:StateVecIndices.YAW_DOT+1] = omega_dot
    # Kinematics
    # Position updates
    update[StateVecIndices.X:StateVecIndices.Z +
           1] = body2inertial([state.vx, state.vy, state.vz], state)

    # Angle updates
    update[StateVecIndices.ROLL:StateVecIndices.YAW +
           1] = body2euler([state.roll_dot, state.pitch_dot, state.yaw_dot], state)
    return update


def build_nonlin_sys(prop: AircraftProperties, wind: np.ndarray) -> NonlinearIOSystem:
    inputs = ('elevator_deflection', 'aileron_deflection', 'rudder_deflection', 'throttle')
    states = ('x', 'y', 'z', 'roll', 'pitch', 'yaw', 'vx',
              'vy', 'vz', 'roll_dot', 'pitch_dot', 'yaw_dot')
    system = NonlinearIOSystem(
        dynamics_kinetmatics_update, inputs=inputs, states=states,
        params={'prop': prop, 'wind': wind}, outputs=states,
        name='dynamics_kinematics'
    )
    return system
