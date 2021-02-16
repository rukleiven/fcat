import numpy as np
from fcat import AircraftProperties, ControlInput, State
from fcat.state_vector import StateVecIndices
from fcat.utilities import (calc_airspeed, wind2body, inertial2body,
                            body2euler, body2inertial)
from control.iosys import NonlinearIOSystem
from fcat.simulation_constants import AIR_DENSITY, GRAVITY_CONST
from fcat import PropertyUpdater, WindModel

__all__ = ('build_nonlin_sys',)


def dynamics_kinetmatics_update(t: float, x: np.ndarray, u: np.ndarray, params: dict) -> np.ndarray:
    prop = params['prop']
    wind = params['wind'].get(t)
    prop_updater = params.get('prop_updater', None)
    state = State(init=x)

    # Update the control inputs
    prop.control_input = ControlInput(init=u)

    if prop_updater is not None:
        prop.update_params(prop_updater.get_param_dict(t))

    update = np.zeros_like(x)

    V_a = np.sqrt(np.sum(calc_airspeed(state, wind)**2))
    b = prop.wing_span()
    S = prop.wing_area()
    c = prop.mean_chord()
    S_prop = prop.propeller_area()
    C_prop = prop.motor_efficiency_fact()
    k_motor = prop.motor_constant()
    qS = 0.5*AIR_DENSITY*S*V_a**2
    force_aero_wind_frame = qS*np.array([-prop.drag_coeff(state, wind),
                                        prop.side_force_coeff(state, wind),
                                        -prop.lift_coeff(state, wind)])
    force_aero_body_frame = wind2body(force_aero_wind_frame, state, wind)

    moment_coeff_vec = np.array([b*prop.roll_moment_coeff(state, wind),
                                 c*prop.pitch_moment_coeff(state, wind),
                                 b*prop.yaw_moment_coeff(state, wind)])
    moment_vec = qS*moment_coeff_vec
    omega = np.array([state.ang_rate_x, state.ang_rate_y, state.ang_rate_z]) - wind[3:]
    velocity = np.array([state.vx, state.vy, state.vz]) - wind[:3]

    gravity_body_frame = inertial2body([0.0, 0.0, prop.mass()*GRAVITY_CONST], state)
    F_propulsion = 0.5*AIR_DENSITY*S_prop*C_prop * \
        np.array([(k_motor*prop.control_input.throttle)**2-V_a**2, 0, 0])
    v_dot = (force_aero_body_frame + gravity_body_frame + F_propulsion) / \
        prop.mass() - np.cross(omega, velocity)

    # Velocity update
    update[StateVecIndices.V_X:StateVecIndices.V_Z+1] = v_dot
    # Momentum equations
    omega_dot = \
        prop.inv_inertia_matrix().dot(moment_vec - np.cross(omega, prop.inertia_matrix().dot(omega)))
    update[StateVecIndices.ANG_RATE_X:StateVecIndices.ANG_RATE_Z+1] = omega_dot
    # Kinematics
    # Position updates
    update[StateVecIndices.X:StateVecIndices.Z +
           1] = body2inertial([state.vx, state.vy, state.vz], state)

    # Angle updates
    update[StateVecIndices.ROLL:StateVecIndices.YAW +
           1] = body2euler([state.ang_rate_x, state.ang_rate_y, state.ang_rate_z], state)
    return update


def build_nonlin_sys(prop: AircraftProperties, wind: WindModel,
                     prop_updater: PropertyUpdater = None) -> NonlinearIOSystem:
    """
    Construct a nonlinear IO system from passed input

    :param prop: Property describing the aircraft
    :param wind: Wind vector
    :param prop_updater: Instance that returns a dictionary that is passed to the
        update_params method of AircraftProperty, which evolves according to the
        a discrete schedule as the simulation evolves.
    """
    inputs = ('elevator_deflection', 'aileron_deflection', 'rudder_deflection', 'throttle')
    states = ('x', 'y', 'z', 'roll', 'pitch', 'yaw', 'vx',
              'vy', 'vz', 'ang_rate_x', 'ang_rate_y', 'ang_rate_z')
    system = NonlinearIOSystem(
        dynamics_kinetmatics_update, inputs=inputs, states=states,
        params={'prop': prop, 'wind': wind, 'prop_updater': prop_updater}, outputs=states,
        name='dynamics_kinematics'
    )
    return system
