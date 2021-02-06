from fcat import (FrictionlessBall, ControlInput, State, build_nonlin_sys, SimpleTestAircraftNoMoments,
    SimpleTestAircraftNoForces)
from control import input_output_response
import numpy as np
from fcat.utilities import body2inertial, inertial2body, wind2body, calc_airspeed
from fcat.simulation_constants import GRAVITY_CONST
from fcat.model_builder import dynamics_kinetmatics_update
from fcat import no_wind

def test_kinematics():
    control_input = ControlInput()
    prop = FrictionlessBall(control_input)
    system = build_nonlin_sys(prop, no_wind())
    t = np.linspace(0.0, 10, 500, endpoint=True)
    state = State()
    state.vx = 20.0
    state.vy = 1
    for i in range(3):
        state.roll = 0
        state.pitch = 0
        state.yaw = 0
        if i == 0:
            state.ang_rate_x = 0.157079633
            state.ang_rate_y = 0
            state.ang_rate_z = 0
        elif i == 1:
            state.ang_rate_x = 0
            state.ang_rate_y = 0.157079633
            state.ang_rate_z = 0
        elif i == 2:
            state.ang_rate_x = 0
            state.ang_rate_y = 0
            state.ang_rate_z = 0.157079633

        T, yout = input_output_response(system, t, U=0, X0=state.state)
        pos = np.array(yout[:3])
        eul_ang = np.array(yout[3:6])
        
        vel_inertial = np.array([np.zeros(len(t)), np.zeros(len(t)), np.zeros(len(t))])
        for j in range(len(vel_inertial[0])):
            state.roll = yout[3,j]
            state.pitch = yout[4,j]
            state.yaw = yout[5,j]
            vel = np.array([yout[6,j],yout[7,j],yout[8,j]])
            vel_inertial_elem = body2inertial(vel,state)
            vel_inertial[0,j] = vel_inertial_elem[0]
            vel_inertial[1,j] = vel_inertial_elem[1]
            vel_inertial[2,j] = vel_inertial_elem[2]

        vx_inertial_expect = 20*np.ones(len(t))
        vy_inertial_expect = 1*np.ones(len(t))
        vz_inertial_expect = GRAVITY_CONST*t
        x_expect = 20*t
        y_expect = 1*t
        z_expect = 0.5*GRAVITY_CONST*t**2
        roll_expect = state.ang_rate_x*t
        pitch_expect = state.ang_rate_y*t
        yaw_expect = state.ang_rate_z*t
        assert np.allclose(vx_inertial_expect, vel_inertial[0], atol = 7e-3)
        assert np.allclose(vy_inertial_expect, vel_inertial[1], atol = 5e-3)
        assert np.allclose(vz_inertial_expect, vel_inertial[2], atol = 8e-3)
        assert np.allclose(x_expect, pos[0], atol = 8e-2)
        assert np.allclose(y_expect, pos[1], atol = 5e-2)
        assert np.allclose(z_expect, pos[2], atol = 7e-2)
        assert np.allclose(roll_expect, eul_ang[0], atol = 1e-3)
        assert np.allclose(pitch_expect, eul_ang[1], atol = 1e-3)
        assert np.allclose(yaw_expect, eul_ang[2], atol = 1e-3)

def test_dynamics_forces():
    control_input = ControlInput()
    prop = SimpleTestAircraftNoMoments(control_input)
    t = 0
    for i in range (-50,101,50):
        control_input.throttle = 0.8
        control_input.elevator_deflection = i
        control_input.aileron_deflection = i
        control_input.rudder_deflection = i

        state = State()
        state.vx = 20.0
        state.vy = 1
        state.vz = 0
        params = {
            "prop": prop,
            "wind": no_wind()
        }
        update = dynamics_kinetmatics_update(t = t, x = state.state, u = control_input.control_input, params = params)
        V_a = np.sqrt(np.sum(calc_airspeed(state, params['wind'].get(0.0))**2))

        forces_aero_wind_frame = np.array([-np.abs(control_input.elevator_deflection), control_input.aileron_deflection, -control_input.rudder_deflection])
        forces_aero_body_frame = wind2body(forces_aero_wind_frame, state, params['wind'].get(0))
        force_propulsion = np.array([(2*control_input.throttle)**2 - V_a**2, 0, 0])
        force_gravity = inertial2body(np.array([0, 0, prop.mass()*GRAVITY_CONST]), state)
        forces_body = forces_aero_body_frame + force_propulsion + force_gravity
        vx_update_expect = (1/prop.mass())*forces_body[0]
        vy_update_expect = (1/prop.mass())*forces_body[1]
        vz_update_expect = (1/prop.mass())*forces_body[2]
        # No moments
        ang_rate_x_update_expect = 0
        ang_rate_y_update_expect = 0
        ang_rate_z_update_expect = 0

        assert np.allclose(vx_update_expect, update[6])
        assert np.allclose(vy_update_expect, update[7])
        assert np.allclose(vz_update_expect, update[8])    
        assert np.allclose(ang_rate_x_update_expect, update[9])
        assert np.allclose(ang_rate_y_update_expect, update[10])
        assert np.allclose(ang_rate_z_update_expect, update[11])


def test_dynamics_moments():
    control_input = ControlInput()
    prop = SimpleTestAircraftNoForces(control_input)
    t = 0
    for i in range (-50,51,50):
        control_input.throttle = 0.0
        control_input.elevator_deflection = i
        control_input.aileron_deflection = i
        control_input.rudder_deflection = i

        state = State()
        state.vx = 20.0
        state.vy = 1
        state.vz = 0
        state.ang_rate_x = 0.157079633
        state.ang_rate_y = 0.157079633
        state.ang_rate_z = 0.157079633
        params = {
            "prop": prop,
            "wind": no_wind()
        }
        update = dynamics_kinetmatics_update(t = t, x = state.state, u = control_input.control_input, params = params)
        moments_aero = np.array([control_input.elevator_deflection, control_input.aileron_deflection, control_input.rudder_deflection])
        omega = np.array([state.ang_rate_x, state.ang_rate_y, state.ang_rate_z])
        coreolis_term = prop.inv_inertia_matrix().dot(np.cross(omega, prop.inertia_matrix().dot(omega)))

        ang_rate_x_update_expect = (2/3)*moments_aero[0] - (1/3)*moments_aero[2] - coreolis_term[0]
        ang_rate_y_update_expect = (1/2)*moments_aero[1] - coreolis_term[1]
        ang_rate_z_update_expect = (2/3)*moments_aero[2] - (1/3)*moments_aero[0] - coreolis_term[2]
        assert np.allclose(ang_rate_x_update_expect, update[9])
        assert np.allclose(ang_rate_y_update_expect, update[10])
        assert np.allclose(ang_rate_z_update_expect, update[11])