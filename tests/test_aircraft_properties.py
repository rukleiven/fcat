import pytest
import numpy as np
from fcat import State
from fcat import ControlInput
from fcat import IcedSkywalkerX8Properties
from fcat.skywalker8 import SkywalkerX8Constants
from fcat.utilities import calc_airspeed


def drag_coeff_test_cases():
    state1 = State()
    state1.vx = 20.0
    state1.vz = 0.0
    wind1 = np.zeros(6)
    wind2 = np.zeros(6)
    wind2[0] = 1.0
    wind2[2] = -19.0*np.tan(6*np.pi/180.0)
    zero_input = ControlInput()

    elevator_input = ControlInput()
    elevator_input.elevator_deflection = 2.0*np.pi/180.0

    return [(state1, wind1, 0.0, zero_input, 0.015039166436721),
            (state1, wind1, 1.0, zero_input, 0.043224285541117),
            (state1, wind2, 0.0, zero_input, 0.027939336576642),
            (state1, wind2, 1.0, zero_input, 0.082879203470842),
            (state1, wind1, 0.0, elevator_input, 0.015039166436721 +
             0.0633*elevator_input.elevator_deflection),
            (state1, wind1, 1.0, elevator_input, 0.043224285541117 + 0.0633*elevator_input.elevator_deflection)]


@pytest.mark.parametrize('state, wind, icing, control_input, expect', drag_coeff_test_cases())
def test_drag_coeff(state, wind, icing, control_input, expect):
    prop = IcedSkywalkerX8Properties(control_input, icing=icing)
    assert prop.drag_coeff(state, wind) == pytest.approx(expect)


def lift_coeff_test_cases():
    constants = SkywalkerX8Constants()
    c = constants.mean_chord
    state1 = State()
    state1.vx = 20.0
    state1.vz = 0.0
    state2 = State()
    state2.vx = 20.0
    state2.vz = 0.0
    state2.ang_rate_y = 5*np.pi/180
    wind1 = np.zeros(6)
    wind2 = np.zeros(6)
    wind3 = np.zeros(6)
    wind2[0] = 1.0
    wind2[2] = -19.0*np.tan(8*np.pi/180.0)
    wind3[0] = 1.0
    wind3[2] = -19.0*np.tan(8*np.pi/180.0)
    wind3[4] = 3*np.pi/180
    ang_rate_y2 = state2.ang_rate_y-wind3[4]
    airspeed2 = np.sqrt(np.sum(calc_airspeed(state2, wind2)**2))
    zero_input = ControlInput()
    elevator_input = ControlInput()
    elevator_input.elevator_deflection = 2.0*np.pi/180.0

    return [(state1, wind1, 0.0, zero_input, 0.030075562375465),
            (state1, wind1, 1.0, zero_input, 0.018798581619545),
            (state1, wind2, 0.0, zero_input, 0.609296679062686),
            (state1, wind2, 1.0, zero_input, 0.454153721254944),
            (state1, wind1, 0.0, elevator_input, 0.030075562375465 +
             0.278*elevator_input.elevator_deflection),
            (state1, wind1, 1.0, elevator_input, 0.018798581619545 +
             0.278*elevator_input.elevator_deflection),
            (state2, wind3, 0.0, elevator_input, 0.609296679062686 + 4.60 * c /
             (2*airspeed2)*ang_rate_y2 + 0.278*elevator_input.elevator_deflection),
            (state2, wind3, 1.0, elevator_input, 0.454153721254944 - 3.51 * c/(2*airspeed2)*ang_rate_y2 + 0.278*elevator_input.elevator_deflection)]


@pytest.mark.parametrize('state, wind, icing, control_input, expect', lift_coeff_test_cases())
def test_lift_coeff(state, wind, icing, control_input, expect):
    prop = IcedSkywalkerX8Properties(control_input, icing=icing)
    assert prop.lift_coeff(state, wind) == pytest.approx(expect)


def side_force_coeff_test_cases():
    constants = SkywalkerX8Constants()
    b = constants.wing_span
    state1 = State()
    state1.vx = 20.0
    state1.vy = 0.0
    state1.vz = 0.0
    state2 = State()
    state2.vx = 28.6362532829
    state2.vy = 1.0
    state2.vz = 0.0
    state2.ang_rate_x = 5*np.pi/180
    state2.ang_rate_z = 5*np.pi/180
    wind = np.zeros(6)
    airspeed = np.sqrt(np.sum(calc_airspeed(state2, wind)**2))
    zero_input = ControlInput()
    aileron_input = ControlInput()
    aileron_input.aileron_deflection = 2.0*np.pi/180.0

    return [(state1, wind, 0.0, zero_input, 2.99968641720902E-08),
            (state1, wind, 1.0, zero_input, -7.94977329537982E-06),
            (state2, wind, 0.0, aileron_input, -0.008604744865183 + -0.085 * b/(2*airspeed)*state2.ang_rate_x +
             0.005 * b/(2*airspeed)*state2.ang_rate_z + 0.0433*aileron_input.aileron_deflection),
            (state2, wind, 1.0, aileron_input, -0.007089388672593 + -0.133 * b/(2*airspeed)*state2.ang_rate_x + 0.002 * b/(2*airspeed)*state2.ang_rate_z + 0.0433*aileron_input.aileron_deflection)]


@pytest.mark.parametrize('state, wind, icing, control_input, expect', side_force_coeff_test_cases())
def test_side_force_coeff(state, wind, icing, control_input, expect):
    prop = IcedSkywalkerX8Properties(control_input, icing=icing)
    assert prop.side_force_coeff(state, wind) == pytest.approx(expect)


def roll_moment_coeff_test_cases():
    constants = SkywalkerX8Constants()
    b = constants.wing_span
    state1 = State()
    state1.vx = 20.0
    state1.vy = 0.0
    state1.vz = 0.0
    state2 = State()
    state2.vx = 28.6362532829
    state2.vy = 1.0
    state2.vz = 0.0
    state2.ang_rate_x = 5*np.pi/180
    state2.ang_rate_z = 5*np.pi/180
    wind = np.zeros(6)
    airspeed = np.sqrt(np.sum(calc_airspeed(state2, wind)**2))
    zero_input = ControlInput()
    aileron_input = ControlInput()
    aileron_input.aileron_deflection = 2.0*np.pi/180.0

    return [(state1, wind, 0.0, zero_input, -8.40821757613653E-05),
            (state1, wind, 1.0, zero_input, -7.34515369827804E-05),
            (state2, wind, 0.0, aileron_input, -0.00380800071177 + -0.409 * b/(2*airspeed) *
             state2.ang_rate_x + 0.039 * b/(2*airspeed)*state2.ang_rate_z + 0.12*aileron_input.aileron_deflection),
            (state2, wind, 1.0, aileron_input, -0.003067251004494 + -0.407 * b/(2*airspeed)*state2.ang_rate_x + 0.158 * b/(2*airspeed)*state2.ang_rate_z + 0.12*aileron_input.aileron_deflection)]


@pytest.mark.parametrize('state, wind, icing, control_input, expect', roll_moment_coeff_test_cases())
def test_roll_moment_coeff(state, wind, icing, control_input, expect):
    prop = IcedSkywalkerX8Properties(control_input, icing=icing)
    assert prop.roll_moment_coeff(state, wind) == pytest.approx(expect)


def pitch_moment_coeff_test_cases():
    state1 = State()
    state1.vx = 20.0
    state1.vz = 0.0
    wind1 = np.zeros(6)
    wind2 = np.zeros(6)
    wind2[0] = 1.0
    wind2[2] = -19.0*np.tan(6*np.pi/180.0)
    zero_input = ControlInput()

    elevator_input = ControlInput()
    elevator_input.elevator_deflection = 2.0*np.pi/180.0

    return [(state1, wind1, 0.0, zero_input, 0.001161535578433),
            (state1, wind1, 1.0, zero_input, -0.004297270367523),
            (state1, wind2, 0.0, zero_input, -0.064477317568596),
            (state1, wind2, 1.0, zero_input, -0.01808304889307),
            (state1, wind1, 0.0, elevator_input, 0.001161535578433 -
             0.206*elevator_input.elevator_deflection),
            (state1, wind1, 1.0, elevator_input, -0.004297270367523 - 0.206*elevator_input.elevator_deflection)]


@pytest.mark.parametrize('state, wind, icing, control_input, expect', pitch_moment_coeff_test_cases())
def ptich_moment_coeff(state, wind, icing, control_input, expect):
    prop = IcedSkywalkerX8Properties(control_input, icing=icing)
    assert prop.pitch_moment_coeff(state, wind) == pytest.approx(expect)


def yaw_moment_coeff_test_cases():
    constants = SkywalkerX8Constants()
    b = constants.wing_span
    state1 = State()
    state1.vx = 20.0
    state1.vy = 0.0
    state1.vz = 0.0
    state2 = State()
    state2.vx = 28.6362532829
    state2.vy = 1.0
    state2.vz = 0.0
    state2.ang_rate_x = 5*np.pi/180
    state2.ang_rate_z = 5*np.pi/180
    wind = np.zeros(6)
    airspeed = np.sqrt(np.sum(calc_airspeed(state2, wind)**2))
    zero_input = ControlInput()
    aileron_input = ControlInput()
    aileron_input.aileron_deflection = 2.0*np.pi/180.0

    return [(state1, wind, 0.0, zero_input, 4.9176697574439E-06),
            (state1, wind, 1.0, zero_input, 1.96093394589053E-05),
            (state2, wind, 0.0, aileron_input, 0.000825947539055 + 0.027 * b/(2*airspeed)*state2.ang_rate_x + -
             0.022 * b/(2*airspeed)*state2.ang_rate_z - 0.00339*aileron_input.aileron_deflection),
            (state2, wind, 1.0, aileron_input, 0.001052911121301 + 0.017 * b/(2*airspeed)*state2.ang_rate_x + -0.049 * b/(2*airspeed)*state2.ang_rate_z - 0.00339*aileron_input.aileron_deflection)]


@pytest.mark.parametrize('state, wind, icing, control_input, expect', yaw_moment_coeff_test_cases())
def test_yaw_moment_coeff(state, wind, icing, control_input, expect):
    prop = IcedSkywalkerX8Properties(control_input, icing=icing)
    assert prop.yaw_moment_coeff(state, wind) == pytest.approx(expect)
