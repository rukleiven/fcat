from enum import IntEnum

PITCH_HINF_CONTOLLER = 'pitch_hinf_controller'
PITCH_GS_CONTROLLER = 'pitch_gs_controller'
PITCH_ERROR = 'pitch_error'
ICING = 'icing'
ELEVATOR_DEFLECTION_COMMAND = 'elevator_deflection_command'

ROLL_GS_CONTROLLER = "roll_gs_controller"
ROLL_HINF_CONTROLLER = 'roll_hinf_controller'
ROLL_ERROR = 'roll_error'
AILERON_DEFLECTION_COMMAND = 'aileron_deflection_command'

AIRSPEED_ERROR = 'airspeed_error'
SWITCH_VALUE_GS = 'switch_value_gs'

ROLL_COMMAND = 'roll_command'
PITCH_COMMAND = 'pitch_command'
AIRSPEED_COMMAND = 'airspeed_command'


class Direction(IntEnum):
    LONGITUDINAL = 0
    LATERAL = 1
