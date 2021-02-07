from fcat import (
    AircraftProperties, IcedSkywalkerX8Properties, ControlInput,
    FrictionlessBall, SimpleTestAircraftNoForces
)

__all__ = ('aircraft_property_from_dct', 'actuator_from_dct')

# Global constants
SKYWALKERX8 = 'skywalkerX8'
FRICTIONLESS_BALL = 'frictionless_ball'
AIRCRAFT_NO_FORCE = 'aircraft_no_force'
FLYING_WINGS = 'flying_wings'


def aircraft_property_from_dct(aircraft: dict) -> AircraftProperties:
    """
    Return an aircraft property class initialized with the passed dictionary
    """
    known_types = [SKYWALKERX8, FRICTIONLESS_BALL, AIRCRAFT_NO_FORCE]

    aircraft_type = aircraft.get('type', "")
    if aircraft_type not in known_types:
        raise ValueError(f"Aircraft type must be one of {known_types}")

    if aircraft_type == SKYWALKERX8:
        return IcedSkywalkerX8Properties(ControlInput(), aircraft.get('icing', 0))
    elif aircraft_type == FRICTIONLESS_BALL:
        return FrictionlessBall(ControlInput())
    elif aircraft_type == AIRCRAFT_NO_FORCE:
        return SimpleTestAircraftNoForces(ControlInput())
    raise ValueError("Unknown aircraft type")


def actuator_from_dct(actuator: dict):
    if actuator['type'] == FLYING_WINGS:
        return None
    return None
