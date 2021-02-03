from typing import NamedTuple, Dict, Sequence


__all__ = ('PropUpdate', 'PropertyUpdater')


class PropUpdate(NamedTuple):
    """
    Type that represents a single property update
    """
    time: float
    value: float


def _get_value_at_time(t: float, updates: Sequence[PropUpdate]) -> float:
    """
    Return the value of the item satisfying updates[i].time < t < updates[i+1].time

    :param t: Current time
    :param updates: Sequence with discrete updates
    """
    if t <= updates[0].time:
        return updates[0].value
    elif t >= updates[-1].time:
        return updates[-1].value

    for i in range(0, len(updates)-1):
        if updates[i+1].time > t:
            return updates[i].value
    return updates[-1].value


class PropertyUpdater:
    """
    Class for returning a property dictionary according to scheduled
    updates.

    :param updates: Dictionary containing property updates

    Example:

    If the AircraftProperty under consideration has a custom property called
    fracture, and we want this property to follow the following time evolution

    0 < t < 10: fracture = 0.0
    10 < t < 12: fracture = 0.5
    t > 12: fracture = 1.0

    updates dict passed to this class would be

    updates = {
        'fracture': [PropUpdate(time=0, value=0.0), PropUpdate(time=10.0, value=0.5),
                     PropUpdate(time=12, value=1.0)]
    }

    e.g. we give a list with the points where the fracture value change
    """
    def __init__(self, updates: Dict[str, Sequence[PropUpdate]]):
        self.updates = updates

    def get_param_dict(self, t: float) -> dict:
        """
        Return a parameter dictionary consistent with time t

        :param t: Time
        """
        return {k: _get_value_at_time(t, updates) for k, updates in self.updates.items()}
