import pytest
from fcat import PropUpdate, PropertyUpdater


def test_icing_updates():
    updates = {
        'icing': [PropUpdate(time=0.0, value=1.0),
                  PropUpdate(time=0.1, value=0.2),
                  PropUpdate(time=0.5, value=1.0)]
    }

    updater = PropertyUpdater(updates)

    # List cases (time, expected value)
    cases = [(-1.0, 1.0), (0.0, 1.0), (0.09, 1.0), (0.3, 0.2),
             (0.46, 0.2), (0.6, 1.0), (10.0, 1.0)]
    for time, expect in cases:
        dct = updater.get_param_dict(time)
        assert dct['icing'] == pytest.approx(expect)


def test_two_keys():
    updates = {
        'icing': [PropUpdate(time=0.0, value=1.0),
                  PropUpdate(time=0.1, value=0.2),
                  PropUpdate(time=0.5, value=1.0)],
        'fracture': [PropUpdate(time=0.0, value=0.0),
                     PropUpdate(time=1.0, value=1.0)]
    }

    updater = PropertyUpdater(updates)

    # List cases (time, cing value, fracture value)
    cases = [(-1.0, 1.0, 0.0), (0.0, 1.0, 0.0), (0.09, 1.0, 0.0), (0.3, 0.2, 0.0),
             (0.46, 0.2, 0.0), (0.6, 1.0, 0.0), (10.0, 1.0, 1.0)]

    for time, icing, fracture in cases:
        dct = updater.get_param_dict(time)
        assert dct['icing'] == pytest.approx(icing)
        assert dct['fracture'] == pytest.approx(fracture)
