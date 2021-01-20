import pytest
from fcat import State
from fcat import IcedSkywalkerX8Properties

@pytest.mark.parametrize('state, expect', [(State(), 1.0),
                                           (State(), 1.0)])
def test_drag_force(state, expect):
    prop = IcedSkywalkerX8Properties([0.0, 0.0, 0.0])
    assert prop.drag_force(state) == pytest.approx(expect)