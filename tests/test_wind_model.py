import numpy as np
from fcat import no_wind, ConstantWind


def test_constant_wind():
    wind_model = no_wind()
    assert np.allclose(wind_model.get(2.0), np.zeros(6))

    wind_model = ConstantWind(np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))
    assert np.allclose(wind_model.get(4.0), wind_model.wind)
