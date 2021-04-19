import pytest
from fcat import DrydenGust
import numpy as np
from scipy.stats import linregress


def test_power_spectra():
    show_plot = False

    # NOTE: These parameters are very large, but by using large values the time constants are larger
    # and the large frequency behavior becomes visible at lower frequency. This is beneficial in this
    # test as we check the asymptotic behavior of the power spectrum
    N = 2**16
    t = np.linspace(0, 20, N)
    turbulence = DrydenGust(2.1, t, seed=42)

    wind_data = turbulence.wind_data
    assert wind_data.shape == (6, N)

    power_spectrum = np.abs(np.fft.fft(wind_data, axis=1))**2
    integral = wind_data.sum(axis=1)**2
    # Verify that 0-frequency is correct
    assert len(integral) == 6
    assert np.allclose(power_spectrum[:, 0], integral)

    freq = np.fft.fftfreq(N)
    dt = t[1]-t[0]
    freq = freq/dt
    start = (np.abs(freq - 100)).argmin()
    end = (np.abs(freq - 500)).argmin()
    freq = freq[start:end]
    power_spectrum = power_spectrum[:, start:end]
    for i in range(6):
        slope, interscept, _, _, _ = linregress(np.log(freq), np.log(power_spectrum[i, :]))
        print(slope)

        # NOTE: We allow some discrepancy (absolute error of 0.4). There can be several reasons why
        # this high threshold is nessecary. Most likely it is caused by the fact that we are not
        # fitting to data exclusively in the asymptotic region. Thus, this test merely serves as
        # a "qualitative" verification the the power spectrum of the signal generated is decaying
        # a power law with an exponent close to -2
        assert slope == pytest.approx(-2.0, abs=0.15)
        if show_plot:
            from matplotlib import pyplot as plt
            plt.figure(i)
            plt.plot(freq, power_spectrum[i, :])
            plt.plot(freq, np.exp(interscept)*freq**slope)
            plt.yscale('log')
            plt.xscale('log')

    if show_plot:
        plt.show()
