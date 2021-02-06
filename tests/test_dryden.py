import pytest
from fcat import DrydenGust
import numpy as np
from scipy.stats import linregress


def test_power_spectra():
    show_plot = False

    # NOTE: These parameters are very large, but by using large values the time constants are larger
    # and the large frequency behavior becomes visible at lower frequency. This is beneficial in this
    # test as we check the asymptotic behavior of the power spectrum
    turbulence = DrydenGust(200.1, 250.0)

    turbulence.seed(42)
    N = 256
    dt = 0.1
    wind_data = np.array([turbulence.get((i+1)*dt) for i in range(N)])
    assert all(x == y for x, y in zip((N, 6), wind_data.shape))

    power_spectrum = np.abs(np.fft.fft(wind_data, axis=0))**2
    power_spectrum = np.fft.fftshift(power_spectrum, axes=0)
    freq = np.fft.fftshift(np.fft.fftfreq(N))
    start = int(N/2+1)

    for i in range(6):
        slope, interscept, _, _, _ = linregress(np.log(freq[start:]), np.log(power_spectrum[start:, i]))

        # NOTE: We allow some discrepancy (absolute error of 0.4). There can be several reasons why
        # this high threshold is nessecary. Most likely it is caused by the fact that we are not
        # fitting to data exclusively in the asymptotic region. Thus, this test merely serves as 
        # a "qualitative" verification the the power spectrum of the signal generated is decaying
        # a power law with an exponent close to -2
        assert slope == pytest.approx(-2.0, abs=0.4)
        if show_plot:
            from matplotlib import pyplot as plt
            plt.figure(i)
            plt.plot(freq[start:], power_spectrum[start:, i])
            plt.plot(freq[start:], np.exp(interscept)*freq[start:]**slope)
            plt.yscale('log')
            plt.xscale('log')
    
    if show_plot:
        plt.show()
    