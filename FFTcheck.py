"""
 Created by alexandra at 15.11.18 15:36
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


def funct(k):
    return np.divide(np.sin(k),
                     np.sqrt(np.power(np.sin(k), 2) +
                             np.power(np.cos(k) + 2, 2)))


def fitfunc(x, alpha, beta, gamma):  # , epsilon
    """Function for fitting"""
    return np.log(np.exp(-beta * x) * np.power(x, alpha) * gamma)


kk = np.linspace(0, 2*np.pi, 1000)
g = np.fft.fft(funct(kk))

xx = range(1000)

xcoo = xx[:12]
ycoo = np.log(g[:12])
popt, pcov = curve_fit(fitfunc, xcoo, ycoo)
xplot = np.linspace(0.01, 15, 100)

plt.figure()
plt.yscale('log')
plt.plot(xx[:20], g[:20], '.')
plt.plot(xplot, np.exp(fitfunc(xplot, *popt)), 'r-')
plt.show()
