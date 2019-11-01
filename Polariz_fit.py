"""
 Created by alexandra at 24.07.19 10:18

 Fitting of polarization for h -> infinity
"""

from scipy.optimize import curve_fit
import numpy as np
import matplotlib.pyplot as plt


def func(x, a, b, c, d):
    """function for fitting"""
    return (d/x**4) #*np.exp(-a*x)  b*x**2 +


xx = np.array([5, 10, 15, 20, 25, 30, 35, 40, 45, 100, 200])
yy = np.array([3.8*10**(-3), 1.2*10**(-4), 2.2*10**(-5), 6.7*10**(-6),
              2.7*10**(-6), 1.3*10**(-6), 6.8*10**(-7), 4*10**(-7),
              2.5*10**(-7), 1*10**(-8), 6.3*10**(-10)])

popt, pcov = curve_fit(func, xx, yy)

plt.figure()
plt.yscale('log')
plt.plot(xx, yy, 'b.', label='data')
plt.plot(xx, func(xx, *popt), 'r-',
         label='fit: a=%5.3f, b=%5.3f, c=%5.3f, d=%5.3f' % tuple(popt))
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()
