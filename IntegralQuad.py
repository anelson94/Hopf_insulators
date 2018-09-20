# -*- coding: utf-8 -*-
"""
Created on Wed Sep 12 12:21:59 2018

@author: aleksandra
"""

# Integrate the functions

import scipy.integrate
import numpy
from math import pi

def integrand1(x, t):
    return 4 * (1 - t**2) * numpy.sqrt((1 - x**2)/((1 - t**2) * x**2 + t**2))

t=2
I = scipy.integrate.quad(integrand1, 0, 1, args = (t))
print(I)
