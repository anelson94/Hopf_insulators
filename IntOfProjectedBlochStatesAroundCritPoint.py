# -*- coding: utf-8 -*-
"""
Created on Fri Sep  7 16:54:15 2018

@author: aleksandra

Calculate the Fourier integral over small vicinity around critical point of
Bloch states projected on the localized orbitals.
"""


import numpy as np
from math import pi, cos, sin
from scipy import integrate


def critfunction_t1_real(rho, kz):
    """Set the real part of the function in the vicinity
    of critical point at t=1"""
    return ((cos(z * kz) * sin(kz)
             - sin(z * kz) * (cos(kz) + 2 + h))
            * rho**2
            / np.sqrt(rho**2 + 2 * (2 + h) * cos(kz) + (2 + h)**2 + 1))


def critfunction_t1_imag(rho, kz):
    """Set the imaginary part of the function in the vicinity
    of critical point at t=1"""
    return ((sin(z * kz) * sin(kz)
             + cos(z * kz) * (cos(kz) + 2 + h))
            * rho**2
            / np.sqrt(rho**2 + 2 * (2 + h) * cos(kz) + (2 + h)**2 + 1))


def critfunction_real(rho, kz, phi):
    """Set the real part of the function in the vicinity
    of critical point at arbitrary t"""
    return np.real(
        np.exp(1j * (z * kz + x * rho * cos(phi) + y * rho * sin(phi)))
        * (cos(phi) + 1j * t * sin(phi))
        / np.sqrt(cos(phi)**2 + t**2 * sin(phi)**2)
        * (sin(kz) + 1j * (cos(kz) + 2 + h))
        / np.sqrt(rho**2 * (cos(phi)**2 + t**2 * sin(phi)**2)
                  + 2 * (2 + h) * cos(kz) + (2 + h)**2 + 1)
    )


def critfunction_imag(rho, kz, phi):
    """Set the imaginary part of the function in the vicinity
    of critical point at arbitrary t"""
    return np.imag(
        np.exp(1j * (z * kz + x * rho * cos(phi) + y * rho * sin(phi)))
        * (cos(phi) + 1j * t * sin(phi))
        / np.sqrt(cos(phi)**2 + t**2 * sin(phi)**2)
        * (sin(kz) + 1j * (cos(kz) + 2 + h))
        / np.sqrt(rho**2 * (cos(phi)**2 + t**2 * sin(phi)**2)
                  + 2 * (2 + h) * cos(kz) + (2 + h)**2 + 1)
    )


x = 0.71
y = 0.36
z = 0.25
h = 0
t = 3

# I_vicinity_im, I_error_im = integrate.dblquad(
#     critfunction_t1_imag, 0, 2 * pi, lambda kzpr: 0, lambda kzpr: 0.01)
# I_vicinity_re, I_error_re = integrate.dblquad(
#     critfunction_t1_real, 0, 2 * pi, lambda kzpr: 0, lambda kzpr: 0.01)

I_vicinity_t_im, I_t_error_im = integrate.tplquad(
    critfunction_imag, 0, 2 * pi,
    lambda phi: 0, lambda phi: 2 * pi,
    lambda phi, kzpr: 0, lambda phi, kzpr: 0.01)
I_vicinity_t_re, I_t_error_re = integrate.tplquad(
    critfunction_real, 0, 2 * pi,
    lambda phi: 0, lambda phi: 2 * pi,
    lambda phi, kzpr: 0, lambda phi, kzpr: 0.01)

print(I_vicinity_t_re + 1j * I_vicinity_t_im, I_t_error_re + 1j * I_t_error_im)
