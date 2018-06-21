# -*- coding: utf-8 -*-
"""
Created on Thu Jun 21 15:59:07 2018

@author: Aleksandra
"""

# The Hamiltonian of Hopf insulator is defined on a lattice (kx, ky, kz),
# kx in (0, 2pi), ky in (0, 2pi), kz in (0, 2pi).

import numpy as np
from math import sin, cos, pi
#import math

kx = np.linspace(0, 2*pi, 11)
ky = 0 #np.arange(0, 2*pi, 2*pi/10)
kz = 0 #np.arange(0, 2*pi, 2*pi/10)

sigmax = np.array([[0, 1], [1, 0]])
sigmay = np.array([[0, -1j], [1j, 0]])
sigmaz = np.array([[1, 0], [0, -1]])

sigmax = sigmax[np.newaxis, :, :]
sigmaxstack = np.tile(sigmax, (11,1,1))

sigmay = sigmay[np.newaxis, :, :]
sigmaystack = np.tile(sigmay, (11,1,1))

sigmaz = sigmaz[np.newaxis, :, :]
sigmazstack = np.tile(sigmaz, (11,1,1))

h = 1
t = 1

# Hopf Hamiltonian is a mapping function from T^3 to S^2.
# It has two energy states, one of them occupied.

lamb = np.divide(1, np.power(np.sin(kx), 2) + sin(ky)**2 + sin(kz)**2 +
                 np.power(np.cos(kx) + cos(ky) + cos(kz) + h, 2))

Hx = np.multiply(2 * lamb, np.sin(kx)*sin(ky) + 
                 t*sin(ky)*(np.cos(kx) + cos(ky) + cos(kz) + h))
Hy = np.multiply(2 * lamb, t*sin(ky)*sin(kz) -
                 np.multiply(np.sin(kx), np.cos(kx) + cos(ky) + cos(kz) + h))
Hz = np.multiply(lamb, np.power(np.sin(kx), 2) + t**2 * sin(ky)**2 - sin(kz)**2 - 
                 np.power(np.cos(kx) + cos(ky) + cos(kz) + h, 2))

Hx = Hx[:, np.newaxis, np.newaxis]
Hy = Hy[:, np.newaxis, np.newaxis]
Hz = Hz[:, np.newaxis, np.newaxis]


HopfH = (np.multiply(Hx, sigmaxstack) + np.multiply(Hy, sigmaystack) + 
         np.multiply(Hz, sigmazstack))

[E,u] = np.linalg.eigh(HopfH[0,:,:])

print(E)
print(u[:,0])
print(u[:,1])
