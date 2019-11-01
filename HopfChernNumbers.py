# -*- coding: utf-8 -*-
"""
Created on Thu Jun 28 13:34:39 2018

@author: aleksandra
"""

# Calculate Chern numbers for Hopf insulator
# Calculate for each kz 2D Chern number in x,y plane

import numpy as np
from math import pi
import pickle
import math
import cmath
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Import parameters for Hopf Hamiltonian from file params.py
import params

t = params.t
h = params.h

# Nx = params.Nx
# Ny = params.Ny
# Nz = params.Nz
Nx = 100
Ny = 100
Nz = 100

kx = np.linspace(0, 2*pi, Nx)
ky = np.linspace(0, 2*pi, Ny)
kz = np.linspace(0, 2*pi, Nz)

# Import eigenstates of Hopf Humiltonian
with open('Berryeigen.pickle', 'rb') as f:
    [E, u] = pickle.load(f)

# Occupied states correspond to smaller eigenvalues
uOcc = u[:, :, :, :, 0]
Chern_x = np.zeros(Nx)
Chern_y = np.zeros(Ny)
Chern_z = np.zeros(Nz)

# Check gap in the spectrum
flag = 0  # No gap closure
for nkz in range(0, Nz - 1):
    for nkx in range(0, Nx - 1):
        for nky in range(0, Ny - 1):
            if np.abs(E[nkx, nky, nkz, 1] - E[nkx, nky, nkz, 0]) < 0.1:
                flag = flag + 1  # Count number of closure points
                print(nkx)
                print(nky)
                print(nkz)
print(flag)

for nkz in range(0, Nz - 1):
    kkz = kz[nkz]
    for nkx in range(0, Nx - 1):
        kkx = kx[nkx]
        for nky in range(0, Ny - 1):
            kky = ky[nky]
            Ux = np.dot(
                np.conj(uOcc[nkx, nky, nkz, :]), uOcc[nkx + 1, nky, nkz, :])
            Uy = np.dot(
                np.conj(uOcc[nkx, nky, nkz, :]), uOcc[nkx, nky + 1, nkz, :])
            Uz = np.dot(
                np.conj(uOcc[nkx, nky, nkz, :]), uOcc[nkx, nky, nkz + 1, :])
            Uxy = np.dot(
                np.conj(uOcc[nkx + 1, nky, nkz, :]),
                uOcc[nkx + 1, nky + 1, nkz, :])
            Uxz = np.dot(
                np.conj(uOcc[nkx + 1, nky, nkz, :]),
                uOcc[nkx + 1, nky, nkz + 1, :])
            Uyx = np.dot(
                np.conj(uOcc[nkx, nky + 1, nkz, :]),
                uOcc[nkx + 1, nky + 1, nkz, :])
            Uyz = np.dot(
                np.conj(uOcc[nkx, nky + 1, nkz, :]),
                uOcc[nkx, nky + 1, nkz + 1, :])
            Uzx = np.dot(
                np.conj(uOcc[nkx, nky, nkz + 1, :]),
                uOcc[nkx + 1, nky, nkz + 1, :])
            Uzy = np.dot(
                np.conj(uOcc[nkx, nky, nkz + 1, :]),
                uOcc[nkx, nky + 1, nkz + 1, :])

            Chern_x[nkx] = (Chern_z[nkx]
                            - (cmath.log(Uy * Uyz * Uzy.conjugate() * Uz.conjugate())).imag
                            / (2 * pi))
            Chern_y[nky] = (Chern_z[nky]
                            - (cmath.log(Uz * Uzx * Uxz.conjugate() * Ux.conjugate())).imag
                            / (2 * pi))
            Chern_z[nkz] = (Chern_z[nkz]
                            - (cmath.log(Ux * Uxy * Uyx.conjugate() * Uy.conjugate())).imag
                            / (2*pi))
            # Chern_pi = Chern_pi - (cmath.log(U1 * U2 * U3 * U4)).imag / (
            #             2 * pi)
            
figy, ax = plt.subplots(1, 3)
ax[0].plot(kx, Chern_x)
ax[1].plot(ky, Chern_y)
ax[2].plot(kz, Chern_z)
plt.show()

# print(Chern_pi)
