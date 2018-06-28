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

# Import parameters for Hopf Hamiltonian from file params.py
import params

t = params.t
h = params.h

Nx = params.Nx
Ny = params.Ny
Nz = params.Nz

kx = np.linspace(0, 2*pi, Nx)
ky = np.linspace(0, 2*pi, Ny)
kz = np.linspace(0, 2*pi, Nz)

# Import eigenstates of Hopf Humiltonian
with open('Hopfeigen.pickle', 'rb') as f:
    [E, u] = pickle.load(f)

# Occupied states correspond to smaller eigenvalues
uOcc = u[:, :, :, :, 0]
Chern = np.zeros(Nz)

for nkz in range(0, Nz):
    kkz = kz[nkz]
    for nkx in range(0, Nx - 1):
        kkx = kx[nkx]
        for nky in range(0, Ny - 1):
            kky = ky[nky]
            U1 = np.dot(np.conj(uOcc[nkx, nky, nkz, :]), uOcc[nkx + 1, nky, nkz, :])
            U2 = np.dot(np.conj(uOcc[nkx + 1, nky, nkz, :]), uOcc[nkx + 1, nky + 1, nkz, :])
            U3 = np.dot(np.conj(uOcc[nkx + 1, nky + 1, nkz, :]), uOcc[nkx, nky + 1, nkz, :])
            U4 = np.dot(np.conj(uOcc[nkx, nky + 1, nkz, :]), uOcc[nkx, nky, nkz, :])
            Chern[nkz] = Chern[nkz] - (cmath.log(U1 * U2 * U3 * U4)).imag / (2*pi)**2
            
figy = plt.figure()
plt.plot(kz,Chern)
plt.show