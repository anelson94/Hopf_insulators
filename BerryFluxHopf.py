# -*- coding: utf-8 -*-
"""
Created on Wed Jul 11 17:14:52 2018

@author: aleksandra
"""

# Calculate Berry flux (curvature) as a function of (kx,ky)
# As always deal with Hopf Hamiltonian

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
TrF = np.zeros((Nx - 1, Ny - 1))

# Calculate Berry flux in z direction as a function of kx, ky
nkz = 50 # kz=0
for nkx in range(0, Nx - 1):
    kkx = kx[nkx]
    for nky in range(0, Ny - 1):
        kky = ky[nky]
        U1 = np.dot(np.conj(uOcc[nkx, nky, nkz, :]), uOcc[nkx + 1, nky, nkz, :])
        U2 = np.dot(np.conj(uOcc[nkx + 1, nky, nkz, :]), uOcc[nkx + 1, nky + 1, nkz, :])
        U3 = np.dot(np.conj(uOcc[nkx + 1, nky + 1, nkz, :]), uOcc[nkx, nky + 1, nkz, :])
        U4 = np.dot(np.conj(uOcc[nkx, nky + 1, nkz, :]), uOcc[nkx, nky, nkz, :])
        TrF[nkx, nky] = - (cmath.log(U1 * U2 * U3 * U4)).imag * Nx * Ny / (2*pi)**2
        
plt.imshow(TrF, cmap='RdBu')
plt.colorbar()
plt.show()