# -*- coding: utf-8 -*-
"""
Created on Fri Jun 22 14:05:25 2018

@author: Aleksandra
"""
# Calculate well localized Hybrid Wannier functions for occupied states of 
# Hopf Hamiltonian for all values of ky, kz

import numpy as np
from math import pi
import pickle
import math
import cmath

# Import parameters for Hopf Hamiltonian from file params.py
import params

t = params.t
h = params.h

Nx = params.Nx
Ny = params.Ny
Nz = params.Nz

# Import eigenstates of Hopf Humiltonian
with open('Hopfeigen.pickle', 'rb') as f:
    [E, u] = pickle.load(f)

# Occupied states correspond to smaller eigenvalues
uOcc = u[:, :, :, :, 0]
usmooth = np.empty([Nx, Ny, Nz, 2], dtype = complex)
usmooth[0, :, :, :] = uOcc[0, :, :, :]
Mprod = 1

for nkx in range(0, Nx - 1):
    Mold = np.sum(np.conj(usmooth[nkx, :, :, :]) * 
                  uOcc[nkx + 1, :, :, :], axis = -1)
    usmooth[nkx + 1, :, :, :] = (
            uOcc[nkx + 1, :, :, :] * 
            np.power(math.e, -1j * np.angle(Mold[:, :, np.newaxis]))
            )
    Mprod = np.multiply(Mprod, abs(Mold))
    
Lamb = np.sum(np.conj(usmooth[0, :, :, :]) * 
              usmooth[Nx - 1, :, :, :], axis = -1)

xAverage = -1/Nx*np.log(np.divide(Mprod,Lamb)).imag

#print(xAverage[2, :])
#print(xAverage[:, 2])

with open('HybridWannierCenters.pickle', 'wb') as f:
    pickle.dump(xAverage, f)