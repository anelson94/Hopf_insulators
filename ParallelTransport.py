# -*- coding: utf-8 -*-
"""
Created on Fri Jul 13 16:37:14 2018

@author: aleksandra
"""

# Parallel transport procedure for Hopf eigenvectors
# To obtain a smooth solution make parallel transport 
# along kx, ky, kz directions

import numpy as np
from math import pi
import pickle
import matplotlib.pyplot as plt

# Import parameters for Hopf Hamiltonian from file params.py
import params

# t = params.t
# h = params.h
#
# Nx = params.Nx
# Ny = params.Ny
# Nz = params.Nz

Nx = 100
Ny = 100
Nz = 100

kx = np.linspace(0, 2*pi, Nx)
ky = np.linspace(0, 2*pi, Ny)
kz = np.linspace(0, 2*pi, Nz)


def scalarprod(A, B):
    # Scalar product of two stackes of wavefunctions of the same size
    # Returns a stack of <A[i,j,...,:]| B[i,j,...,:]>
    Prod = np.sum(np.multiply(np.conj(A), B), axis = -1)
    return Prod

# Import eigenstates of Hopf Humiltonian
# with open('Hopfgeneigen.pickle', 'rb') as f:
with open('Hopfeigen.pickle', 'rb') as f:
    [E, u] = pickle.load(f)

# Occupied states correspond to smaller eigenvalues
uOcc = u[:, :, :, :, 0]
usmooth = np.empty([Nx, Ny, Nz, 2], dtype = complex)

# Initial smooth function is equal to calculated one
usmooth[0, 0, 0, :] = uOcc[0, 0, 0, :]

# First of all make parallel transport in kx direction for ky=kz=0
for nkx in range(0, Nx - 1):
    Mold = scalarprod(usmooth[nkx, 0, 0, :], uOcc[nkx + 1, 0, 0, :])
    usmooth[nkx + 1, 0, 0, :] = np.multiply(
            uOcc[nkx + 1, 0, 0, :], np.exp( -1j * np.angle(Mold)))

# The function gains the multiplier    
Lamb = scalarprod(usmooth[0, 0, 0, :], usmooth[Nx - 1, 0, 0, :])

nxs = np.linspace(0, Nx-1, Nx)
# Distribute the multiplier among functions at kx in [0, 2pi]
usmooth = np.multiply(usmooth, 
                      np.power(Lamb, 
                      - nxs[:, np.newaxis, np.newaxis, np.newaxis] / (Nx - 1)))

# For all kx make parallel transport along ky
for nky in range(0, Ny - 1):
    Mold = scalarprod(usmooth[:, nky, 0, :], uOcc[:, nky + 1, 0, :])
    usmooth[:, nky + 1, 0, :] = np.multiply(uOcc[:, nky + 1, 0, :], 
                                  np.exp( -1j * np.angle(Mold[:, np.newaxis])))

# The function gains the multiplier
Lamb2 = scalarprod(usmooth[:, 0, 0, :], usmooth[:, Ny - 1, 0, :])

# Get the phase of lambda
Langle2 = np.angle(Lamb2)

# Construct smooth phase of lambda (without 2pi jumps)
for nkx in range(0, Nx - 1):
    if (np.abs(Langle2[nkx + 1] - Langle2[nkx]) > pi):
        Langle2[nkx + 1 : Nx] = (Langle2[nkx + 1 : Nx] - 
               np.sign(Langle2[nkx + 1] - Langle2[nkx]) * (2 * pi))

nys = np.linspace(0, Ny-1, Ny)
# Distribute the multiplier among functions at ky in [0, 2pi]
usmooth = np.multiply(usmooth, 
                      np.exp(1j * np.multiply(Langle2[:, np.newaxis, np.newaxis, np.newaxis], 
                      - nys[np.newaxis, :, np.newaxis, np.newaxis] / (Ny - 1))))

# For all kx, ky make parallel transport along kz
for nkz in range(0, Nz - 1):
    Mold = scalarprod(usmooth[:, :, nkz, :], uOcc[:, :, nkz + 1, :])
    usmooth[:, :, nkz + 1, :] = np.multiply(uOcc[:, :, nkz + 1, :],
           np.exp( -1j * np.angle(Mold[:, :, np.newaxis])))
    
Lamb3 = scalarprod(usmooth[:, :, 0, :], usmooth[:, :, Nz - 1, :])
Langle3 = np.angle(Lamb3)

# Langle3 = np.where(Langle3 < 0, Langle3 + 2 * pi, Langle3)

# First make the lambda phase smooth along x-axis
for nkx in range(0, Nx - 1):
    jump = (np.abs(Langle3[nkx + 1, :] - Langle3[nkx, :]) > pi * np.ones(Ny))
    Langlechange = np.multiply(jump, 
           np.sign(Langle3[nkx + 1, :] - Langle3[nkx, :]) * (2 * pi))
    Langle3[nkx + 1 : Nx, :] = (Langle3[nkx + 1 : Nx, :] - 
           Langlechange[np.newaxis, :])

# Then make the phase smooth along y-axis similar for all x        
for nky in range(0, Ny - 1):
    if np.abs(Langle3[0, nky + 1] - Langle3[0, nky]) > pi:
        Langle3[:, nky + 1 : Ny] = (Langle3[:, nky + 1 : Ny] - 
               np.sign(Langle3[0, nky + 1] - Langle3[0, nky]) * (2 * pi))


nzs = np.linspace(0, Nz-1, Nz)
# Distribute the multiplier among functions at kz in [0, 2pi]
usmooth = np.multiply(usmooth, 
                      np.exp(1j * np.multiply(Langle3[:, :, np.newaxis, np.newaxis], 
                      - nzs[np.newaxis, np.newaxis, :, np.newaxis] / (Nz - 1))))
mult = np.exp(1j * np.multiply(Langle3[:, :, np.newaxis, np.newaxis], 
                      - nzs[np.newaxis, np.newaxis, :, np.newaxis] / (Nz - 1)))


with open('Hopfsmoothstates.pickle', 'wb') as f:
    pickle.dump(usmooth, f)
print(usmooth[13,6,Nz-1,0] - usmooth[13,6,0,0])
#print(usmooth[13,6,79,0])

# plt.figure()
# plt.imshow(np.real(usmooth[:, 39, :, 1]))
# plt.show()
