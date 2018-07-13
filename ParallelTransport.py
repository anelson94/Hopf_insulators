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

t = params.t
h = params.h

Nx = params.Nx
Ny = params.Ny
Nz = params.Nz 

kx = np.linspace(0, 2*pi, Nx)
ky = np.linspace(0, 2*pi, Ny)
kz = np.linspace(0, 2*pi, Nz)

def scalarprod(A, B):
    # Scalar product of two stackes of wavefunctions of the same size
    # Returns a stack of <A[i,j,...,:]| B[i,j,...,:]>
    Prod = np.sum(np.multiply(np.conj(A), B), axis = -1)
    return Prod

# Import eigenstates of Hopf Humiltonian
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
    
Lamb = scalarprod(usmooth[0, 0, 0, :], usmooth[Nx - 1, 0, 0, :])

nxs = np.linspace(0, Nx-1, Nx)
usmooth = np.multiply(usmooth, 
                      np.power(Lamb, 
                      - nxs[:, np.newaxis, np.newaxis, np.newaxis] / (Nx - 1)))

# For all kx make parallel transport along ky
for nky in range(0, Ny - 1):
    Mold = scalarprod(usmooth[:, nky, 0, :], uOcc[:, nky + 1, 0, :])
    usmooth[:, nky + 1, 0, :] = np.multiply(uOcc[:, nky + 1, 0, :], 
           np.exp( -1j * np.angle(Mold[:, np.newaxis])))

Lamb2 = scalarprod(usmooth[:, 0, 0, :], usmooth[:, Ny - 1, 0, :])

nys = np.linspace(0, Ny-1, Ny)
usmooth = np.multiply(usmooth, 
                      np.power(Lamb2[:, np.newaxis, np.newaxis, np.newaxis], 
                      - nys[np.newaxis, :, np.newaxis, np.newaxis] / (Ny - 1)))

# For all kx, ky make parallel transport along kz
for nkz in range(0, Nz - 1):
    Mold = scalarprod(usmooth[:, :, nkz, :], uOcc[:, :, nkz + 1, :])
    usmooth[:, :, nkz + 1, :] = np.multiply(uOcc[:, :, nkz + 1, :],
           np.exp( -1j * np.angle(Mold[:, :, np.newaxis])))
    
Lamb3 = scalarprod(usmooth[:, :, 0, :], usmooth[:, :, Nz - 1, :])

nzs = np.linspace(0, Nz-1, Nz)
usmooth = np.multiply(usmooth, 
                      np.power(Lamb3[:, :, np.newaxis, np.newaxis], 
                      - nzs[np.newaxis, np.newaxis, :, np.newaxis] / (Nz - 1)))
mult = np.power(Lamb3[:, :, np.newaxis, np.newaxis], 
                      - nzs[np.newaxis, np.newaxis, :, np.newaxis] / (Nz - 1))

#plt.imshow(mult[4,:,:,0].real)
#plt.colorbar
#plt.show

plt.imshow(np.imag(mult[:,60,:,0]))
plt.colorbar()
plt.show()
plt.imshow(np.imag(Lamb3))
plt.show()
print(np.max(np.angle(mult[:,60,:,0])))
#figy = plt.figure()
#plt.plot(np.linspace(0, Nx - 1 ,Nx),usmooth[:,6,7,0].imag)
#plt.show
#figy = plt.figure()
#plt.plot(np.linspace(0, Nx - 1 ,Nx),usmooth[:,6,7,0].real)
#plt.show
#figy = plt.figure()
#plt.plot(np.linspace(0, Nx - 1, Nx),usmooth[:,6,7,1].imag)
#plt.show
#figy = plt.figure()
#plt.plot(np.linspace(0, Nx - 1, Nx),usmooth[:,6,7,1].real)
#plt.show
with open('Hopfsmoothstates.pickle', 'wb') as f:
    pickle.dump(usmooth, f)
#print(usmooth[13,6,100,0] - usmooth[13,6,0,0])
#print(usmooth[13,6,79,0])
