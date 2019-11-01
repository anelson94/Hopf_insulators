# -*- coding: utf-8 -*-
"""
Created on Wed Jul 11 17:56:21 2018

@author: aleksandra
"""

# Calculate Hopf invariant

import numpy as np
from math import pi
import pickle
import cmath

# Import parameters for Hopf Hamiltonian from file params.py
# import params

# t = 1
# h = 0

# Nx = params.Nx
# Ny = params.Ny
# Nz = params.Nz
Nx = 100
Ny = 100
Nz = 100

kx = np.linspace(0, 2*pi, Nx)
ky = np.linspace(0, 2*pi, Ny)
kz = np.linspace(0, 2*pi, Nz)

[kkx, kky, kkz] = np.meshgrid(kx, ky, kz, indexing='ij')

# Import eigenstates of Hopf Humiltonian
with open('Hopfsmoothstates.pickle', 'rb') as f:
    usmooth = pickle.load(f)

# Occupied states correspond to smaller eigenvalues
# usmooth = u[:, :, :, :, 0]

# Check gauge invariance: multiply usmooth(13,13,13) by additional phase
# expmatr = np.exp(1j * np.cos(kkx) * np.sin(kky) * np.cos(kkz))
# usmooth = usmooth * expmatr[:, :, :, np.newaxis]
# usmooth[65, 38, :, :] = usmooth[65, 38, :, :] * cmath.exp(0.2j)
# usmooth[65, :, 21, :] = usmooth[65, :, 21, :] * cmath.exp(0.7j)

# Constract the overlaps between neighbor points in all possible directions
Uxy1 = np.sum(np.multiply(np.conj(usmooth[0:Nx-1, 0:Ny-1, 0:Nz-1, :]),
                          usmooth[1:Nx, 0:Ny-1, 0:Nz-1, :]), axis=-1)
Uxy2 = np.sum(np.multiply(np.conj(usmooth[1:Nx, 0:Ny-1, 0:Nz-1, :]),
                          usmooth[1:Nx, 1:Ny, 0:Nz-1, :]), axis=-1)
Uxy3 = np.sum(np.multiply(np.conj(usmooth[1:Nx, 1:Ny, 0:Nz-1, :]),
                          usmooth[0:Nx-1, 1:Ny, 0:Nz-1, :]), axis=-1)

Uyz1 = np.sum(np.multiply(np.conj(usmooth[0:Nx-1, 0:Ny-1, 0:Nz-1, :]),
                          usmooth[0:Nx-1, 1:Ny, 0:Nz-1, :]), axis=-1)
Uyz2 = np.sum(np.multiply(np.conj(usmooth[0:Nx-1, 1:Ny, 0:Nz-1, :]),
                          usmooth[0:Nx-1, 1:Ny, 1:Nz, :]), axis=-1)
Uyz3 = np.sum(np.multiply(np.conj(usmooth[0:Nx-1, 1:Ny, 1:Nz, :]),
                          usmooth[0:Nx-1, 0:Ny-1, 1:Nz, :]), axis=-1)

Uzx1 = np.sum(np.multiply(np.conj(usmooth[0:Nx-1, 0:Ny-1, 0:Nz-1, :]),
                          usmooth[0:Nx-1, 0:Ny-1, 1:Nz, :]), axis=-1)
Uzx2 = np.sum(np.multiply(np.conj(usmooth[0:Nx-1, 0:Ny-1, 1:Nz, :]),
                          usmooth[1:Nx, 0:Ny-1, 1:Nz, :]), axis=-1)
Uzx3 = np.sum(np.multiply(np.conj(usmooth[1:Nx, 0:Ny-1, 1:Nz, :]),
                          usmooth[1:Nx, 0:Ny-1, 0:Nz-1, :]), axis=-1)


# Use the formula for F and A in terms of overlaps and calculate sum_i(A_i*F_i)
underHopf = (np.multiply(
        (np.log(np.multiply(np.multiply(Uxy1, Uxy2), 
                            np.multiply(Uxy3, np.conj(Uyz1))))).imag, 
        (np.log(Uzx1)).imag) + 
            np.multiply(
        (np.log(np.multiply(np.multiply(Uyz1, Uyz2), 
                            np.multiply(Uyz3, np.conj(Uzx1))))).imag, 
        (np.log(Uxy1)).imag) + 
            np.multiply(
        (np.log(np.multiply(np.multiply(Uzx1, Uzx2), 
                            np.multiply(Uzx3, np.conj(Uxy1))))).imag, 
        (np.log(Uyz1)).imag))

underdipz = np.multiply(
        (np.log(np.multiply(np.multiply(Uxy1, Uxy2),
                            np.multiply(Uxy3, np.conj(Uyz1))))).imag,
        (np.log(Uzx1)).imag)

underdipx = np.multiply(
        (np.log(np.multiply(np.multiply(Uyz1, Uyz2),
                            np.multiply(Uyz3, np.conj(Uzx1))))).imag,
        (np.log(Uxy1)).imag)

underdipy = np.multiply(
        (np.log(np.multiply(np.multiply(Uzx1, Uzx2),
                            np.multiply(Uzx3, np.conj(Uxy1))))).imag,
        (np.log(Uyz1)).imag)

# Hopf invariant is a sum of A*F over the whole BZ                           
Hopf = - np.sum(underHopf)/(2*pi)**2

dipz = np.sum(underdipz) / (2 * pi)**2
dipx = np.sum(underdipx) / (2 * pi)**2
dipy = np.sum(underdipy) / (2 * pi)**2

print(Hopf)
print(dipz, dipx, dipy)
with open('Hopfinvariant.pickle', 'wb') as f:
    pickle.dump(underHopf, f)
