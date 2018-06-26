# -*- coding: utf-8 -*-
"""
Created on Tue Jun 26 09:41:50 2018

@author: aleksandra
"""

# Construct Hopf Hamiltonian in a for-loop for every (kx, ky, kz)
# Calc eigenvalues and eigenfunctions of H Hopf
# We need this to check numpy calculations

import numpy as np
from math import pi, sin, cos
import pickle

import params

t = params.t
h = params.h

Nx = params.Nx
Ny = params.Ny
Nz = params.Nz 

kx = np.linspace(0, 2*pi, Nx)
ky = np.linspace(0, 2*pi, Ny)
kz = np.linspace(0, 2*pi, Nz)

# Pauli matrices
sigmax = np.array([[0, 1], [1, 0]])
sigmay = np.array([[0, -1j], [1j, 0]])
sigmaz = np.array([[1, 0], [0, -1]])

Ek = np.empty([Nx, Ny, Nz, 2])
uk = np.empty([Nx, Ny, Nz, 2, 2], dtype = complex)

for nkx in range(0, Nx):
    for nky in range(0, Ny):
        for nkz in range(0, Nz):
            kkx = kx[nkx]
            kky = ky[nky]
            kkz = kz[nkz]
            Lamb = 1 / (sin(kkx)**2 + sin(kky)**2 + sin(kkz)**2 + 
                        (cos(kkx) + cos(kky) + cos(kkz) + h)**2)
            Hx = 2 * Lamb * (sin(kkx)*sin(kky) + 
                             t*sin(kky)*(cos(kkx) + cos(kky) + cos(kkz) + h))
            Hy = 2 * Lamb * (t*sin(kky)*sin(kkz) - 
                             sin(kkx)*(cos(kkx) + cos(kky) + cos(kkz) + h))
            Hz = Lamb * (sin(kkx)**2 + t**2*sin(kky)**2 - sin(kkz)**2 - 
                         (cos(kkx) + cos(kky) + cos(kkz) + h)**2)
            HopfH = Hx*sigmax + Hy*sigmay + Hz*sigmaz
            
            [Ek[nkx, nky, nkz, :], 
             uk[nkx, nky, nkz, :, :]] = np.linalg.eigh(HopfH)
            
with open('Hopfeigen.pickle', 'rb') as f:
    [E, u] = pickle.load(f)
    
print(np.sum(abs(E-Ek)))

with open('Hopfeigeneachk.pickle', 'wb') as f:
    pickle.dump([Ek, uk], f)