# -*- coding: utf-8 -*-
"""
Created on Fri Aug  3 15:49:52 2018

@author: aleksandra
"""

# The generalized Hamiltonian of Hopf insulator on a lattice (kx, ky, kz),
# kx in (0, 2pi), ky in (0, 2pi), kz in (0, 2pi). Add parameters p, q
# For all (kx, ky, kz) calculate eigenvalues and eigenvectors 
# of Hopf Hamiltonian.

import numpy as np
from math import pi
import pickle
#import math

# Import parameters for Hopf Hamiltonian from file params.py
import params

t = params.t
h = params.h

p=params.p
q=params.q

Nx = params.Nx
Ny = params.Ny
Nz = params.Nz 

kx = np.linspace(0, 2*pi, Nx)
ky = np.linspace(0, 2*pi, Ny)
kz = np.linspace(0, 2*pi, Nz)

# Cartesian coordinates, indexing for correct order x,y,z (not y,x,z)
[kkx, kky, kkz] = np.meshgrid(kx, ky, kz, indexing = 'ij')


# Hopf Hamiltonian is a mapping function from T^3 to S^2.
# It has two energy states, one of them occupied.

lamb = np.divide(1, np.power(np.abs(np.sin(kkx) + 1j*t*np.sin(kky)), 2*p) + 
                 np.power(np.abs(np.sin(kkz) + 
                 1j*(np.cos(kkx) + np.cos(kky) + np.cos(kkz) + h)), 2*q))

H11 = np.multiply(lamb, np.power(np.abs(np.sin(kkx) + 1j*t*np.sin(kky)), 2*p) - 
                 np.power(np.abs(np.sin(kkz) + 
                 1j*(np.cos(kkx) + np.cos(kky) + np.cos(kkz) + h)), 2*q))

H12 = np.multiply(2*lamb, np.multiply(np.power(np.sin(kkx) - 1j*t*np.sin(kky), p),
                  np.power(np.sin(kkz) + 
                 1j*(np.cos(kkx) + np.cos(kky) + np.cos(kkz) + h), q)))

HopfH = np.stack((np.stack((H11, np.conj(H12)), axis = -1), 
              np.stack((H12, -H11), axis = -1)), axis = -1)


# Calculate eigenvalues and eigenvectors of H-Hopf
[E, u] = np.linalg.eigh(HopfH)

with open('Hopfgeneigen.pickle', 'wb') as f:
    pickle.dump([E, u], f)
    

# Check that H = UEU^-1 = UEU^+
#print('H=',HopfH[5, 1, 3, :, :])
#Hopfreturn = u[5, 1, 3, :, :] @ np.diag(E[5, 1, 3, :]) @ np.conjugate(np.transpose(u[5, 1, 3, :, :]))
#print('Hfromdiagonalization=', Hopfreturn)
#Hopfretinv = u[5, 1, 3, :, :] @ np.diag(E[5, 1, 3, :]) @ np.linalg.inv(u[5, 1, 3, :, :])
#print('HfromdiagUinv=', Hopfretinv)