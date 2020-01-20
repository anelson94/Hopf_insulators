# -*- coding: utf-8 -*-
"""
Created on Thu Jun 21 15:59:07 2018

@author: Aleksandra
"""

# The Hamiltonian of Hopf insulator is defined on a lattice (kx, ky, kz),
# kx in (0, 2pi), ky in (0, 2pi), kz in (0, 2pi).
# For all (kx, ky, kz) calculate eigenvalues and eigenvectors 
# of Hopf Hamiltonian.

import numpy as np
from math import pi
import pickle
#import math

# Import parameters for Hopf Hamiltonian from file params.py
import params

# t = params.t
# h = params.h
#
# Nx = params.Nx
# Ny = params.Ny
# Nz = params.Nz

t = 1
h = 2
alpha = 1  # multiplier in front of kx

Nx = 101
Ny = 101
Nz = 101

kx = np.linspace(0, 2*pi, Nx)
ky = np.linspace(0, 2*pi, Ny)
kz = np.linspace(0, 2*pi, Nz)

# Cartesian coordinates, indexing for correct order x,y,z (not y,x,z)
[kkx, kky, kkz] = np.meshgrid(kx, ky, kz, indexing = 'ij')

# Pauli matrices
sigmax = np.array([[0, 1], [1, 0]])
sigmay = np.array([[0, -1j], [1j, 0]])
sigmaz = np.array([[1, 0], [0, -1]])

#Stack Nx*Ny*Nz Pauli matrices for calculations at all (kx, ky, kz)
sigmax = sigmax[np.newaxis, np.newaxis, np.newaxis, :, :]
sigmaxstack = np.tile(sigmax, (Nx, Ny, Nz, 1, 1))

sigmay = sigmay[np.newaxis, np.newaxis, np.newaxis, :, :]
sigmaystack = np.tile(sigmay, (Nx, Ny, Nz, 1, 1))

sigmaz = sigmaz[np.newaxis, np.newaxis, np.newaxis, :, :]
sigmazstack = np.tile(sigmaz, (Nx, Ny, Nz, 1, 1))

# Hopf Hamiltonian is a mapping function from T^3 to S^2.
# It has two energy states, one of them occupied.

lamb = np.divide(1, np.power(np.sin(alpha * kkx), 2)
                 + t**2 * np.power(np.sin(kky), 2) +
                 np.power(np.sin(kkz), 2) + 
                 np.power(np.cos(alpha * kkx) + np.cos(kky) + np.cos(kkz) + h, 2))

lamb = 1

Hx = np.multiply(2 * lamb, np.multiply(np.sin(alpha * kkx), np.sin(kkz)) +
                 t*np.multiply(np.sin(kky), (np.cos(alpha * kkx) + np.cos(kky) +
                                             np.cos(kkz) + h)))
Hy = np.multiply(
    2 * lamb, -t*np.multiply(np.sin(kky), np.sin(kkz))
    + np.multiply(np.sin(alpha * kkx),
                  (np.cos(alpha * kkx) + np.cos(kky) + np.cos(kkz) + h)))
Hz = np.multiply(lamb, (np.power(np.sin(alpha * kkx), 2) +
                        t**2 * np.power(np.sin(kky), 2) - 
                        np.power(np.sin(kkz), 2) -
                        np.power((np.cos(alpha * kkx) + np.cos(kky) +
                                  np.cos(kkz) + h), 2)))

Hx = Hx[:, :, :, np.newaxis, np.newaxis]
Hy = Hy[:, :, :, np.newaxis, np.newaxis]
Hz = Hz[:, :, :, np.newaxis, np.newaxis]


HopfH = (np.multiply(Hx, sigmaxstack) + np.multiply(Hy, sigmaystack) + 
         np.multiply(Hz, sigmazstack))

with open('HopfHamiltonian.pickle', 'wb') as f:
    pickle.dump(HopfH, f)

# Calculate eigenvalues and eigenvectors of H-Hopf
[E, u] = np.linalg.eigh(HopfH)

with open('Hopfeigen.pickle', 'wb') as f:
    pickle.dump([E, u], f)
    

# Check that H = UEU^-1 = UEU^+
#print('H(pi,pi/5,3pi/5)=',HopfH[5, 1, 3, :, :])
#Hopfreturn = u[5, 1, 3, :, :] @ np.diag(E[5, 1, 3, :]) @ np.conjugate(np.transpose(u[5, 1, 3, :, :]))
#print('Hfromdiagonalization=', Hopfreturn)
#Hopfretinv = u[5, 1, 3, :, :] @ np.diag(E[5, 1, 3, :]) @ np.linalg.inv(u[5, 1, 3, :, :])
#print('HfromdiagUinv=', Hopfretinv)