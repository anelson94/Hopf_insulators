"""
 Created by alexandra at 18.01.19 10:51

 Construct Hopf Hamiltonian on a lattice with term splitting to Weyl points
"""

import numpy as np
from math import pi
import pickle


def stackmatr(matr):
    """Make the stack of Nx*Ny*Nz matrices"""
    matr = matr[np.newaxis, np.newaxis, np.newaxis, :, :]
    return np.tile(matr, (Nx, Ny, Nz, 1, 1))


t = 1
h = 1
Asquare = 0  # Splitting constant

Nx = 201
Ny = 201
Nz = 201

kx = np.linspace(0, 2 * pi, Nx)
ky = np.linspace(0, 2 * pi, Ny)
kz = np.linspace(0, 2 * pi, Nz)

# Cartesian coordinates, indexing for correct order x,y,z (not y,x,z)
[kkx, kky, kkz] = np.meshgrid(kx, ky, kz, indexing='ij')

# Pauli matrices
sigmax = np.array([[0, 1], [1, 0]])
sigmay = np.array([[0, -1j], [1j, 0]])
sigmaz = np.array([[1, 0], [0, -1]])

# Stack Nx*Ny*Nz Pauli matrices for calculations at all (kx, ky, kz)
sigmaxstack = stackmatr(sigmax)
sigmaystack = stackmatr(sigmay)
sigmazstack = stackmatr(sigmaz)

# Construct not normalized Hamiltonian with additional splitting term

# # kx -> 2kx
# kkx = 3 * kkx

Hx = 2 * (np.multiply(np.sin(kkx), np.sin(kkz)) +
          t * np.multiply(np.sin(kky),
                          (np.cos(kkx) + np.cos(kky) + np.cos(kkz) + h)))
Hy = 2 * (t * np.multiply(np.sin(kky), np.sin(kkz)) -
          np.multiply(np.sin(kkx),
                      (np.cos(kkx) + np.cos(kky) + np.cos(kkz) + h)))
Hz = (np.power(np.sin(kkx), 2) + t ** 2 * np.power(np.sin(kky), 2)
      - np.power(np.sin(kkz), 2)
      - np.power((np.cos(kkx) + np.cos(kky) + np.cos(kkz) + h), 2)
      + Asquare)

# Normalize Hamiltonian
# Norm = np.divide(1, np.sqrt(np.power(Hx, 2) + np.power(Hy, 2)
#                             + np.power(Hz, 2)))
#
# Hx = Hx * Norm
# Hy = Hy * Norm
# Hz = Hz * Norm

Hx = Hx[:, :, :, np.newaxis, np.newaxis]
Hy = Hy[:, :, :, np.newaxis, np.newaxis]
Hz = Hz[:, :, :, np.newaxis, np.newaxis]

HopfH = (np.multiply(Hx, sigmaxstack) + np.multiply(Hy, sigmaystack) +
         np.multiply(Hz, sigmazstack))

# Calculate eigenvalues and eigenvectors of H-Hopf
[E, u] = np.linalg.eigh(HopfH)

with open('HopfeigenWeyl.pickle', 'wb') as f:
    pickle.dump([E, u], f)

