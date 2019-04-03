"""
 Created by alexandra at 01.02.19 12:43
 
 The Hamiltonian of 4*4 Hopf insulator 
 Calculate eigenvalues and eigenvectors 
 of Hopf Hamiltonian.
"""

import numpy as np
from math import pi
import pickle


def stackmatr(matr):
    """Make the stack of Nx*Ny*Nz matrices"""
    matr = matr[np.newaxis, np.newaxis, np.newaxis, :, :]
    return np.tile(matr, (Nx, Ny, Nz, 1, 1))


def adddimmatr(matr):
    """Add additional dimentions to hamilt matrices"""
    return matr[:, :, :, np.newaxis, np.newaxis]


def normalise(kkx, kky, kkz):
    """normalisation factor for hamilt vector"""
    return np.divide(
        1,
        np.power(np.sin(kkx), 2) + np.power(np.sin(kky), 2)
        + np.power(np.sin(kkz), 2)
        + np.power(np.cos(kkx) + np.cos(kky) + np.cos(kkz) + h, 2)
    )


def hx(kkx, kky, kkz):
    """x component of hamilt vector"""
    return 2 * (
            np.sin(kkx) * np.sin(kkz)
            + np.sin(kky) * (np.cos(kkx) + np.cos(kky) + np.cos(kkz) + h)
    )


def hy(kkx, kky, kkz):
    """y component of hamilt vector"""
    return 2 * (
            np.sin(kky) * np.sin(kkz)
            - np.sin(kkx) * (np.cos(kkx) + np.cos(kky) + np.cos(kkz) + h)
    )


def hz(kkx, kky, kkz):
    """z component of hamilt vector"""
    return (
        np.power(np.sin(kkx), 2)
        + np.power(np.sin(kky), 2)
        - np.power(np.sin(kkz), 2)
        - np.power(np.cos(kkx) + np.cos(kky) + np.cos(kkz) + h, 2)
    )


h = 2

Nx = 201
Ny = 201
Nz = 201

Kx = np.linspace(0, 2 * pi, Nx)
Ky = np.linspace(0, 2 * pi, Ny)
Kz = np.linspace(0, 2 * pi, Nz)

# Cartesian coordinates, indexing for correct order x,y,z (not y,x,z)
[KKx, KKy, KKz] = np.meshgrid(Kx, Ky, Kz, indexing='ij')

# Pauli matrices
sigmax = np.array([[0, 1], [1, 0]])
sigmay = np.array([[0, -1j], [1j, 0]])
sigmaz = np.array([[1, 0], [0, -1]])

# Stack Nx*Ny*Nz Pauli matrices for calculations at all (kx, ky, kz)
sigmaxstack = stackmatr(sigmax)
sigmaystack = stackmatr(sigmay)
sigmazstack = stackmatr(sigmaz)

# Construct 2*2 Hopf Hamiltonian
Hx = hx(KKx, KKy, KKz) * normalise(KKx, KKy, KKz)
Hy = hy(KKx, KKy, KKz) * normalise(KKx, KKy, KKz)
Hz = hz(KKx, KKy, KKz) * normalise(KKx, KKy, KKz)

Hx = adddimmatr(Hx)
Hy = adddimmatr(Hy)
Hz = adddimmatr(Hz)

HopfH = Hx * sigmaxstack + Hy * sigmaystack + Hz * sigmazstack

# Add two more bands
zeromatr = np.zeros((Nx, Ny, Nz, 2, 2))
twolevmatr = np.array([[1, 0], [0, -1]])
twolevmatrstack = stackmatr(twolevmatr)

HopfH44 = np.concatenate(
    (np.concatenate((HopfH, zeromatr), axis=-1),
     np.concatenate((zeromatr, twolevmatrstack), axis=-1)),
    axis=-2
)

# Calculate eigenvalues and eigenvectors of H-Hopf
[E44, u44] = np.linalg.eigh(HopfH44)

with open('Hopf44eigen.pickle', 'wb') as f:
    pickle.dump([E44, u44], f)
