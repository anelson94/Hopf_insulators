"""
 Created by alexandra at 11.09.19 18:20
 Check Chern number functions on the simplest Chern insulator
"""

import numpy as np
from math import pi, sqrt
import matplotlib.pyplot as plt


def scalarprod(a, b):
    # Scalar product of two stackes of wavefunctions of the same size
    # Returns a stack of <A[i,j,...,:]| B[i,j,...,:]>
    prod = np.sum(np.multiply(np.conj(a), b), axis=-1)
    return prod


def chern_number_many_band(u):
    """Calculate the Chern number of a set of bands"""
    print(u.shape)
    n_bands = u.shape[-1]
    c = 0
    for idx in range(Nx - 1):
        for idy in range(Nx - 1):
            for n in range(n_bands):
                u1 = scalarprod(u[idx, idy, :, n], u[idx + 1, idy, :, n])
                u2 = scalarprod(u[idx + 1, idy, :, n], u[idx + 1, idy + 1, :, n])
                u3 = scalarprod(u[idx + 1, idy + 1, :, n], u[idx, idy + 1, :, n])
                u4 = scalarprod(u[idx, idy + 1, :, n], u[idx, idy, :, n])
                c += -np.angle(u1 * u2 * u3 * u4) / 2 / pi
    return c


def chern_number(u):
    """Calculate the Chern number of a band"""
    c = 0
    berry = np.empty((Nx-1, Nx-1))
    for idx in range(Nx - 1):
        for idy in range(Nx - 1):
            u1 = scalarprod(u[idx, idy, :], u[idx + 1, idy, :])
            u2 = scalarprod(u[idx + 1, idy, :], u[idx + 1, idy + 1, :])
            u3 = scalarprod(u[idx + 1, idy + 1, :], u[idx, idy + 1, :])
            u4 = scalarprod(u[idx, idy + 1, :], u[idx, idy, :])
            c += -np.angle(u1 * u2 * u3 * u4) / 2 / pi
            berry[idx, idy] = -np.angle(u1 * u2 * u3 * u4)
    return c, berry


def chern_states(kx, ky):
    """Eigenstates of the simplest Chern insulator"""
    u = np.empty((Nx, Nx, 2, 2), dtype=complex)
    d1 = np.sin(kx)
    d2 = np.sin(ky)
    d3 = 2 - m - np.cos(kx) + np.cos(ky)
    d = np.sqrt(np.power(d1, 2) + np.power(d2, 2) +np.power(d3, 2))
    lamb = np.sqrt(2 * d * (d - d3))
    u[:, :, 0, 0] = (d3 - d) / lamb
    u[:, :, 1, 0] = (d1 - 1j * d2) / lamb
    u[:, :, 0, 1] = (d3 + d) / lamb
    u[:, :, 1, 1] = (d1 - 1j * d2) / lamb
    return u


m = 3

Nx = 101
Kx, Ky = np.meshgrid(np.linspace(0.001, 0.001 + 2 * pi, Nx),
                     np.linspace(0.001, 0.001 + 2 * pi, Nx),
                     indexing='ij')
# Kx = np.reshape(Kx, -1)
# Ky = np.reshape(Ky, -1)

States = chern_states(Kx, Ky)

C, Berry = chern_number(States[:, :, :, 0])
print(C)
plt.figure()
plt.imshow(Berry)
plt.colorbar()
plt.show()

C = chern_number_many_band(States)
print(C)
