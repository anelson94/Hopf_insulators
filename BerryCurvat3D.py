"""
 Created by alexandra at 14.02.19 17:56

 Plot the Berry curvature of Hopf insulator in 3 dimentional space
"""

import numpy as np
from math import pi
import pickle
import cmath
import math
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from mpl_toolkits.mplot3d import Axes3D


def dirandnorm(x, y, z):
    """Define the directions and module of vector field"""
    r = np.sqrt(np.power(x, 2) + np.power(y, 2) + np.power(z, 2))
    return r, x / r, y / r, z / r


Nx = 101
Ny = 101
Nz = 101

kx = np.linspace(0, 2 * pi, Nx)
ky = np.linspace(0, 2 * pi, Ny)
kz = np.linspace(0, 2 * pi, Nz)

# Import eigenstates of Hopf Humiltonian
with open('HopfeigenWeyl.pickle', 'rb') as f:
    [E, u] = pickle.load(f)

# Occupied states correspond to smaller eigenvalues
uOcc = u[:, :, :, :, 0]
Nz_small = math.floor((Nz - 1) / 20) + 1
TrFx = np.zeros((Nx, Ny, Nz_small))
TrFy = np.zeros((Nx, Ny, Nz_small))
TrFz = np.zeros((Nx, Ny, Nz_small))

# Calculate Berry flux in x and z direction as a function of kx, kz
# Use numerical calculation method
for nkzf in range(0, Ny, 20):
    nkz = nkzf % Nz - 1
    kkz = kz[nkz]
    for nkxf in range(0, Nx):
        nkx = nkxf % Nx - 1
        kkx = kx[nkx]
        for nkyf in range(0, Ny):
            nky = nkyf % Ny - 1
            kky = ky[nky]
            U1x = np.dot(np.conj(uOcc[nkx, nky, nkz, :]),
                         uOcc[nkx, nky + 1, nkz, :])
            U2x = np.dot(np.conj(uOcc[nkx, nky + 1, nkz, :]),
                         uOcc[nkx, nky + 1, nkz + 1, :])
            U3x = np.dot(np.conj(uOcc[nkx, nky + 1, nkz + 1, :]),
                         uOcc[nkx, nky, nkz + 1, :])
            U4x = np.dot(np.conj(uOcc[nkx, nky, nkz + 1, :]),
                         uOcc[nkx, nky, nkz, :])
            TrFx[nkx, nky, round(nkz / 20)] = - (
                    (cmath.log(U1x * U2x * U3x * U4x)).imag
                    * Ny * Nz / (2 * pi) ** 2)

            U1z = np.dot(np.conj(uOcc[nkx, nky, nkz, :]),
                         uOcc[nkx + 1, nky, nkz, :])
            U2z = np.dot(np.conj(uOcc[nkx + 1, nky, nkz, :]),
                         uOcc[nkx + 1, nky + 1, nkz, :])
            U3z = np.dot(np.conj(uOcc[nkx + 1, nky + 1, nkz, :]),
                         uOcc[nkx, nky + 1, nkz, :])
            U4z = np.dot(np.conj(uOcc[nkx, nky + 1, nkz, :]),
                         uOcc[nkx, nky, nkz, :])
            TrFz[nkx, nky, round(nkz / 20)] = - (
                    (cmath.log(U1z * U2z * U3z * U4z)).imag
                    * Nx * Ny / (2 * pi) ** 2)

            U1y = np.dot(np.conj(uOcc[nkx, nky, nkz, :]),
                         uOcc[nkx, nky, nkz + 1, :])
            U2y = np.dot(np.conj(uOcc[nkx, nky, nkz + 1, :]),
                         uOcc[nkx + 1, nky, nkz + 1, :])
            U3y = np.dot(np.conj(uOcc[nkx + 1, nky, nkz + 1, :]),
                         uOcc[nkx + 1, nky, nkz, :])
            U4y = np.dot(np.conj(uOcc[nkx + 1, nky, nkz, :]),
                         uOcc[nkx, nky, nkz, :])
            TrFy[nkx, nky, round(nkz / 20)] = - (
                    (cmath.log(U1y * U2y * U3y * U4y)).imag
                    * Nx * Nz / (2 * pi) ** 2)

fig = plt.figure()
ax = fig.gca(projection='3d')

# grid
[Kx, Ky, Kz] = np.meshgrid(kx, ky, kz[::20], indexing='ij')
ax.quiver(Kx[8:Nx - 8:9, 8:Ny - 8:9, 2:Nz_small - 3],
          Ky[8:Nx - 8:9, 8:Ny - 8:9, 2:Nz_small - 3],
          Kz[8:Nx - 8:9, 8:Ny - 8:9, 2:Nz_small - 3],
          TrFx[8:Nx - 8:9, 8:Ny - 8:9, 2:Nz_small - 3],
          TrFy[8:Nx - 8:9, 8:Ny - 8:9, 2:Nz_small - 3],
          TrFz[8:Nx - 8:9, 8:Ny - 8:9, 2:Nz_small - 3])
plt.show()
