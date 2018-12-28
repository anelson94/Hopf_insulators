"""
 Created by alexandra at 26.12.18 14:41

 Calculate surface states for the Hopf slab with additional surface potential
 (Consider system infinite in x and y directions and finite in z direction)
"""

import numpy as np
from math import pi
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# Parameters
h = 2
t = 1

Nx = 101
Ny = 101
Nz = 16

kx = np.linspace(0, 2 * pi, Nx)
ky = np.linspace(0, 2 * pi, Ny)

[kkx, kky] = np.meshgrid(kx, ky, indexing='ij')

# Define the shape of Hamiltonian
H = np.zeros((Nx, Ny, 2 * Nz, 2 * Nz), dtype=complex)

# Construct blockes for Hopf Hamiltonian
A = (np.power(np.sin(kkx), 2) + t ** 2 * np.power(np.sin(kky), 2) -
     np.power(np.cos(kkx) + np.cos(kky) + h, 2) - 1)
B = - np.cos(kkx) - np.cos(kky) - h
C = 2 * np.multiply(t * np.sin(kky) + 1j * np.sin(kkx),
                    np.cos(kkx) + np.cos(kky) + h)
D = 2 * (t * np.sin(kky) + 1j * np.sin(kkx))

E = np.stack((np.stack((A, np.conj(C)), axis=-1),
              np.stack((C, -A), axis=-1)), axis=-1)

Delta = np.stack((np.stack((B, np.zeros((Nx, Ny))), axis=-1),
                  np.stack((D, -B), axis=-1)), axis=-1)

# Surface potential
alpha = -0.5
V1 = np.zeros((Nx, Ny))
V2 = alpha * np.ones((Nx, Ny))

Hsurf = np.stack((np.stack((V2, V1), axis=-1),
                  np.stack((V1, V2), axis=-1)), axis=-1)

# Construct Hamiltonian for all Nz sites from these blockes
H[:, :, 0:2, 0:2] = E
for nz in range(0, Nz - 1):
    H[:, :, 2 * nz + 2: 2 * nz + 4, 2 * nz + 2: 2 * nz + 4] = E
    H[:, :, 2 * nz: 2 * nz + 2, 2 * nz + 2: 2 * nz + 4] = Delta
    H[:, :, 2 * nz + 2: 2 * nz + 4, 2 * nz: 2 * nz + 2] = (
        np.transpose(np.conj(Delta), (0, 1, 3, 2)))

# Add surface potential
H[:, :, 0:2, 0:2] += Hsurf
H[:, :, -2:, -2:] += Hsurf

# Calculate eigenvalues and eigenvectors of H-Hopf
[Energy, States] = np.linalg.eigh(H)

# Set weight of eigenstates to define which are edge states
# Weight multiplier
lamb = 0.5
# zline = np.arange(2 * Nz)

# We take into accont that each atom has two orbitals
# which should have the same weight
zline = np.stack((np.arange(Nz), np.arange(Nz)), axis=-1)
zline = np.reshape(zline, 2 * Nz, order='C')
weight = np.exp(-lamb * zline)
# Left eigenstate
L = np.sum(np.multiply(np.power(np.abs(States), 2),
                       weight[np.newaxis, np.newaxis, :, np.newaxis]), axis=-2)
# Right eigenstate
R = np.sum(np.multiply(np.power(np.abs(States), 2),
                       np.flip(weight[np.newaxis, np.newaxis, :, np.newaxis],
                               axis=-2)),
           axis=-2)

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.plot_surface(kkx, kky, Energy[:, :, 15])
# ax.plot_surface(kkx, kky, Energy[:, :, 16])
# plt.show()

# Define colormap
cdict1 = {'red': ((0.0, 0.0, 0.0),
                  (0.5, 0.0, 0.1),
                  (1.0, 1.0, 1.0)),

          'green': ((0.0, 0.0, 0.0),
                    (1.0, 0.0, 0.0)),

          'blue': ((0.0, 0.0, 1.0),
                   (0.5, 0.1, 0.0),
                   (1.0, 0.0, 0.0))
          }

blue_red1 = LinearSegmentedColormap('BlueRed1', cdict1)

kxrep = np.transpose(np.tile(kx, (2 * Nz, 1)))
kyrep = np.transpose(np.tile(ky, (2 * Nz, 1)))
nkx = 50
fig = plt.figure()
# plt.scatter(kyrep, Energy[nkx, :, :], c = L[nkx, :, :] - R[nkx, :, :],
#             s = 1, cmap=blue_red1)
plt.scatter(kyrep, Energy[:, nkx, :], c=L[:, nkx, :] - R[:, nkx, :], s=1,
            cmap=blue_red1)
plt.colorbar()
plt.show()
# cross = (np.abs(Energy[:, :, 15]-Energy[:, :, 16])<0.05)
# plt.figure()
# plt.imshow(cross, cmap='winter')
# plt.show()