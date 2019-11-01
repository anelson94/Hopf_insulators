# -*- coding: utf-8 -*-
"""
Created on Thu Jul 19 10:48:01 2018

@author: aleksandra
"""

# Calculate edge states in Hopf insulator
# Consider system infinite in x and y directions and finite in z direction
# Hamiltonian depends on (kx,ky,z)

import numpy as np
from math import pi, sqrt
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LinearSegmentedColormap

# Parameters
h = 0.5
t = 1

Nx = 101
Ny = 101
N_slab = 16

kx = np.linspace(0, 2*pi, Nx)
ky = np.linspace(0, 2*pi, Ny)
Kx = np.append(np.linspace(0, pi, round(Nx * sqrt(2))), np.ones(Nx) * pi)
Kx = np.append(Kx, np.linspace(pi, 0, Nx))
Ky = np.append(np.linspace(0, pi, round(Nx * sqrt(2))), np.linspace(pi, 0, Nx))
Ky = np.append(Ky, np.zeros(Nx))
Ntotal = 2 * Nx + round(Nx * sqrt(2))

[kkx, kky] = np.meshgrid(kx, ky, indexing='ij')

# Define the shape of Hamiltonian
H = np.zeros((Ntotal, 2 * N_slab, 2 * N_slab), dtype=complex)

# Construct blockes for Hopf Hamiltonian
A = (np.power(np.sin(Kx), 2) + t**2 * np.power(np.sin(Ky), 2) - 
     np.power(np.cos(Kx) + np.cos(Ky) + h, 2) - 1)
B = - np.cos(Kx) - np.cos(Ky) - h
C = 2 * np.multiply(t * np.sin(Ky) + 1j * np.sin(Kx), 
                    np.cos(Kx) + np.cos(Ky) + h)
D = 2 * (t * np.sin(Ky) + 1j * np.sin(Kx))

E = np.stack((np.stack((A, np.conj(C)), axis=-1),
              np.stack((C, -A), axis=-1)), axis=-1)

Delta = np.stack((np.stack((B, np.zeros(Ntotal)), axis=-1),
                  np.stack((D, -B), axis=-1)), axis=-1)

# Construct Hamiltonian for all N_slab sites from these blockes
H[:, 0:2, 0:2] = E
for nz in range(0, N_slab-1):
    H[:, 2*nz + 2: 2*nz + 4, 2*nz + 2: 2*nz + 4] = E
    H[:, 2*nz: 2*nz + 2, 2*nz + 2: 2*nz + 4] = Delta
    H[:, 2*nz + 2: 2*nz + 4, 2*nz: 2*nz + 2] = (
            np.transpose(np.conj(Delta), (0, 2, 1)))
   
# Calculate eigenvalues and eigenvectors of H-Hopf
[Energy, States] = np.linalg.eigh(H)

# Set weight of eigenstates to define which are edge states
# Weight multiplier
lamb = 0.5
# zline = np.arange(2 * N_slab)

# We take into accont that each atom has two orbitals 
# which should have the same weight
zline = np.stack((np.arange(N_slab), np.arange(N_slab)), axis=-1)
zline = np.reshape(zline, 2*N_slab, order='C')
weight = np.exp(-lamb * zline)
# Left eigenstate
L = np.sum(np.multiply(np.power(np.abs(States), 2),
           weight[np.newaxis, :, np.newaxis]), axis=-2)
# Right eigenstate
R = np.sum(np.multiply(np.power(np.abs(States), 2),
           np.flip(weight[np.newaxis, :, np.newaxis], axis=-2)),
           axis=-2)


# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.plot_surface(kkx, kky, Energy[:, :, 15])
# ax.plot_surface(kkx, kky, Energy[:, :, 16])
# plt.show()

# Define colormap 
cdict1 = {'red':   ((0.0, 0.0, 0.0),
                    (0.5, 0.0, 0.1),
                    (1.0, 1.0, 1.0)),

          'green': ((0.0, 0.0, 0.0),
                    (1.0, 0.0, 0.0)),

          'blue':  ((0.0, 0.0, 1.0),
                    (0.5, 0.1, 0.0),
                    (1.0, 0.0, 0.0))
          }

blue_red1 = LinearSegmentedColormap('BlueRed1', cdict1)


Kxplot = np.append(np.linspace(0, 1, round(Nx * sqrt(2))) * sqrt(2),
                   np.linspace(0, 1, Nx) + sqrt(2))
Kxplot = np.append(Kxplot, np.linspace(0, 1, Nx) + sqrt(2) + 1)
Kxrep = np.transpose(np.tile(Kxplot, (2 * N_slab, 1)))
fig = plt.figure()
# plt.scatter(kyrep, Energy[nkx, :, :], c = L[nkx, :, :] - R[nkx, :, :],
# s = 1, cmap=blue_red1)
plt.scatter(Kxrep, Energy, c=L[:, :] - R[:, :], s=1, cmap=blue_red1)
plt.colorbar()
plt.show()

# cross = (np.abs(Energy[:, :, 15]-Energy[:, :, 16])<0.05)
# plt.figure()
# plt.imshow(cross, cmap='winter')
# plt.show()
