"""
 Created by alexandra at 26.12.18 14:41

 Calculate surface states for the Hopf slab with additional surface potential
 (Consider system infinite in x and y directions and finite in z direction)
"""

import numpy as np
from math import pi, sqrt
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# Parameters
h = 0.5
t = 1

Nx = 101
N_slab = 16

Kx = np.append(np.linspace(0, pi, round(Nx * sqrt(2))), np.ones(Nx) * pi)
Kx = np.append(Kx, np.linspace(pi, 0, Nx))
Ky = np.append(np.linspace(0, pi, round(Nx * sqrt(2))), np.linspace(pi, 0, Nx))
Ky = np.append(Ky, np.zeros(Nx))
Ntotal = 2 * Nx + round(Nx * sqrt(2))

# Define the shape of Hamiltonian
H = np.zeros((Ntotal, 2 * N_slab, 2 * N_slab), dtype=complex)

# Construct blockes for Hopf Hamiltonian
A = (np.power(np.sin(Kx), 2) + t**2 * np.power(np.sin(Ky), 2) -
     np.power(np.cos(Kx) + np.cos(Ky) + h, 2) - 1)
B = - np.cos(Kx) - np.cos(Ky) - h
C = 2 * np.multiply(t * np.sin(Ky) + 1j * np.sin(Kx),
                    np.cos(Kx) + np.cos(Ky) + h)
D = 2 * (t * np.sin(Ky) + 1j * np.sin(Kx))

# Break the mxT symmetry:
# kz: cos<->sin
# B = -1j * B
# C = 2 * np.multiply(t * np.sin(Ky + Kx) + 1j * np.sin(Kx - Ky),
#                 np.cos(Kx - Ky) + np.cos(Ky + Kx) + h)
# D = 2 * (t * np.sin(Ky) + 1j * np.sin(Kx))

E = np.stack((np.stack((A, np.conj(C)), axis=-1),
              np.stack((C, -A), axis=-1)), axis=-1)

Delta = np.stack((np.stack((B, np.zeros(Ntotal)), axis=-1),
                  np.stack((D, -B), axis=-1)), axis=-1)

# Surface potential
alpha = 1.5
V1 = np.zeros(Ntotal)
V2 = alpha * np.ones(Ntotal)

Hsurf = np.stack((np.stack((V2, V1), axis=-1),
                  np.stack((V1, V2), axis=-1)), axis=-1)

# Construct Hamiltonian for all Nz sites from these blockes
H[:, 0:2, 0:2] = E
for nz in range(0, N_slab - 1):
    H[:, 2 * nz + 2: 2 * nz + 4, 2 * nz + 2: 2 * nz + 4] = E
    H[:, 2 * nz: 2 * nz + 2, 2 * nz + 2: 2 * nz + 4] = Delta
    H[:, 2 * nz + 2: 2 * nz + 4, 2 * nz: 2 * nz + 2] = (
        np.transpose(np.conj(Delta), (0, 2, 1)))

# Add surface potential
H[:, 0:2, 0:2] += -Hsurf
H[:, -2:, -2:] += Hsurf

# Calculate eigenvalues and eigenvectors of H-Hopf
[Energy, States] = np.linalg.eigh(H)

# Set weight of eigenstates to define which are edge states
# Weight multiplier
lamb = 0.5

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


Kxplot = np.append(np.linspace(0, 1, round(Nx * sqrt(2))) * sqrt(2),
                   np.linspace(0, 1, Nx) + sqrt(2))
Kxplot = np.append(Kxplot, np.linspace(0, 1, Nx) + sqrt(2) + 1)
Kxrep = np.transpose(np.tile(Kxplot, (2 * N_slab, 1)))
xcoords = [0, sqrt(2), 1 + sqrt(2), 2 + sqrt(2)]

# Plot the spectrum
# Size foe 'Energy' label
# fig = plt.figure(figsize=(1.78, 1.5))
# ax = fig.add_axes([0.22, 0.15, 0.7303, 0.83])
# Size without labels
# fig = plt.figure(figsize=(1.68, 1.5))
# ax = fig.add_axes([0.2024, 0.15, 0.7738, 0.83])
# Size with colorbar
fig = plt.figure(figsize=(2, 1.5))
ax = fig.add_axes([0.17, 0.15, 0.65, 0.83])
colax = fig.add_axes([0.85, 0.15, 0.05, 0.83])
fs = 10
fss = 8
lw = 1.2
# ax.set_ylabel('Energy', fontsize=fs)
ax.yaxis.set_label_coords(-0.15, 0.5)
ax.tick_params(width=lw, labelsize=fss)
for axis in ['top', 'bottom', 'left', 'right']:
    ax.spines[axis].set_linewidth(lw)
Spectr = ax.scatter(Kxrep, Energy, c=L[:, :] - R[:, :], s=0.004, cmap=blue_red1)
ax.set_xlim(0, 2 + sqrt(2))
ax.set_xticks(xcoords)
ax.set_xticklabels(['$\Gamma$', 'M', 'K', '$\Gamma$'], fontsize=fs)
for xc in xcoords:
    ax.axvline(x=xc, color='k', linewidth=lw)
cbar = fig.colorbar(Spectr, cax=colax)
cbar.ax.tick_params(labelsize=fss, width=lw, labelrotation=90)
for axis in ['top', 'bottom', 'left', 'right']:
    cbar.ax.spines[axis].set_linewidth(lw)
# plt.show()
plt.savefig('Surfpmpot15_h05t1.png', bbox_inches=None)
