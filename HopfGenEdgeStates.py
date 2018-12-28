"""
 Created by alexandra at 27.12.18 16:07

 Create slab Hamiltonian for generalized Hopf (p, q) and calculate edge states
"""

import numpy as np
from math import pi, sqrt
from scipy import integrate
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


def slab_ham(nx, n_slab):
    """Create slab hamiltonian"""
    n_max = 2 * q
    hamilt = np.zeros((nx, 2 * (n_slab + n_max), 2 * (n_slab + n_max)),
                      dtype=complex)
    for n in range(n_max + 1):
        matr = ham_n(n, nx)
        for nz in range(n_slab):
            hamilt[:, 2 * nz:2 * nz + 2,
                   2 * nz + 2 * n:2 * nz + 2 + 2 * n] = matr
            if n != 0:
                hamilt[:, 2 * nz + 2 * n:2 * nz + 2 + 2 * n,
                       2 * nz:2 * nz + 2] = np.conj(np.transpose(matr,
                                                    (0, 2, 1)))
    return hamilt[:, :2 * n_slab, :2 * n_slab]


def ham_n(n, nx):
    """Hamiltonian of interection of sites separated by n cells"""
    ham_matr = np.zeros((nx, 2, 2), dtype=complex)
    for idx in range(nx):
        kkx = Kx[idx]
        kky = Ky[idx]
        ham_matr[idx, 0, 0] = fourier(kkx, kky, 0, n)  # (1, 1)
        ham_matr[idx, 1, 1] = -ham_matr[idx, 0, 0]  # (2, 2)
        ham_matr[idx, 0, 1] = fourier(kkx, kky, 1, n)  # (1, 2)
        ham_matr[idx, 1, 0] = fourier(kkx, kky, 2, n)
        # (2, 1)
    return ham_matr


def func11(kkx, kky, kz):
    """11 component of Hopf hamiltonian"""
    return ((np.power(np.sin(kkx), 2) + t**2 * np.power(np.sin(kky), 2))**p
            - (np.power(np.sin(kz), 2)
               + np.power(np.cos(kkx) + np.cos(kky) + np.cos(kz) + h, 2))**q)


def func12(kkx, kky, kz):
    """12 component of Hopf hamiltonian"""
    return 2 * ((np.sin(kkx) - 1j * t * np.sin(kky))**p
                * (np.sin(kz) - 1j * (np.cos(kkx) + np.cos(kky)
                                      + np.cos(kz) + h))**q)


def func21(kkx, kky, kz):
    """12 component of Hopf hamiltonian"""
    return 2 * ((np.sin(kkx) + 1j * t * np.sin(kky))**p
                * (np.sin(kz) + 1j * (np.cos(kkx) + np.cos(kky)
                                      + np.cos(kz) + h))**q)


def fourier(kkx, kky, idx, n):
    def integrandre(kz):
        return np.real(func_dict[idx](kkx, kky, kz) * np.exp(1j * n * kz))

    def integrandim(kz):
        return np.imag(func_dict[idx](kkx, kky, kz) * np.exp(1j * n * kz))

    [valre, erre] = integrate.quad(integrandre, 0, 2 * pi)
    [valim, erim] = integrate.quad(integrandim, 0, 2 * pi)
    return valre + 1j * valim


# parameters
p = 1
q = 1
h = 2
t = 1

Nx = 101
Ny = 101
N_slab = 16

Kx = np.append(np.linspace(0, pi, round(Nx * sqrt(2))), np.ones(Nx) * pi)
Kx = np.append(Kx, np.linspace(pi, 0, Nx))
Ky = np.append(np.linspace(0, pi, round(Nx * sqrt(2))), np.linspace(pi, 0, Nx))
Ky = np.append(Ky, np.zeros(Nx))

func_dict = {0: func11, 1: func12, 2: func21}

H_slab = slab_ham(2 * Nx + round(Nx * sqrt(2)), N_slab)

# Onsite surface potential
alpha = 0
V1 = np.zeros(2 * Nx + round(Nx * sqrt(2)))
V2 = alpha * np.ones(2 * Nx + round(Nx * sqrt(2)))

Hsurf = np.stack((np.stack((V2, V1), axis=-1),
                  np.stack((V1, V2), axis=-1)), axis=-1)

# Add surface potential
H_slab[:, 0:2, 0:2] += Hsurf
H_slab[:, -2:, -2:] += Hsurf


# Calculate eigenvalues and eigenvectors of H-Hopf
[Energy, States] = np.linalg.eigh(H_slab)

Kxplot = np.append(np.linspace(0, 1, round(Nx * sqrt(2))) * sqrt(2),
                   np.linspace(0, 1, Nx) + sqrt(2))
Kxplot = np.append(Kxplot, np.linspace(0, 1, Nx) + sqrt(2) + 1)
Kxrep = np.transpose(np.tile(Kxplot, (2 * N_slab, 1)))

# Set weight of eigenstates to define which are edge states
# Weight multiplier
lamb = 20

# We take into accont that each atom has two orbitals
# which should have the same weight
zline = np.stack((np.arange(N_slab), np.arange(N_slab)), axis=-1)
zline = np.reshape(zline, 2 * N_slab, order='C')
# Order of layer in relation to the surface
# Look for the states with maximal projection on this layer
n_edge = 1
weight = np.exp(-lamb * abs(zline-n_edge))
# Left eigenstate
L = np.sum(np.multiply(np.power(np.abs(States), 2),
           weight[np.newaxis, :, np.newaxis]), axis=-2)
# Right eigenstate
R = np.sum(np.multiply(np.power(np.abs(States), 2),
           np.flip(weight[np.newaxis, :, np.newaxis], axis=-2)),
           axis=-2)

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

xcoords = [0, sqrt(2), 1 + sqrt(2), 2 + sqrt(2)]
fig = plt.figure()
plt.scatter(Kxrep, Energy[:, :], c=L[:, :] - R[:, :], s=1, cmap=blue_red1)
plt.ylim(-30, 30)
plt.xlim(0, 2 + sqrt(2))
plt.xticks(xcoords, ['$\Gamma$', 'M', 'K', '$\Gamma$'])
for xc in xcoords:
    plt.axvline(x=xc, color='k', linewidth=1)
plt.ylabel('E')
plt.colorbar()
plt.show()
