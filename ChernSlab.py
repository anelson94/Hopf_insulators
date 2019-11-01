"""
 Created by alexandra at 18.09.19 14:16

 Calculate energy spectrum for a Chern finite slab in y direction.
 PBC in x direction
"""

import numpy as np
from math import pi, sqrt
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


def slab_hamiltonian(kx):
    """Construct a y-slab hamiltonian for kx wavevector"""
    ntotal = len(kx)
    hh = np.zeros((ntotal, 2 * N_slab, 2 * N_slab), dtype=complex)

    # Construct blockes for Hopf Hamiltonian
    a = 2 - m - np.cos(kx)
    b = np.sin(kx)
    c = -1/2 * np.ones(ntotal)
    d = -1/2 * np.ones(ntotal)

    e = np.stack((np.stack((a, b), axis=-1),
                  np.stack((b, -a), axis=-1)), axis=-1)

    delta = np.stack((np.stack((c, -d), axis=-1),
                      np.stack((d, -c), axis=-1)), axis=-1)

    # Construct Hamiltonian for all Nz sites from these blockes
    hh[:, 0:2, 0:2] = e
    for nz in range(0, N_slab - 1):
        hh[:, 2 * nz + 2: 2 * nz + 4, 2 * nz + 2: 2 * nz + 4] = e
        hh[:, 2 * nz: 2 * nz + 2, 2 * nz + 2: 2 * nz + 4] = delta
        hh[:, 2 * nz + 2: 2 * nz + 4, 2 * nz: 2 * nz + 2] = (
            np.transpose(np.conj(delta), (0, 2, 1)))
    return hh


def surf_hamiltonian(hh):
    """Additional surface potential for the slab hamiltonian h"""
    ntotal = hh.shape[0]
    # Surface potential
    alpha = 0
    beta = 0
    gamma = 0
    delta = 0
    v1 = np.zeros(ntotal)
    v2 = np.ones(ntotal)

    hsurfx = np.stack(
        (np.stack((v1, v2), axis=-1),
         np.stack((v2, v1), axis=-1)),
        axis=-1) * alpha  # * np.cos(ky[:, np.newaxis, np.newaxis])
    hsurfy = np.stack(
        (np.stack((v1, 1j * v2), axis=-1),
         np.stack((-1j * v2, v1), axis=-1)),
        axis=-1) * beta  # * np.sin(ky[:, np.newaxis, np.newaxis])
    hsurfz = np.stack(
        (np.stack((v2, v1), axis=-1),
         np.stack((v1, -v2), axis=-1)),
        axis=-1) * gamma  # * np.cos(ky[:, np.newaxis, np.newaxis])
    hsurf1 = np.stack(
        (np.stack((v2, v1), axis=-1),
         np.stack((v1, v2), axis=-1)),
        axis=-1) * delta  # * (np.cos(ky[:, np.newaxis, np.newaxis]))

    hsurf = hsurfx + hsurfy + hsurfz + hsurf1

    # Add surface potential
    hh[:, 0:2, 0:2] += hsurf
    # hh[:, 2:4, 2:4] += -hsurf
    # hh[:, 4:6, 4:6] += hsurf
    # hh[:, -4:-2, -4:-2] += hsurf
    # hh[:, -2:, -2:] += hsurf

    # Disentangle surface states
    hh[:, 0:2, 0:2] /= 2.5
    # hh[:, 2:4, 2:4] /= 2.0
    # hh[:, 0:2, 2:4] /= 1.5
    # hh[:, -4:-2, -4:-2] /=2.0
    # hh[:, -2:, -2:] /= 2.5

    return hh


def weight_surface(states):
    """Calculate the upper and lower weights of energy bands"""
    # Weight multiplier
    lamb = 0.5

    # We take into accont that each atom has two orbitals
    # which should have the same weight
    zline = np.stack((np.arange(N_slab), np.arange(N_slab)), axis=-1)
    zline = np.reshape(zline, 2 * N_slab, order='C')
    weight = np.exp(-lamb * zline)
    # Left eigenstate
    l = np.sum(np.multiply(np.power(np.abs(states), 2),
                           weight[np.newaxis, :, np.newaxis]), axis=-2)
    # print(L[round(Nx * sqrt(2)), :])
    # print(Energy[round(Nx * sqrt(2)), :])
    # Right eigenstate
    r = np.sum(np.multiply(np.power(np.abs(states), 2),
                           np.flip(weight[np.newaxis, :, np.newaxis], axis=-2)),
               axis=-2)
    print(np.max(l))
    return l, r


def plot_spectrum():
    """Plot spectrum of hamiltonian for kx in [-pi,pi]"""
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

    kxplot = np.linspace(0, 1, Nx)
    kxrep = np.transpose(np.tile(kxplot, (2 * N_slab, 1)))
    xcoords = [0, 1/2, 1]

    # Plot the spectrum
    # Size foe 'Energy' label
    # fig = plt.figure(figsize=(1.78, 1.5))
    # ax = fig.add_axes([0.22, 0.15, 0.7303, 0.83])
    # Size without labels
    fig = plt.figure(figsize=(3, 3))
    ax = fig.add_axes([0.15, 0.14, 0.81, 0.82])
    # Size with colorbar
    # fig = plt.figure(figsize=(2, 1.5))
    # ax = fig.add_axes([0.17, 0.15, 0.65, 0.83])
    # colax = fig.add_axes([0.85, 0.15, 0.05, 0.83])
    fs = 20
    fss = 17
    lw = 1.2
    ps = 0.02
    tl = 3
    pd = 4

    # Size for poster
    # fig = plt.figure(figsize=(10, 9))
    # ax = fig.add_axes([0.13, 0.1, 0.85, 0.85])
    # fs = 40
    # fss = 35
    # lw = 3.5
    # ps = 3.5
    # tl = 10
    # pd = 20
    ax.set_xlabel('$k_x$', fontsize=fss)
    ax.xaxis.set_label_coords(0.85, -0.02)
    ax.tick_params(width=lw, length=tl, labelsize=fss)
    ax.tick_params(axis='both', pad=pd)
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(lw)
    # remove bottom states
    dsize = np.zeros((Nx, 2 * N_slab))
    for idslab in range(2 * N_slab):
        for idx in range(Nx):
            if R[idx, idslab] > 0.3:
                Energy[idx, idslab] = 10
            if L[idx, idslab] > 0.3:
                dsize[idx, idslab] = 1
    spectr = ax.scatter(kxrep, Energy, c=L[:, :] - R[:, :],
                        s=ps + 4 * dsize * ps,
                        cmap=blue_red1)
    ax.set_xlim(0, 1)
    ax.set_ylim(-2, 2)
    ax.set_xticks(xcoords)
    ax.set_yticks([-2, 0, 2])
    ax.set_xticklabels(['$-\pi$', '$0$', '$\pi$'], fontsize=fss)
    for xc in xcoords:
        ax.axvline(x=xc, color='k', linewidth=ps)
    # cbar = fig.colorbar(spectr, cax=colax)
    # cbar.ax.tick_params(labelsize=fss, width=lw, labelrotation=90)
    # for axis in ['top', 'bottom', 'left', 'right']:
    #     cbar.ax.spines[axis].set_linewidth(lw)
    plt.show()


m = 1
Nx = 300
N_slab = 100

Kx = np.linspace(-pi, pi, Nx)

H = slab_hamiltonian(Kx)

H = surf_hamiltonian(H)

# Calculate eigenvalues and eigenvectors of H-Hopf
[Energy, States] = np.linalg.eigh(H)

L, R = weight_surface(States)

plot_spectrum()
