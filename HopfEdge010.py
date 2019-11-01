"""
 Created by alexandra at 29.08.19 12:57

 Surface states for s finite slab in y direction (010)
"""

import numpy as np
from math import pi, sqrt
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import LogNorm


def scalarprod(a, b):
    # Scalar product of two stackes of wavefunctions of the same size
    # Returns a stack of <A[i,j,...,:]| B[i,j,...,:]>
    prod = np.sum(np.multiply(np.conj(a), b), axis=-1)
    return prod


def berry_curvature(u):
    """Calculate Berry curvature of a band"""
    u = np.reshape(u, (Nx, Nx, -1))
    berry = np.empty((Nx-1, Nx-1))
    for idx in range(Nx - 1):
        for idy in range(Nx - 1):
            u1 = scalarprod(u[idx, idy, :], u[idx + 1, idy, :])
            u2 = scalarprod(u[idx + 1, idy, :], u[idx + 1, idy + 1, :])
            u3 = scalarprod(u[idx + 1, idy + 1, :], u[idx, idy + 1, :])
            u4 = scalarprod(u[idx, idy + 1, :], u[idx, idy, :])
            berry[idx, idy] = -np.angle(u1 * u2 * u3 * u4)
    c = np.sum(berry) / 2 / pi
    return berry, c


def chern_number_many_band(u):
    """Calculate the Chern number of a set of bands"""
    u = np.reshape(u, (Nx, Nx, N_slab * 2, -1))
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
    u = np.reshape(u, (Nx, Nx, -1))
    c = 0
    for idx in range(Nx - 1):
        for idy in range(Nx - 1):
            u1 = scalarprod(u[idx, idy, :], u[idx + 1, idy, :])
            u2 = scalarprod(u[idx + 1, idy, :], u[idx + 1, idy + 1, :])
            u3 = scalarprod(u[idx + 1, idy + 1, :], u[idx, idy + 1, :])
            u4 = scalarprod(u[idx, idy + 1, :], u[idx, idy, :])
            c = c - np.angle(u1 * u2 * u3 * u4) / 2 / pi
    return c


def gap_size(e1, e2, eps):
    """Check the size of the gap between two bands"""
    e1 = np.reshape(e1, (Nx, Nx))
    e2 = np.reshape(e2, (Nx, Nx))
    kx = np.reshape(Kx, (Nx, Nx))
    kz = np.reshape(Kz, (Nx, Nx))
    for idx in range(Nx):
        for idy in range(Nx):
            if np.abs(e1[idx, idy] - e2[idx, idy]) < eps:
                print('kx=', kx[idx, idy])
                print('ky=', kz[idx, idy])


def parallel_transport(uk):
    """parallel transport in one direction"""
    usmooth = np.empty(uk.shape, dtype=complex)
    usmooth[0, :] = uk[0, :]
    # make parallel transport in one direction
    nk_lim = uk.shape[0]
    for nk in range(0, nk_lim - 1):
        mold = scalarprod(usmooth[nk, :], uk[nk + 1, :])
        usmooth[nk + 1, :] = uk[nk + 1, :] * np.exp(-1j * np.angle(mold))

    # The function gains the multiplier
    lamb = scalarprod(usmooth[0, :], usmooth[nk_lim - 1, :])

    nxs = np.linspace(0, nk_lim - 1, nk_lim)
    # Distribute the multiplier among functions at kx in [0, 2pi]
    usmooth = np.multiply(usmooth,
                          np.power(lamb, - nxs[:, np.newaxis] / (nk_lim - 1)))
    return usmooth


def flow_hyb_wcc(u):
    """Calculate flow of hybrid wannier function at first diraction
    along the second direction"""
    u = np.reshape(u, (Nx, Nx, -1))
    wcc = np.zeros(Nx)
    for ind in range(Nx):
        usmooth = parallel_transport(u[:, ind, :])
        psi = usmooth[0, :]
        for nkz in range(1, Nx):
            psi_step = usmooth[nkz, :]
            wcc[ind] += np.angle(np.sum(np.conj(psi) * psi_step))
            psi = psi_step
    return -wcc / 2 / pi


def slab_hamiltonian(kx, kz):
    """Construct a z-slab hamiltonian for kx,ky wavevectors
    (always written in a 1d array)"""
    ntotal = len(kx)
    hh = np.zeros((ntotal, 2 * N_slab, 2 * N_slab), dtype=complex)
    # Construct blockes for Hopf Hamiltonian
    a = (np.power(np.sin(kx), 2) - np.power(np.sin(kz), 2) -
         np.power(np.cos(kx) + np.cos(kz) + h, 2))
    b = - np.cos(kx) - np.cos(kz) - h
    c = -1 / 2 * np.ones(ntotal)
    d = 2 * (np.sin(kx) * np.sin(kz)
             - 1j * np.sin(kx) * (np.cos(kx) + np.cos(kz) + h))
    f = np.sin(kz) - 1j * (np.cos(kx) + np.cos(kz) + h)
    g = -1j * np.sin(kx)
    j = -1j * 1 / 2 * np.ones(ntotal)

    e = np.stack((np.stack((a, np.conj(d)), axis=-1),
                  np.stack((d, -a), axis=-1)), axis=-1)

    delta = np.stack((np.stack((b, -np.conj(f) + np.conj(g)), axis=-1),
                      np.stack((f + g, -b), axis=-1)), axis=-1)

    n = np.stack((np.stack((c, -np.conj(j)), axis=-1),
                  np.stack((j, -c), axis=-1)), axis=-1)

    # Construct Hamiltonian for all Nz sites from these blockes
    hh[:, 0:2, 0:2] = e
    hh[:, 0:2, 2:4] = delta
    hh[:, 2:4, 0:2] = np.transpose(np.conj(delta), (0, 2, 1))
    hh[:, 2:4, 2:4] = e
    for nz in range(0, N_slab - 2):
        hh[:, 2 * nz + 4: 2 * nz + 6, 2 * nz + 4: 2 * nz + 6] = e
        hh[:, 2 * nz + 2: 2 * nz + 4, 2 * nz + 4: 2 * nz + 6] = delta
        hh[:, 2 * nz + 4: 2 * nz + 6, 2 * nz + 2: 2 * nz + 4] = (
            np.transpose(np.conj(delta), (0, 2, 1)))
        hh[:, 2 * nz: 2 * nz + 2, 2 * nz + 4: 2 * nz + 6] = n
        hh[:, 2 * nz + 4: 2 * nz + 6, 2 * nz: 2 * nz + 2] = (
            np.transpose(np.conj(n), (0, 2, 1)))
    return hh


def perturb_hamiltonian(kx, kz):
    """Construct perturbation potential to violate the symmetries"""
    ntotal = len(kx)
    hh = np.zeros((ntotal, 2 * N_slab, 2 * N_slab), dtype=complex)
    a = -((0.7 + 0.3 * np.sin(kx) + 1.4 * np.sin(kz)
           + 1.2 * np.cos(kx) + 0.8 * np.cos(kz))
          * (2.1 + 0.8 * np.sin(kx) + 1.1 * np.sin(kz)
             + 0.5 * np.cos(kx) + 1.8 * np.cos(kz)))
    b = ((2 + 0.8 * np.sin(kx) + 1.0 * np.sin(kz)
          + 0.9 * np.cos(kx) + 1.1 * np.cos(kz))
         * (0.4 + 0.5 * np.sin(kx) + 0.1 * np.sin(kz)
            + 1.5 * np.cos(kx) + 0.8 * np.cos(kz)))
    c = -((1.1 + 0.9 * np.sin(kx) + 0.9 * np.sin(kz)
           + 0.3 * np.cos(kx) + 1.5 * np.cos(kz))
          * (1.2 + 0.3 * np.sin(kx) + 0.4 * np.sin(kz)
             + 1.1 * np.cos(kx) + 1.2 * np.cos(kz)))
    d = ((0.2 + 0.9j + 0.9 * np.sin(kx) + 1.8 * np.sin(kz)
          + 0.3 * np.cos(kx) + 0.2 * np.cos(kz))
         * (1.1 + 1.5 * np.sin(kx) + 2.0 * np.sin(kz)
            + 0.9 * np.cos(kx) + 0.6 * np.cos(kz)))
    f = ((1.5 + 1.5j + 1.2 * np.sin(kx) + 1.1 * np.sin(kz)
          + 0.6 * np.cos(kx) + 0.3 * np.cos(kz))
         * (1.8 + 0.7 * np.sin(kx) + 2.2 * np.sin(kz)
            + 1.2 * np.cos(kx) + 0.3 * np.cos(kz)))
    g = -((0.2 + 0.3j + 1.0 * np.sin(kx) + 1.4 * np.sin(kz)
           + 0.5 * np.cos(kx) + 1.5 * np.cos(kz))
          * (1.8 + 1.1 * np.sin(kx) + 1.2 * np.sin(kz)
             + 0.6 * np.cos(kx) + 0.2 * np.cos(kz)))
    j = ((0.6 + 1.1j + 1.1 * np.sin(kx) + 1.2 * np.sin(kz)
          + 0.6 * np.cos(kx) + 1.5 * np.cos(kz))
         * (1.1 + 1.0 * np.sin(kx) + 0.2 * np.sin(kz)
            + 0.7 * np.cos(kx) + 1.5 * np.cos(kz)))
    e = np.stack((np.stack((a, np.conj(d)), axis=-1),
                  np.stack((d, -a), axis=-1)), axis=-1)

    delta = np.stack((np.stack((b, -np.conj(f) + np.conj(g)), axis=-1),
                      np.stack((f + g, -b), axis=-1)), axis=-1)

    n = np.stack((np.stack((c, -np.conj(j)), axis=-1),
                  np.stack((j, -c), axis=-1)), axis=-1)

    # Construct Hamiltonian for all Nz sites from these blockes
    hh[:, 0:2, 0:2] = e
    hh[:, 0:2, 2:4] = delta
    hh[:, 2:4, 0:2] = np.transpose(np.conj(delta), (0, 2, 1))
    hh[:, 2:4, 2:4] = e
    for nz in range(0, N_slab - 2):
        hh[:, 2 * nz + 4: 2 * nz + 6, 2 * nz + 4: 2 * nz + 6] = e
        hh[:, 2 * nz + 2: 2 * nz + 4, 2 * nz + 4: 2 * nz + 6] = delta
        hh[:, 2 * nz + 4: 2 * nz + 6, 2 * nz + 2: 2 * nz + 4] = (
            np.transpose(np.conj(delta), (0, 2, 1)))
        hh[:, 2 * nz: 2 * nz + 2, 2 * nz + 4: 2 * nz + 6] = n
        hh[:, 2 * nz + 4: 2 * nz + 6, 2 * nz: 2 * nz + 2] = (
            np.transpose(np.conj(n), (0, 2, 1)))
    return hh


def surf_hamiltonian(hh):
    """Additional surface potential for the slab hamiltonian h"""
    ntotal = hh.shape[0]
    # Surface potential
    alpha = 0
    beta = 0
    gamma = 0
    delta = 1
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


    # Disentangle surface states
    hh[:, 0:2, 0:2] /= 16
    hh[:, 2:4, 0:2] /= 4
    hh[:, 0:2, 2:4] /= 4
    # hh[:, -4:-2, -2:] /= 2
    # hh[:, -2:, -4:-2] /= 2
    # hh[:, -2:, -2:] /= 8

    # Add surface potential
    hh[:, 0:2, 0:2] += 0.15 * hsurf
    # hh[:, 2:4, 2:4] += hsurf
    # hh[:, 4:6, 4:6] += hsurf
    hh[:, -4:-2, -4:-2] += -1.5 * hsurf
    hh[:, -2:, -2:] += hsurf

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
    # print('Llow=', l[round(Nx * sqrt(2)), N_slab - 1])
    # print('Lhigh=', l[round(Nx * sqrt(2)), N_slab])
    # print(L[round(Nx * sqrt(2)), :])
    # print(Energy[round(Nx * sqrt(2)), :])
    # Right eigenstate
    r = np.sum(np.multiply(np.power(np.abs(states), 2),
                           np.flip(weight[np.newaxis, :, np.newaxis], axis=-2)),
               axis=-2)
    # print('Rlow=', r[round(Nx * sqrt(2)), N_slab - 1])
    # print('Rhigh=', r[round(Nx * sqrt(2)), N_slab])
    return l, r


def plot_spectrum():
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

    cdict2 = {'red': ((0.0, 0.0, 0.1),
                      (1.0, 1.0, 0.0)),

              'green': ((0.0, 0.0, 0.0),
                        (1.0, 0.0, 0.0)),

              'blue': ((0.0, 0.0, 0.0),
                       (1.0, 0.0, 0.0))
              }

    cdict3 = {'red': ((0.0, 0.0, 0.0),
                      (1.0, 0.0, 0.0)),

              'green': ((0.0, 0.0, 0.0),
                        (1.0, 0.0, 0.0)),

              'blue': ((0.0, 0.0, 0.1),
                       (1.0, 1.0, 0.0))
              }

    blue_red1 = LinearSegmentedColormap('BlueRed1', cdict1)
    red2 = LinearSegmentedColormap('BlueRed1', cdict2)
    blue3 = LinearSegmentedColormap('BlueRed1', cdict3)

    kxplot = np.append(np.linspace(0, 1, round(Nx * sqrt(2))) * sqrt(2),
                       np.linspace(0, 1, Nx) + sqrt(2))
    kxplot = np.append(kxplot, np.linspace(0, 1, Nx) + sqrt(2) + 1)
    kxrep = np.transpose(np.tile(kxplot, (2 * N_slab, 1)))
    xcoords = [0, sqrt(2), 1 + sqrt(2), 2 + sqrt(2)]

    # Plot the spectrum
    # Size foe 'Energy' label
    # fig = plt.figure(figsize=(1.78, 1.5))
    # ax = fig.add_axes([0.22, 0.15, 0.7303, 0.83])
    # Size without labels
    # fig = plt.figure(figsize=(1.68, 1.5))
    # ax = fig.add_axes([0.2024, 0.15, 0.7738, 0.83])
    # Size with colorbar
    # fig = plt.figure(figsize=(2, 1.5))
    # ax = fig.add_axes([0.17, 0.15, 0.65, 0.83])
    # colax = fig.add_axes([0.85, 0.15, 0.05, 0.83])
    # fs = 10
    # fss = 8
    # lw = 1.2
    # ps = 0.004

    # Size for poster
    fig = plt.figure(figsize=(10, 9))
    ax = fig.add_axes([0.13, 0.1, 0.85, 0.85])
    fs = 40
    fss = 35
    lw = 3.5
    ps = 3.5
    tl = 10
    # ax.set_ylabel('Energy', fontsize=fs)
    ax.yaxis.set_label_coords(-0.15, 0.5)
    ax.tick_params(width=lw, length=tl, labelsize=fss)
    ax.tick_params(axis='both', pad=20)
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(lw)
    spectr = ax.scatter(kxrep, Energy, c=L[:, :] - R[:, :], s=ps,
                        cmap=blue_red1)
    ax.set_xlim(0, 2 + sqrt(2))
    ax.set_ylim(-3, 3)
    ax.set_xticks(xcoords)
    ax.set_xticklabels(['$\Gamma$', 'M', 'K', '$\Gamma$'], fontsize=fs)
    for xc in xcoords:
        ax.axvline(x=xc, color='k', linewidth=ps)
    cbar = fig.colorbar(spectr)  # , cax=colax
    # cbar.ax.tick_params(labelsize=fss, width=lw, labelrotation=90)
    # for axis in ['top', 'bottom', 'left', 'right']:
    #     cbar.ax.spines[axis].set_linewidth(lw)
    plt.show()
    # plt.savefig('Images/Spectra/Spectrum_h05t1alpha15_big.png', bbox_inches=None)


def plot_spectrum_line():
    """Plot spectrum along one line in BZ"""
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

    fig = plt.figure(figsize=(10, 9))
    ax = fig.add_axes([0.13, 0.1, 0.85, 0.85])
    fs = 40
    fss = 35
    lw = 3.5
    ps = 3.5
    tl = 10
    ax.tick_params(width=lw, length=tl, labelsize=fss)
    ax.tick_params(axis='both', pad=20)
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(lw)

    spectr = ax.scatter(kxrep, Energy, c=L[:, :] - R[:, :], s=ps,
                        cmap=blue_red1)

    ax.set_xlim(0, 1)
    ax.set_ylim(-3, 3)
    cbar = fig.colorbar(spectr)
    plt.show()


# Parameters
h = 0.5
t = 1

Nx = 101
N_slab = 16

Kx = np.append(np.linspace(0, pi, round(Nx * sqrt(2))), np.ones(Nx) * pi)
Kx = np.append(Kx, np.linspace(pi, 0, Nx))
Kz = np.append(np.linspace(0, pi, round(Nx * sqrt(2))), np.linspace(pi, 0, Nx))
Kz = np.append(Kz, np.zeros(Nx))

# Kx = np.linspace(-pi, pi, Nx)
# Kz = np.linspace(0, 2 * pi, Nx)

# Kx, Kz = np.meshgrid(np.linspace(0, 2 * pi, Nx), np.linspace(0, 2 * pi, Nx),
#                      indexing='ij')
# # Kx, Kz = np.meshgrid(np.linspace(2.6, 2.8, Nx), np.linspace(3.1, 3.3, Nx),
# #                      indexing='ij')
# Kx = np.reshape(Kx, -1)
# Kz = np.reshape(Kz, -1)

H = slab_hamiltonian(Kx, Kz)

# H_perturb = 0.08 * perturb_hamiltonian(Kx, Kz)

# H += H_perturb

# H = surf_hamiltonian(H)

# Calculate eigenvalues and eigenvectors of H-Hopf
[Energy, States] = np.linalg.eigh(H)
print(States.shape)

L, R = weight_surface(States)


# Csum = 0
# for ind in range(N_slab * 2):
#     C = chern_number(States[:, :, ind])
#     Csum += C
#     print(ind, C)
# print(Csum)
# C = chern_number(States[:, :, N_slab - 1])
# print('C_lower', C)
# D = chern_number(States[:, :, N_slab])
# print('C_upper', D)
# C = chern_number_many_band(States[:, :, :N_slab - 2])
# print('C_occ_bulk=', C)
# C = chern_number_many_band(States[:, :, N_slab + 2:])
# print('C_cond_bulk=', C)
# C = chern_number_many_band(States[:, :, N_slab - 2:N_slab + 2])
# print('C_surface=', C)
# C = chern_number_many_band(States)
# print('C_all=', C)

# Berry curvature on the surface
# Berry_lower, C_lower = berry_curvature(States[:, :, N_slab - 1])
# Berry_upper, C_upper = berry_curvature(States[:, :, N_slab])
# print('C_lower=', C_lower)
# print('C_upper=', C_upper)
# fig, ax = plt.subplots(1, 2)
# fig.subplots_adjust(hspace=0.3)
# im1 = ax[0].imshow(Berry_lower) #, norm=LogNorm(vmin=0.01, vmax=1))
# fig.colorbar(im1, ax=ax[0])
# im2 = ax[1].imshow(Berry_upper) #, norm=LogNorm(vmin=0.01, vmax=1))
# fig.colorbar(im2, ax=ax[1])
# plt.show()

# Hybrid WCC flow along second axis
# WCC_lower = flow_hyb_wcc(States[:, :, N_slab - 1])
# WCC_upper = flow_hyb_wcc(States[:, :, N_slab])
# fig, ax = plt.subplots(1, 2)
# im1 = ax[0].plot(np.linspace(0, 2 * pi, Nx), WCC_lower, 'ro')
# im2 = ax[1].plot(np.linspace(0, 2 * pi, Nx), WCC_upper, 'ro')
# plt.show()

# Gap size check
# print('Occ bulk and lower surface')
# gap_size(Energy[:, N_slab - 2], Energy[:, N_slab - 1], 0.05)
# print('Lower and upper surface')
# gap_size(Energy[:, N_slab - 1], Energy[:, N_slab], 0.05)
# print('Cond bulk and upper surface')
# gap_size(Energy[:, N_slab], Energy[:, N_slab + 1], 0.05)

# Plot the spectrum between Gamma, M, K, Gamma points
# plot_spectrum_line()
plot_spectrum()

