"""
 Created by alexandra at 26.12.18 14:41

 Calculate surface states for the Hopf slab with additional surface potential
 (Consider system infinite in x and y directions and finite in z direction)
"""

import numpy as np
from math import pi, sqrt
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from random import random as rnd


def scalarprod(a, b):
    # Scalar product of two vectors
    prod = np.sum(np.multiply(np.conj(a), b))  # , axis=-1
    return prod


def symm_c4z_eigenvalues(nk):
    """Calculate eigenvalues of symmetry operator C4z corresponding to
    surface eigenstate"""
    # ind_surf = np.argmin(np.abs(Energy[nk, :]))
    # print(ind_surf)
    ind_surf = N_slab
    # print(States[nk, :, ind_surf])
    surf_state = States[nk, :, ind_surf]
    eig_value = (
        np.sum(np.abs(surf_state[::2])**2)
        + 1j * np.sum(np.abs(surf_state[1::2])**2))
    return eig_value


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
            c += -np.angle(u1 * u2 * u3 * u4) / 2 / pi
    return c


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
        for nky in range(1, Nx):
            psi_step = usmooth[nky, :]
            wcc[ind] += np.angle(np.sum(np.conj(psi) * psi_step))
            psi = psi_step
    return -wcc / 2 / pi


def gap_size(e1, e2, eps):
    """Check the size of the gap between two bands"""
    e1 = np.reshape(e1, (Nx, Nx))
    e2 = np.reshape(e2, (Nx, Nx))
    kx = np.reshape(Kx, (Nx, Nx))
    ky = np.reshape(Ky, (Nx, Nx))
    for idx in range(Nx):
        for idy in range(Nx):
            if np.abs(e1[idx, idy] - e2[idx, idy]) < eps:
                print('kx=', kx[idx, idy])
                print('ky=', ky[idx, idy])


def slab_hamiltonian(kx, ky):
    """Construct a z-slab hamiltonian for kx,ky wavevectors
    (always written in a 1d array)"""
    ntotal = len(kx)
    hh = np.zeros((ntotal, 2 * N_slab, 2 * N_slab), dtype=complex)

    # Construct blockes for Hopf Hamiltonian
    # a = (np.power(np.sin(kx), 2) + t ** 2 * np.power(np.sin(ky), 2) -
    #      np.power(np.cos(kx) + np.cos(ky) + h, 2) - 1)
    # b = - np.cos(kx) - np.cos(ky) - h
    # c = 2 * np.multiply(t * np.sin(ky) - 1j * np.sin(kx),
    #                     np.cos(kx) + np.cos(ky) + h)
    # d = 2 * (t * np.sin(ky) - 1j * np.sin(kx))

    # PRL model
    a = (-np.power(np.sin(kx), 2) - np.power(np.sin(ky), 2)
         + 1 + np.power(np.cos(kx) + np.cos(ky) + m - 3, 2))
    b = np.cos(kx) + np.cos(ky) + m - 3
    c = -2 * (np.sin(ky) - 1j * np.sin(kx)) * (np.cos(kx) + np.cos(ky) + m - 3)
    d = -2 * (np.sin(ky) - 1j * np.sin(kx))

    # Compose onsite and hopping matrices
    e = np.stack((np.stack((a, np.conj(c)), axis=-1),
                  np.stack((c, -a), axis=-1)), axis=-1)

    delta = np.stack((np.stack((b, np.zeros(ntotal)), axis=-1),
                      np.stack((d, -b), axis=-1)), axis=-1)  # this gives
                                                             # [b d
                                                             #  0 -b]

    # Construct Hamiltonian for all Nz sites from these blockes
    hh[:, 0:2, 0:2] = e
    for nz in range(0, N_slab - 1):
        hh[:, 2 * nz + 2: 2 * nz + 4, 2 * nz + 2: 2 * nz + 4] = e
        hh[:, 2 * nz: 2 * nz + 2, 2 * nz + 2: 2 * nz + 4] = (
            np.transpose(np.conj(delta), (0, 2, 1)))
        hh[:, 2 * nz + 2: 2 * nz + 4, 2 * nz: 2 * nz + 2] = delta
    return hh


def slab_hamiltonian_2kz(kx, ky):
    """Construct a z-slab hamiltonian for kz->2kz for kx,ky wavevectors
    (always written in a 1d array)"""
    ntotal = len(kx)
    hh = np.zeros((ntotal, 2 * N_slab, 2 * N_slab), dtype=complex)

    # Construct blockes for Hopf Hamiltonian
    a = (np.power(np.sin(kx), 2) + t ** 2 * np.power(np.sin(ky), 2) -
         np.power(np.cos(kx) + np.cos(ky) + h, 2) - 1)
    b = - np.cos(kx) - np.cos(ky) - h
    c = 2 * np.multiply(t * np.sin(ky) - 1j * np.sin(kx),
                        np.cos(kx) + np.cos(ky) + h)
    d = 2 * (t * np.sin(ky) - 1j * np.sin(kx))

    e = np.stack((np.stack((a, np.conj(c)), axis=-1),
                  np.stack((c, -a), axis=-1)), axis=-1)

    delta = np.stack((np.stack((b, np.zeros(ntotal)), axis=-1),
                      np.stack((d, -b), axis=-1)), axis=-1)

    # Construct Hamiltonian for all Nz sites from these blockes
    hh[:, 0:2, 0:2] = e
    hh[:, 2:4, 2:4] = e
    for nz in range(0, N_slab - 2):
        hh[:, 2 * nz + 4: 2 * nz + 6, 2 * nz + 4: 2 * nz + 6] = e
        hh[:, 2 * nz: 2 * nz + 2, 2 * nz + 4: 2 * nz + 6] = (
            np.transpose(np.conj(delta), (0, 2, 1)))
        hh[:, 2 * nz + 4: 2 * nz + 6, 2 * nz: 2 * nz + 2] = delta
    return hh


def rnd_f(kx, ky):
    """Generate random function for perturbation"""
    return (rnd() * np.sin(kx) + rnd() * np.cos(kx)
            + rnd() * np.sin(ky) + rnd() * np.cos(ky)
            + rnd() * np.cos(kx) * np.sin(ky) + rnd() * np.cos(kx) * np.cos(ky)
            + rnd() * np.sin(kx) * np.cos(ky) + rnd() * np.sin(kx) * np.sin(kx))


def perturb_hamiltonian(kx, ky):
    """Construct perturbation potential to violate the symmetries"""
    ntotal = len(kx)
    hh = np.zeros((ntotal, 2 * N_slab, 2 * N_slab), dtype=complex)
    a = rnd_f(kx, ky)
    b = rnd_f(kx, ky)
    c = rnd_f(kx, ky) + 1j * rnd_f(kx, ky)
    d = rnd_f(kx, ky) + 1j * rnd_f(kx, ky)
    f = rnd_f(kx, ky) + 1j * rnd_f(kx, ky)
    e = np.stack((np.stack((a, np.conj(c)), axis=-1),
                  np.stack((c, -a), axis=-1)), axis=-1)

    delta = np.stack((np.stack((b, f), axis=-1),
                      np.stack((d, -b), axis=-1)), axis=-1)

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
    delta = -1.1
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
    hh[:, 0:2, 0:2] *= 0.6
    # hh[:, 0:6, 0:2] /= 2.0
    # hh[:, 0:2, 2:4] /= 2.0
    # hh[:, -4:-2, -4:-2] /=2.0
    hh[:, -2:, -2:] *= 0.21

    # Add surface potential
    hh[:, 0:2, 0:2] += hsurf
    # hh[:, 2:4, 2:4] += -hsurf
    # hh[:, 4:6, 4:6] += hsurf
    # hh[:, -4:-2, -4:-2] += hsurf
    # hh[:, -2:, -2:] += hsurf

    return hh


def projector_down_hamiltonian(u, alpha=1):
    """Define projection of the state u from down surface into the bulk
    and add it to the Hamiltonian"""
    p = np.conjugate(u[:, np.newaxis, :]) * u[:, :, np.newaxis]
    ppr = np.zeros((2 * N_slab, 2 * N_slab))
    ppr[0, 0] = 1
    ppr[1, 1] = 1
    ppr = np.tile(ppr[np.newaxis, :, :], (u.shape[0], 1, 1))
    return alpha * np.matmul(np.matmul(ppr, p), ppr)


def projector_up_hamiltonian(u, alpha=1):
    """Define projection of the state u from up surface into the bulk
    and add it to the Hamiltonian"""
    p = np.conjugate(u[:, np.newaxis, :]) * u[:, :, np.newaxis]
    ppr = np.zeros((2 * N_slab, 2 * N_slab))
    ppr[-1, -1] = 1
    ppr[-2, -2] = 1
    ppr = np.tile(ppr[np.newaxis, :, :], (u.shape[0], 1, 1))
    return alpha * np.matmul(np.matmul(ppr, p), ppr)


def projector_occ(u_occ):
    """Define the projector on the occupied states"""
    # x = u_occ[:, :, np.newaxis, :] * np.conjugate(u_occ[:, :, :, np.newaxis])
    # print(x.shape)
    # p = np.sum(x, axis=-3)
    p = np.matmul(u_occ, np.transpose(np.conjugate(u_occ), (0, 2, 1)))
    nk = 200
    vect = np.random.rand(2 * N_slab)
    vect = vect / sqrt(np.sum(np.power(vect, 2)))
    # print(np.matmul(p[nk, :, :], States[nk, :, 18]) - States[nk, :, 18])
    # print(p.shape)
    # print(np.max(np.abs(p - q)))
    return p


def pzp(states):
    """Define PzP operator where P is projector on occupied and surface states
    and z is position operator in a slab"""

    p = projector_occ(states)
    z = np.stack((np.arange(N_slab), np.arange(N_slab)), axis=-1)
    z = np.reshape(z, 2 * N_slab, order='C') + 1
    z = np.diag(z)
    z = z[np.newaxis, :, :]
    z = np.tile(z, (p.shape[0], 1, 1))
    # print(z.shape, z)
    return np.matmul(p, np.matmul(z, p))


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
    # print('L_14=', l[round(Nx * sqrt(2)) + 50, 14])
    # print('L_15=', l[round(Nx * sqrt(2)) + 50, 15])
    # print(L[round(Nx * sqrt(2)), :])
    # print(Energy[round(Nx * sqrt(2)), :])
    # Right eigenstate
    r = np.sum(np.multiply(np.power(np.abs(states), 2),
                           np.flip(weight[np.newaxis, :, np.newaxis], axis=-2)),
               axis=-2)
    # print('R_14=', r[round(Nx * sqrt(2)) + 50, 14])
    # print('R_15=', r[round(Nx * sqrt(2)) + 50, 15])
    return l, r


def spin_expectation_value(state):
    """Calculate the expectation value of sigma_z operator at the eigenstates
    state for all points in BZ"""
    sigma_z = np.stack((np.ones(N_slab), -np.ones(N_slab)), axis=-1)
    sigma_z = np.reshape(sigma_z, 2 * N_slab, order='C')
    sigma_z = np.diag(sigma_z)
    sigma_z = sigma_z[np.newaxis, :, :]
    sigma_z = np.tile(sigma_z, (state.shape[0], 1, 1))
    print('sigma made')
    print(state.shape)
    print(sigma_z.shape)
    expect = np.matmul(np.conj(np.transpose(state, (0, 2, 1))),
                       np.matmul(sigma_z, state))
    print(expect.shape)
    print(np.max(np.abs(expect - np.conjugate(expect))))
    return expect


def plot_spectrum():
    """Plot spectrum of hamiltonian between Gamma, M, K, Gamma points"""
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

    cdict2 = {'red': ((0.0, 0.0, 0.0),
                      (1.0, 1.0, 1.0)),

              'green': ((0.0, 0.0, 0.0),
                        (1.0, 0.0, 0.0)),

              'blue': ((0.0, 0.0, 0.0),
                       (1.0, 0.0, 0.0)),

              }

    blue_red1 = LinearSegmentedColormap('BlueRed1', cdict1)
    red1 = LinearSegmentedColormap('Red1', cdict2)

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
    fig = plt.figure(figsize=(4, 3))
    ax = fig.add_axes([0.12, 0.15, 0.85, 0.8])
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
    # ax.set_ylabel('Energy', fontsize=fs)
    ax.yaxis.set_label_coords(-0.15, 0.5)
    ax.tick_params(width=lw, length=tl, labelsize=fss)
    ax.tick_params(axis='both', pad=pd)
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(lw)
    # remove bottom states
    for idslab in range(2 * N_slab):
        for idx in range(2 * Nx + round(Nx * sqrt(2))):
            if R[idx, idslab] > 0.3:
                Energy[idx, idslab] = 10

    # for idslab in range(2 * N_slab):
    #     if idslab != N_slab - 1:
    #         L[:, idslab] = 0
    #         R[:, idslab] = 0
    spectr = ax.scatter(kxrep, Energy, c=L[:, :] - R[:, :], s=ps,
                        cmap=red1)  # blue_
    ax.set_xlim(0, 2 + sqrt(2))
    ax.set_ylim(-2.5, 2.5)
    ax.set_xticks(xcoords)
    ax.set_yticks([-2, 0, 2])
    ax.set_xticklabels(['$\Gamma$', 'M', 'X', '$\Gamma$'], fontsize=fs)
    for xc in xcoords:
        ax.axvline(x=xc, color='k', linewidth=lw)
    # cbar = fig.colorbar(spectr, cax=colax)
    # cbar.ax.tick_params(labelsize=fss, width=lw, labelrotation=90)
    # for axis in ['top', 'bottom', 'left', 'right']:
    #     cbar.ax.spines[axis].set_linewidth(lw)
    plt.show()
    # plt.savefig('Images/Spectra/Spectrum_h05t1alpha15_big.png', bbox_inches=None)


def plot_spectrum_MGXM():
    """Plot the spectrum between points MGXM"""
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

    cdict2 = {'red': ((0.0, 0.0, 0.0),
                      (1.0, 1.0, 1.0)),

              'green': ((0.0, 0.0, 0.0),
                        (1.0, 0.0, 0.0)),

              'blue': ((0.0, 0.0, 0.0),
                       (1.0, 0.0, 0.0)),

              }

    blue_red1 = LinearSegmentedColormap('BlueRed1', cdict1)
    red1 = LinearSegmentedColormap('Red1', cdict2)

    kxplot = np.append(np.linspace(0, 1, round(Nx * sqrt(2))) * sqrt(2),
                       np.linspace(0, 1, Nx) + sqrt(2))
    kxplot = np.append(kxplot, np.linspace(0, 1, Nx) + sqrt(2) + 1)
    kxrep = np.transpose(np.tile(kxplot, (2 * N_slab, 1)))
    xcoords = [0, sqrt(2), 1 + sqrt(2), 2 + sqrt(2)]

    # Plot the spectrum
    fig = plt.figure(figsize=(4, 3))
    ax = fig.add_axes([0.16, 0.15, 0.79, 0.82])
    fs = 30
    fss = 27
    lw = 1.2
    ps = 0.02
    tl = 3
    pd = 4

    ax.yaxis.set_label_coords(-0.15, 0.5)
    ax.tick_params(width=lw, length=tl, labelsize=fss)
    ax.tick_params(axis='both', pad=pd)
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(lw)
    # remove bottom states
    if removebottom == 1:
        for idslab in range(2 * N_slab):
            for idx in range(2 * Nx + round(Nx * sqrt(2))):
                if L[idx, idslab] > 0.3:
                    Energy[idx, idslab] = 10

    if addsurfpot == 1:
        for idslab in range(2 * N_slab):
            if idslab != N_slab:
                L[:, idslab] = 0
                R[:, idslab] = 0

    # Choose cmap
    if addsurfpot == 1:
        colmap = red1
    else:
        colmap = blue_red1

    # dot size
    dsize = np.zeros((2 * Nx + round(Nx * sqrt(2)), 2 * N_slab))
    for idslab in range(2 * N_slab):
        for idx in range(2 * Nx + round(Nx * sqrt(2))):
                if R[idx, idslab] > 0.3:
                    dsize[idx, idslab] = 1
    spectr = ax.scatter(kxrep, Energy, c=R[:, :] - L[:, :],
                        s=ps + 4 * dsize * ps,
                        cmap=colmap)  #
    # spectr = ax.scatter(kxrep, np.real(zbar),
    #                     s=ps)
    ax.set_xlim(0, 2 + sqrt(2))
    ax.set_ylim(-3, 3)
    ax.set_xticks(xcoords)
    ax.set_yticks([-2, 0, 2])
    ax.set_xticklabels(['M', '$\Gamma$', 'X', 'M'], fontsize=fs)
    for xc in xcoords:
        ax.axvline(x=xc, color='k', linewidth=lw)
    plt.show()


def slab_plot_MGXM():
    """Plot WCC along MGXM for a slab with finite number of layers"""
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
    kxplot = np.append(np.linspace(0, 1, round(Nx * sqrt(2))) * sqrt(2),
                       np.linspace(0, 1, Nx) + sqrt(2))
    kxplot = np.append(kxplot, np.linspace(0, 1, Nx) + sqrt(2) + 1)
    kxrep = np.transpose(np.tile(kxplot, (N_layers, 1)))
    print(kxrep.shape)

    # Sizes for paper
    fig = plt.figure(figsize=[3.2, 3])
    ax = fig.add_axes([0.15, 0.13, 0.8, 0.83])
    fs = 25
    fss = 22
    lss = 1.4
    ls = 2.
    ps = 0.5

    ax.yaxis.set_label_coords(-0.03, 0.5)

    x_ticks = [0, sqrt(2), sqrt(2) + 1, sqrt(2) + 2]
    ax.tick_params(labelsize=fss, width=lss)
    ax.tick_params(axis='x', labelsize=fs)
    ax.tick_params(axis='y', pad=3)
    ax.set_xlim(0, sqrt(2) + 2)
    # ax.set_ylim(0.5, N_layers + 0.5)
    # y_ticks = [i + 1 for i in range(N_layers)]

    # ax.set_yticks(y_ticks)
    # ax.set_yticklabels([r'1', r'2', r'3', r'4'])
    # ax.set_yticklabels([])

    ax.set_xticks(x_ticks)
    ax.set_xticklabels([r'$M$', '$\Gamma$', r'$X$', r'$M$'])

    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(lss)
    # settings = {'marker': '.', 'linewidth': ls, 'markersize': ps}
    spectr = ax.plot(kxplot, np.real(zbar[:, N_slab:N_slab+5]), color='black',
                     marker='.', markersize=ps,
                     linestyle='None', linewidth=lss,
                     markeredgecolor='black')
    # spectr = ax.scatter(kxplot, zbar[:, N_slab], sigma_z_expect[:, 0, 0],
    #                     cmap=blue_red1)
    # for ind in range(N_layers):
        # ax.plot(kxplot, WCCz_m1 + ind, color='black',
        #         linestyle='solid', linewidth=lss)
        # ax.scatter(x_ticks, np.ones(len(x_ticks)) * (ind + 1),
        #            s=120, facecolors='b', edgecolors='b')
    # ax.scatter(sqrt(2), 4, s=120, facecolors='none', edgecolors='b')
    # ax.plot([0, 2 + sqrt(2)], [-0.5, -0.5], 'k--')
    # ax.plot([0, 2 + sqrt(2)], [3.5, 3.5], 'k--')
    for xc in x_ticks:
        ax.axvline(x=xc, color='k', linewidth=lss)
    plt.show()
    # plt.savefig('Images/Polarization/Slab/WCC_low_slab5_m-10.png',
    #             transparent=True)


# What we calculate:
calcPzPspec = 0  # Plot spectrum of PzP operator
calcPzPchern = 0  # Calculate Chern number of the eigenstate of PzP operator
calcspec = 1  # Plot spectrum of the Hamiltonian
addsurfpot = 1  # Add surface potential to hamiltonian
addrandpot = 0  # Add random potential on the surface
calcchern = 0  # Calculare Chern number of the Hamiltonian
statesremove = 1  # Remove unused surface states by projecting them on the bulk
removebottom = 0  # Remove bottom surface states from the spectrum plot
# by seting all L>0.3 to big value

# Parameters
h = 1.5
t = 1
m = 1

Nx = 300
N_slab = 100

# TODO
# np.block
# np.kron
# remove np.power
# ...
# np.ravel

# GMXG
# Kx = np.append(np.linspace(0, pi, round(Nx * sqrt(2))), np.ones(Nx) * pi)
# Kx = np.append(Kx, np.linspace(pi, 0, Nx))
# Ky = np.append(np.linspace(0, pi, round(Nx * sqrt(2))), np.linspace(pi, 0, Nx))
# Ky = np.append(Ky, np.zeros(Nx))

if calcPzPspec == 1 or calcspec == 1:
    # MGXM
    Kx = np.append(np.linspace(pi, 0, round(Nx * sqrt(2))), np.linspace(0, pi, Nx))
    Kx = np.append(Kx, np.ones(Nx) * pi)
    Ky = np.append(np.linspace(pi, 0, round(Nx * sqrt(2))), np.zeros(Nx))
    Ky = np.append(Ky, np.linspace(0, pi, Nx))
# Ntotal = 2 * Nx + round(Nx * sqrt(2))
#
if calcPzPchern == 1 or calcchern == 1:
    Kx, Ky = np.meshgrid(np.linspace(0, 2 * pi, Nx), np.linspace(0, 2 * pi, Nx),
                         indexing='ij')
    Kx = np.reshape(Kx, -1)
    Ky = np.reshape(Ky, -1)

# Constract the Hamiltonian of the slab in z direction
H = slab_hamiltonian(Kx, Ky)

if addrandpot == 1:
    H_perturb = 0.1 * perturb_hamiltonian(Kx, Ky)
    H += H_perturb

if addsurfpot == 1:
    H = surf_hamiltonian(H)

# Calculate eigenvalues and eigenvectors of H-Hopf
[Energy, States] = np.linalg.eigh(H)
print(Energy.shape)

# Calculate surface weights of eigenstates
L, R = weight_surface(States)


if statesremove == 1:
    H = (H
         + projector_up_hamiltonian(States[:, :, N_slab - 1], -3.5)
         + projector_up_hamiltonian(States[:, :, N_slab + 1], 0.77)
         # - projector_up_hamiltonian(States[:, :, N_slab - 3], 1)
         + projector_down_hamiltonian(States[:, :, N_slab - 1], -0.5)
         + projector_down_hamiltonian(States[:, :, N_slab + 1], 1.4))
         # - projector_down_hamiltonian(States[:, :, N_slab - 3], 3))

    [Energy, States] = np.linalg.eigh(H)


# Calculate PzP operator  - position operator projected on set of occupied
# or empty states
if calcPzPchern == 1 or calcPzPspec == 1:
    nstart = 0
    nfinish = N_slab
    PzP = pzp(States[:, :, nstart:nfinish])
    QzQ = pzp(States[:, :, nfinish:])

    zbar, zstates = np.linalg.eigh(PzP)
    # zbar, zstates = np.linalg.eigh(QzQ)
    sigma_z_expect = spin_expectation_value(zstates[:, :, N_slab:2*N_slab])

# TODO check sortnig algorithm
# for ind in range(zbar.shape[0]):
#     index = np.argsort(zbar[ind, :], axis=-1)
#     zbar[ind, :] = zbar[ind, index]
#     zstates[ind, :, :] = zstates[ind, :, index]
# zstates = np.reshape(zstates, (Nx, Nx, 2 * N_slab, 2 * N_slab))
# a = zstates[36, 17, :, N_slab]

# Calculate or plot
if calcPzPchern == 1:
    C = chern_number_many_band(zstates[:, :, N_slab:N_slab + 3])
    # C = chern_number(zstates[:, :, N_slab])
    print(C)
if calcPzPspec == 1:
    N_layers = N_slab
    slab_plot_MGXM()

if calcspec == 1:
    plot_spectrum_MGXM()

if calcchern == 1:
    # TODO organize Chern calculation
    C = chern_number_many_band(zstates[:, :, :])
    D = flow_hyb_wcc(zstates[:, :, 16])
    print(C)
    plt.figure()
    plt.plot(np.linspace(0, 2 * pi, Nx), D)
    plt.show()

# Csum = 0
# for ind in range(N_slab * 2):
#     C = chern_number(States[:, :, ind])
#     Csum += C
#     print(ind, C)
# # print(Csum)
# C = chern_number_many_band(States[:, :, :N_slab])
# D = chern_number(States[:, :, N_slab])
# print(D)
# # print(C, D)
# print(C)
#
#
# # Hybrid WCC flow along second axis
# WCC_lower = flow_hyb_wcc(States[:, :, N_slab - 2])
# WCC_upper = flow_hyb_wcc(States[:, :, N_slab - 1])
# fig, ax = plt.subplots(1, 2)
# im1 = ax[0].plot(np.linspace(0, 2 * pi, Nx), WCC_lower, 'ro')
# im2 = ax[1].plot(np.linspace(0, 2 * pi, Nx), WCC_upper, 'ro')
# plt.show()

# Gap size check
# print('Occ bulk and lower surface')
# gap_size(Energy[:, N_slab - 3], Energy[:, N_slab - 2], 0.05)
# # print('Lower and upper surface')
# # gap_size(Energy[:, N_slab - 1], Energy[:, N_slab], 0.05)
# print('Cond bulk and upper surface')
# gap_size(Energy[:, N_slab - 1], Energy[:, N_slab], 0.05)

# Calculate eigenstates of C4z at Gamma:
# print(symm_c4z_eigenvalues(round(Nx * sqrt(2))))  # round(Nx * sqrt(2))

# Plot the spectrum between Gamma, M, K, Gamma points

# C = chern_number_many_band(zstates[:, :, :])
# D = flow_hyb_wcc(zstates[:, :, 16])
# print(C)
# plt.figure()
# plt.plot(np.linspace(0, 2 * pi, Nx), D)
# plt.show()


