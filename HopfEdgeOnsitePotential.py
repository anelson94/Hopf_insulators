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


def chern_number(u, nx):
    """Calculate the Chern number of a band"""
    u = np.reshape(u, (nx, nx, -1))
    c = 0
    for idx in range(nx - 1):
        for idy in range(nx - 1):
            u1 = scalarprod(u[idx, idy, :], u[idx + 1, idy, :])
            u2 = scalarprod(u[idx + 1, idy, :], u[idx + 1, idy + 1, :])
            u3 = scalarprod(u[idx + 1, idy + 1, :], u[idx, idy + 1, :])
            u4 = scalarprod(u[idx, idy + 1, :], u[idx, idy, :])
            c += -np.angle(u1 * u2 * u3 * u4) / 2 / pi
    return c


def chern_number_many_band(u, nx, n_slab):
    """Calculate the Chern number of a set of bands"""
    u = np.reshape(u, (nx, nx, n_slab * 2, -1))
    print(u.shape)
    n_bands = u.shape[-1]
    c = 0
    for idx in range(nx - 1):
        for idy in range(nx - 1):
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


def flow_hyb_wcc(u, nx):
    """Calculate flow of hybrid wannier function at first diraction
    along the second direction"""
    u = np.reshape(u, (nx, nx, -1))
    wcc = np.zeros(nx)
    for ind in range(nx):
        usmooth = parallel_transport(u[:, ind, :])
        psi = usmooth[0, :]
        for nky in range(1, nx):
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


def slab_hamiltonian(m, t, kx, ky, n_slab, ham_model):
    """Construct a z-slab hamiltonian for kx,ky wavevectors
    (always written in a 1d array)"""
    ntotal = len(kx)
    hh = np.zeros((ntotal, 2 * n_slab, 2 * n_slab), dtype=complex)

    if ham_model is 'original':
        # Construct blockes for Hopf Hamiltonian
        a = (np.sin(kx)**2 + t ** 2 * np.sin(ky)**2
             - (np.cos(kx) + np.cos(ky) + m - 3)**2 - 1)
        b = - np.cos(kx) - np.cos(ky) - m + 3
        c = 2 * (t * np.sin(ky) - 1j * np.sin(kx)) * (np.cos(kx) + np.cos(ky)
                                                      + m - 3)
        d = 2 * (t * np.sin(ky) - 1j * np.sin(kx))
    elif ham_model is 'PRL':
        # PRL model
        a = (-np.sin(kx)**2 - np.sin(ky)**2
             + 1 + (np.cos(kx) + np.cos(ky) + m - 3)**2)
        b = np.cos(kx) + np.cos(ky) + m - 3
        c = -2 * (np.sin(ky) - 1j * np.sin(kx)) * (np.cos(kx) + np.cos(ky) + m - 3)
        d = -2 * (np.sin(ky) - 1j * np.sin(kx))
    else:
        print('Define model of the hamiltonian')

    # Compose onsite and hopping matrices
    e = np.stack((np.stack((a, np.conj(c)), axis=-1),
                  np.stack((c, -a), axis=-1)), axis=-1)

    delta = np.stack((np.stack((b, np.zeros(ntotal)), axis=-1),
                      np.stack((d, -b), axis=-1)), axis=-1)  # this gives
                                                             # [b d
                                                             #  0 -b]

    # Construct Hamiltonian for all Nz sites from these blockes
    hh[:, 0:2, 0:2] = e
    for nz in range(0, n_slab - 1):
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


def surf_hamiltonian(hh, red_up=1, red_up_next=1, red_up_next_next=1,
                     red_down=1, red_down_next=1, red_down_next_next=1,
                     alpha=0, beta=0, gamma=0, delta=0,
                     pot_up=0, pot_up_next=0, pot_down=0, pot_down_next=0):
    """Additional surface potential for the slab hamiltonian h"""
    ntotal = hh.shape[0]
    # Surface potential
    # alpha = 0
    # beta = 0
    # gamma = 0
    # delta = -1.1
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

    # Detach upper state:
    # red_up, red_up_next, red_down, red_down_next = 0.6, 1, 0.21, 1
    # Move both surfaces below:
    # red_up, red_up_next, red_down, red_down_next = 1, 1, 0.03, 1

    # Disentangle surface states
    hh[:, 0:2, 0:2] *= red_down
    hh[:, 2:4, 2:4] *= red_down_next
    hh[:, 4:6, 4:6] *= red_down_next_next
    hh[:, -6:-4, -6:-4] *= red_up_next_next
    hh[:, -4:-2, -4:-2] *= red_up_next
    hh[:, -2:, -2:] *= red_up

    # Add surface potential
    hh[:, 0:2, 0:2] += pot_down * hsurf
    hh[:, 2:4, 2:4] += pot_down_next * hsurf
    hh[:, -4:-2, -4:-2] += pot_up_next * hsurf
    hh[:, -2:, -2:] += pot_up * hsurf

    return hh


def projector_down_hamiltonian(u, n_slab, alpha=1):
    """Define projection of the state u from down surface into the bulk
    and add it to the Hamiltonian"""
    p = np.conjugate(u[:, np.newaxis, :]) * u[:, :, np.newaxis]
    ppr = np.zeros((2 * n_slab, 2 * n_slab))
    ppr[0, 0] = 1
    ppr[1, 1] = 1
    ppr = np.tile(ppr[np.newaxis, :, :], (u.shape[0], 1, 1))
    return alpha * np.matmul(np.matmul(ppr, p), ppr)


def projector_up_hamiltonian(u, n_slab, alpha=1):
    """Define projection of the state u from up surface into the bulk
    and add it to the Hamiltonian"""
    p = np.conjugate(u[:, np.newaxis, :]) * u[:, :, np.newaxis]
    ppr = np.zeros((2 * n_slab, 2 * n_slab))
    ppr[-1, -1] = 1
    ppr[-2, -2] = 1
    ppr = np.tile(ppr[np.newaxis, :, :], (u.shape[0], 1, 1))
    return alpha * np.matmul(np.matmul(ppr, p), ppr)


def projector_occ(u_occ, n_slab):
    """Define the projector on the occupied states"""
    # x = u_occ[:, :, np.newaxis, :] * np.conjugate(u_occ[:, :, :, np.newaxis])
    # print(x.shape)
    # p = np.sum(x, axis=-3)
    p = np.matmul(u_occ, np.transpose(np.conjugate(u_occ), (0, 2, 1)))
    nk = 200
    vect = np.random.rand(2 * n_slab)
    vect = vect / sqrt(np.sum(np.power(vect, 2)))
    # print(np.matmul(p[nk, :, :], States[nk, :, 18]) - States[nk, :, 18])
    # print(p.shape)
    # print(np.max(np.abs(p - q)))
    return p


def pzp(states, n_slab):
    """Define PzP operator where P is projector on occupied and surface states
    and z is position operator in a slab"""

    p = projector_occ(states, n_slab)
    z = np.stack((np.arange(n_slab), np.arange(n_slab)), axis=-1)
    z = np.reshape(z, 2 * n_slab, order='C') + 1
    z = np.diag(z)
    z = z[np.newaxis, :, :]
    z = np.tile(z, (p.shape[0], 1, 1))
    # print(z.shape, z)
    return np.matmul(p, np.matmul(z, p))


def weight_surface(states, n_slab):
    """Calculate the upper and lower weights of energy bands"""
    # Weight multiplier
    lamb = 0.5

    # We take into accont that each atom has two orbitals
    # which should have the same weight
    zline = np.stack((np.arange(n_slab), np.arange(n_slab)), axis=-1)
    zline = np.reshape(zline, 2 * n_slab, order='C')
    weight = np.exp(-lamb * zline)
    # Down eigenstate
    down = np.sum(np.multiply(np.power(np.abs(states), 2),
                              weight[np.newaxis, :, np.newaxis]), axis=-2)
    # print('L_14=', l[round(Nx * sqrt(2)) + 50, 14])
    # print('L_15=', l[round(Nx * sqrt(2)) + 50, 15])
    # print(L[round(Nx * sqrt(2)), :])
    # print(Energy[round(Nx * sqrt(2)), :])
    # Up eigenstate
    up = np.sum(np.multiply(np.power(np.abs(states), 2),
                            np.flip(weight[np.newaxis, :, np.newaxis],
                                    axis=-2)),
                axis=-2)
    # print('R_14=', r[round(Nx * sqrt(2)) + 50, 14])
    # print('R_15=', r[round(Nx * sqrt(2)) + 50, 15])
    return down, up


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


def plot_spectrum_MGXM(energy, down, up, nx, n_slab, removebottom, redcolormap,
                       ylimit=3):
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

    kxplot = np.append(np.linspace(0, 1, round(nx * sqrt(2))) * sqrt(2),
                       np.linspace(0, 1, nx) + sqrt(2))
    kxplot = np.append(kxplot, np.linspace(0, 1, nx) + sqrt(2) + 1)
    kxrep = np.transpose(np.tile(kxplot, (2 * n_slab, 1)))
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
        for idslab in range(2 * n_slab):
            for idx in range(2 * nx + round(nx * sqrt(2))):
                if down[idx, idslab] > 0.3:
                    energy[idx, idslab] = 10

    if redcolormap == 1:
        for idslab in range(2 * n_slab):
            if idslab != n_slab:
                down[:, idslab] = 0
                up[:, idslab] = 0

    # Choose cmap
    if redcolormap == 1:
        colmap = red1
    else:
        colmap = blue_red1

    # dot size
    dsize = np.zeros((2 * nx + round(nx * sqrt(2)), 2 * n_slab))
    for idslab in range(2 * n_slab):
        for idx in range(2 * nx + round(nx * sqrt(2))):
                if (up[idx, idslab] > 0.3) or (down[idx, idslab] > 0.3):
                    dsize[idx, idslab] = 1
    spectr = ax.scatter(kxrep, energy, c=up[:, :] - down[:, :],
                        s=ps + 4 * dsize * ps,
                        cmap=colmap)  #
    # spectr = ax.scatter(kxrep, np.real(zbar),
    #                     s=ps)
    ax.set_xlim(0, 2 + sqrt(2))
    if ylimit is None:
        pass
    else:
        ax.set_ylim(-ylimit, ylimit)
    ax.set_xticks(xcoords)
    ax.set_yticks([-2, 0, 2])
    ax.set_xticklabels(['M', '$\Gamma$', 'X', 'M'], fontsize=fs)
    for xc in xcoords:
        ax.axvline(x=xc, color='k', linewidth=lw)
    plt.show()


def slab_plot_MGXM(zbar, nx, n_slab, start_layer=1, n_layers=1):
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
    kxplot = np.append(np.linspace(0, 1, round(nx * sqrt(2))) * sqrt(2),
                       np.linspace(0, 1, nx) + sqrt(2))
    kxplot = np.append(kxplot, np.linspace(0, 1, nx) + sqrt(2) + 1)
    kxrep = np.transpose(np.tile(kxplot, (n_layers, 1)))
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
    # ax.set_ylim(0.5, n_layers + 0.5)
    # y_ticks = [i + 1 for i in range(n_layers)]

    # ax.set_yticks(y_ticks)
    # ax.set_yticklabels([r'1', r'2', r'3', r'4'])
    # ax.set_yticklabels([])

    ax.set_xticks(x_ticks)
    ax.set_xticklabels([r'$M$', '$\Gamma$', r'$X$', r'$M$'])

    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(lss)
    # settings = {'marker': '.', 'linewidth': ls, 'markersize': ps}
    spectr = ax.plot(
        kxplot,
        np.real(zbar[:, n_slab+start_layer:n_slab+start_layer+n_layers]),
        color='black',
        marker='.', markersize=ps,
        linestyle='None', linewidth=lss,
        markeredgecolor='black')
    # spectr = ax.scatter(kxplot, zbar[:, n_slab], sigma_z_expect[:, 0, 0],
    #                     cmap=blue_red1)
    # for ind in range(n_layers):
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


def maker(m, nx, n_slab, ham_model, t=1, calcPzPspec=0, calcPzPchern=0,
          calcQzQspec=0, calcQzQchern=0, calcspec=0,
          addrandpot=0, calcchern=0, statesremove=0,
          removebottom=0, redcolormap=0,
          surf_par={}, spec_par={}, slabspec_par={}, PzPchern_par={}):
    """Main function to run all other functions

    Parameters:
        m  : parameter of Hamiltonian
        nx  : number of points in reduced BZ in one direction
        n_slab  : number of layers in the slab
        ham_model  : model of the Hopf Hamiltonain
        t  : parameter to break C4z
        calcPzPspec  : Plot spectrum of PzP operator
        calcPzPchern  : Calculate Chern number of the eigenstate of
        PzP operator
        calcspec  : Plot spectrum of the Hamiltonian
        addrandpot  : Add random potential on the surface
        calcchern  : Calculare Chern number of the Hamiltonian
        statesremove  : Remove unused surface states by projecting them on
        the bulk
        removebottom  : Remove bottom surface states from the spectrum plot
        by seting all L>0.3 to big value
        redcolormap  : Color detached surface state in red and all others
        in black
        surf_par  : parameters for surface potential
        spec_par  : parameters for spectrum plot
        slabspec_par  : tells for which layers to calculate the spectrum
        PzPchern_par  : tells for which layers to calculate Chern number
    """

    if calcPzPspec == 1 or calcQzQspec or calcspec == 1:
        # MGXM
        kx = np.append(np.linspace(pi, 0, round(nx * sqrt(2))),
                       np.linspace(0, pi, nx))
        kx = np.append(kx, np.ones(nx) * pi)
        ky = np.append(np.linspace(pi, 0, round(nx * sqrt(2))), np.zeros(nx))
        ky = np.append(ky, np.linspace(0, pi, nx))
    # ntotal = 2 * nx + round(nx * sqrt(2))
    #
    if calcPzPchern == 1 or calcQzQchern or calcchern == 1:
        kx, ky = np.meshgrid(np.linspace(0, 2 * pi, nx),
                             np.linspace(0, 2 * pi, nx),
                             indexing='ij')
        kx = np.reshape(kx, -1)
        ky = np.reshape(ky, -1)

    # Constract the Hamiltonian of the slab in z direction
    h = slab_hamiltonian(m, t, kx, ky, n_slab, ham_model)

    if addrandpot == 1:
        h_perturb = 0.1 * perturb_hamiltonian(kx, ky)
        h += h_perturb

    h = surf_hamiltonian(h, **surf_par)

    # Calculate eigenvalues and eigenvectors of H-Hopf
    [energy, states] = np.linalg.eigh(h)
    print(energy.shape)

    # Calculate surface weights of eigenstates
    down, up = weight_surface(states, n_slab)

    if statesremove == 1:
        h = (h
             + projector_up_hamiltonian(states[:, :, n_slab - 1], n_slab, -3.5)
             + projector_up_hamiltonian(states[:, :, n_slab + 1], n_slab, 0.77)
             # - projector_up_hamiltonian(states[:, :, n_slab - 3], n_slab, 1)
             + projector_down_hamiltonian(states[:, :, n_slab - 1], n_slab, -0.5)
             + projector_down_hamiltonian(states[:, :, n_slab + 1], n_slab, 1.4))
        # - projector_down_hamiltonian(states[:, :, n_slab - 3], n_slab, 3))

        [energy, states] = np.linalg.eigh(h)

    # Calculate PzP operator  - position operator projected on set of occupied
    # or empty states
    if calcPzPchern == 1 or calcPzPspec == 1:
        nstart = 0
        nfinish = n_slab
        PzP = pzp(states[:, :, nstart:nfinish], n_slab)
        zbar, zstates = np.linalg.eigh(PzP)
        # sigma_z_expect = spin_expectation_value(
        #     zstates[:, :, n_slab:2 * n_slab])

    if calcQzQchern == 1 or calcQzQspec == 1:
        nstart = 0
        nfinish = n_slab
        QzQ = pzp(states[:, :, nfinish:], n_slab)
        zbar, zstates = np.linalg.eigh(QzQ)
        # sigma_z_expect = spin_expectation_value(
        #     zstates[:, :, n_slab:2 * n_slab])

    # TODO check sortnig algorithm
    # for ind in range(zbar.shape[0]):
    #     index = np.argsort(zbar[ind, :], axis=-1)
    #     zbar[ind, :] = zbar[ind, index]
    #     zstates[ind, :, :] = zstates[ind, :, index]
    # zstates = np.reshape(zstates, (nx, nx, 2 * n_slab, 2 * n_slab))
    # a = zstates[36, 17, :, n_slab]

    # Calculate or plot
    if calcPzPchern == 1 or calcQzQchern == 1:
        start = n_slab + PzPchern_par['st_chern_layer']
        finish = (n_slab + PzPchern_par['st_chern_layer']
                  + PzPchern_par['n_chern_layers'])
        print(start, finish)
        c = chern_number_many_band(zstates[:, :, start:finish], nx, n_slab)
        # c = chern_number(zstates[:, :, n_slab])
        print(c)
    if calcPzPspec == 1 or calcQzQspec == 1:
        slab_plot_MGXM(zbar, nx, n_slab, **slabspec_par)

    if calcspec == 1:
        plot_spectrum_MGXM(energy, down, up, nx, n_slab, removebottom,
                           redcolormap, **spec_par)

    if calcchern == 1:
        # TODO organize Chern calculation
        # c = chern_number_many_band(states[:, :, n_slab], nx, n_slab)
        c = chern_number(states[:, :, n_slab], nx)
        d = flow_hyb_wcc(states[:, :, n_slab], nx)
        print(c)
        plt.figure()
        plt.plot(np.linspace(0, 2 * pi, nx), d)
        plt.show()


# What we calculate:
# calcPzPspec = 0  # Plot spectrum of PzP operator
# calcPzPchern = 0  # Calculate Chern number of the eigenstate of PzP operator
# calcspec = 1  # Plot spectrum of the Hamiltonian
# addsurfpot = 1  # Add surface potential to hamiltonian
# addrandpot = 0  # Add random potential on the surface
# calcchern = 0  # Calculare Chern number of the Hamiltonian
# statesremove = 0  # Remove unused surface states by projecting them on the bulk
# removebottom = 0  # Remove bottom surface states from the spectrum plot
# # by seting all L>0.3 to big value
# redcolormap = 0  # Color detached surface state in red and all others in black
#

# About the model:
# In PRL model Hopf invariant is defined with - sign. Hence at m=1 chi=-1.
# Chern number of the upper surface is -1 as well.

# Parameters of the system
Nx = 200  # Mesh in kx, ky
N_slab = 50  # Number of layers in z direction
m = 1  # Topological phase defining parameter
# model of the Hamiltonian:
# model = 'PRL'  # has additional - sign. Hopf invariant is defined with -
model = 'original'  # Hopf invariant is defined with +


# Parameters for surface potential
# detach upper surface band in PRL model
# surf_param = {'red_down': 0.6, 'red_up': 0.21, 'delta': -1.1, 'pot_down': 1}
# detach lower surface band in original model
# surf_param = {'red_down': 0.21, 'red_up': 0.6, 'delta': -1.1, 'pot_up': 1}
# both surfaces below Fermi level
# surf_param = {'red_up': 0.03, 'delta': -1.1, 'pot_down': 2,
#               'pot_up_next': -1.5, 'pot_up': 1}
# no surface pot
surf_param = {}

# Parameters for spectrum plot
# part of spectrum
spec_param = {'ylimit': 3}
# full spectrum
# spec_param = {'ylimit': None}

# Parameters for slab PzP spectrum
slabspec_param = {'start_layer': N_slab - 5, 'n_layers': N_slab - 1}

# Parameters for PzP chern number
PzPchern_param = {'st_chern_layer': 0, 'n_chern_layers': 3}

maker(m, Nx, N_slab, model, calcspec=1,
      surf_par=surf_param, spec_par=spec_param)

# maker(m, Nx, N_slab, model, calcchern=1,
#       surf_par=surf_param)
#  TODO add parameters for chern number calculation

# maker(m, Nx, N_slab, model, calcPzPspec=1,
#       surf_par=surf_param, slabspec_par=slabspec_param)

# maker(m, Nx, N_slab, model, calcQzQchern=1,
#       surf_par=surf_param, PzPchern_par=PzPchern_param)



# TODO
# np.block
# np.kron
# remove np.power
# ...
# np.ravel

# Gap size check  -- doesn't work now; need to add kx, ky arguments to function
# print('Occ bulk and lower surface')
# gap_size(Energy[:, N_slab - 3], Energy[:, N_slab - 2], 0.05)
# # print('Lower and upper surface')
# # gap_size(Energy[:, N_slab - 1], Energy[:, N_slab], 0.05)
# print('Cond bulk and upper surface')
# gap_size(Energy[:, N_slab - 1], Energy[:, N_slab], 0.05)

# Calculate eigenstates of C4z at Gamma:  -- doesn't work now; need to add
# states arguments to function
# print(symm_c4z_eigenvalues(round(Nx * sqrt(2))))  # round(Nx * sqrt(2))
