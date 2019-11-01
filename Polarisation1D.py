"""
 Created by alexandra at 21.01.19 17:22
 
 Calculate the polarisation of 1D chain at points 0, pi
"""

import numpy as np
from math import pi, sqrt
import matplotlib.pyplot as plt
import pickle
from random import random as rnd


def hamiltonian(kx, ky, kz):
    """Hopf hamiltonian with modifications"""
    hx = (2 * np.sin(kx) * np.sin(kz)
          + np.sin(ky) * (np.cos(kx) + np.cos(ky) + np.cos(kz) + h))
    hy = -(2 * np.sin(ky) * np.sin(kz)
           - np.sin(kx) * (np.cos(kx) + np.cos(ky) + np.cos(kz) + h))
    hz = ((np.power(np.sin(kx), 2) + np.power(np.sin(ky), 2)
           - np.power(np.sin(kz), 2)
           - np.power((np.cos(kx) + np.cos(ky) + np.cos(kz) + h), 2)))
    sigmax = np.array([[0, 1], [1, 0]])
    sigmay = np.array([[0, -1j], [1j, 0]])
    sigmaz = np.array([[1, 0], [0, -1]])
    id = np.array([[1, 0], [0, 1]])
    sym_break_term = (0.15 * sigmaz * np.sin(kz) * (1 + np.cos(ky) + 0.3 * np.sin(kz) + np.sin(kx))
                      + 0.15 * sigmay * (1 + np.sin(kx) + 2 * np.cos(ky)) * (1 + 1.2 * np.sin(kz) + np.cos(ky))
                      + 0.15 * sigmax * (np.sin(kx) - 2.3 * np.cos(kz)) * (1 + 0.7 * np.sin(ky)))
    # sym_break_term = (0.1 * rand_pert(kx, ky, kz) * sigmax
    #                   + 0.1 * rand_pert(kx, ky, kz) * sigmay
    #                   + 0.1 * rand_pert(kx, ky, kz) * sigmaz)
    return hx * sigmax + hy * sigmay + hz * sigmaz + sym_break_term  #


def rand_pert(kx, ky, kz):
    """Generate function of kx, ky, kz with random coeffitients"""
    return (
        (rnd() + rnd() * np.sin(kx) + rnd() * np.sin(ky) + rnd() * np.sin(kz)
         + rnd() * np.cos(kx) + rnd() * np.cos(ky) + rnd() * np.cos(kz))
        * (rnd() + rnd() * np.sin(kx) + rnd() * np.sin(ky) + rnd() * np.sin(kz)
           + rnd() * np.cos(kx) + rnd() * np.cos(ky) + rnd() * np.cos(kz)))


def eigenstate(kx, ky, kz):
    """Occupied eigenstate of hamiltonian at kx, ky, kz"""
    [e, u] = np.linalg.eigh(hamiltonian(kx, ky, kz))
    return u[:, 0]


def scalarprod(a, b):
    # Scalar product of two stackes of wavefunctions of the same size
    # Returns a stack of <A[i,j,...,:]| B[i,j,...,:]>
    prod = np.sum(np.multiply(np.conj(a), b), axis=-1)
    return prod


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
    return usmooth[:, 0], usmooth[:, 1]


def polz_numeric(kx, ky, nz):
    """Calculate z polarization
    using numerically evaluated wavefunctions"""
    u = np.empty((nz, 2), dtype=complex)
    for idz in range(nz):
        kz = 2 * pi * idz / (nz - 1)
        u[idz, :] = eigenstate(kx, ky, kz)
    usmooth1, usmooth2 = parallel_transport(u)
    ps1 = usmooth1[0]
    ps2 = usmooth2[0]
    p001 = 0
    for nkz in range(1, nz):
        psi1_step = usmooth1[nkz]
        psi2_step = usmooth2[nkz]
        p001 += np.angle(np.conj(ps1) * psi1_step + np.conj(ps2) * psi2_step)
        ps1 = psi1_step
        ps2 = psi2_step
    return -p001 / 2 / pi


def u1(kx, ky, kz):
    """First component of eigenvector"""
    lamb = np.sqrt(
        np.power(np.sin(kx), 2) + t**2 * np.power(np.sin(ky), 2)
        + np.power(np.sin(kz), 2) + np.power(
            np.cos(kx) + np.cos(ky) + np.cos(kz) + h, 2))
    return np.divide(np.sin(kz) - 1j * (
            np.cos(kx) + np.cos(ky) + np.cos(kz) + h), lamb)


def u2(kx, ky, kz):
    """Second component of eigenvector"""
    lamb = np.sqrt(
        np.power(np.sin(kx), 2) + t**2 * np.power(np.sin(ky), 2)
        + np.power(np.sin(kz), 2) + np.power(
            np.cos(kx) + np.cos(ky) + np.cos(kz) + h, 2))
    return np.divide(-np.sin(kx) + 1j * t * np.sin(ky), lamb)


def u1_prl(kx, ky, kz):
    """First component of lower eigenvector for PRL model"""
    lamb = np.sqrt(
        np.power(np.sin(kx), 2) + t ** 2 * np.power(np.sin(ky), 2)
        + np.power(np.sin(kz), 2) + np.power(
            np.cos(kx) + np.cos(ky) + np.cos(kz) + m - 3, 2))
    return np.divide(np.sin(kx) + 1j * t * np.sin(ky), lamb)


def u2_prl(kx, ky, kz):
    """Second component of lower eigenvector for PRL model"""
    lamb = np.sqrt(
        np.power(np.sin(kx), 2) + t ** 2 * np.power(np.sin(ky), 2)
        + np.power(np.sin(kz), 2) + np.power(
            np.cos(kx) + np.cos(ky) + np.cos(kz) + m - 3, 2))
    return np.divide(np.sin(kz) + 1j * (
            np.cos(kx) + np.cos(ky) + np.cos(kz) + m - 3), lamb)


def polariz(kx, ky, nk):
    """Calculate the polarization in z direction at (kx,ky) point"""
    ps1 = u1(kx, ky, 0)
    ps2 = u2(kx, ky, 0)
    p001_complex = 0
    for nkz in range(1, nk):
        kz = 2 * pi * nkz / (nk - 1)
        psi1_step = u1(kx, ky, kz)
        psi2_step = u2(kx, ky, kz)
        p001_complex += (np.conj(ps1) * (psi1_step - ps1)
                         + np.conj(ps2) * (psi2_step - ps2))
        ps1 = psi1_step
        ps2 = psi2_step
    p001 = -np.real(1j * p001_complex) / 2 / pi
    return p001


def polz_by_angle(kx, ky, nz):
    """Calculate z polarization
    taking in account only angle between <uk|uk+dk>"""
    ps1 = u1(kx, ky, 0)
    ps2 = u2(kx, ky, 0)
    p001 = 0
    for nkz in range(1, nz):
        kz = 2 * pi * nkz / (nz - 1)
        psi1_step = u1(kx, ky, kz)
        psi2_step = u2(kx, ky, kz)
        p001 += np.angle(np.conj(ps1) * psi1_step + np.conj(ps2) * psi2_step)
        ps1 = psi1_step
        ps2 = psi2_step
    return -p001 / 2 / pi


def polx_by_angle(nx, ky, kz):
    """Calculate x polarization
    taking in account only angle between <uk|uk+dk>"""
    ps1 = u1(0, ky, kz)
    ps2 = u2(0, ky, kz)
    p100 = 0
    for nkx in range(1, nx):
        kx = 2 * pi * nkx / (nx - 1)
        psi1_step = u1(kx, ky, kz)
        psi2_step = u2(kx, ky, kz)
        p100 += np.angle(np.conj(ps1) * psi1_step + np.conj(ps2) * psi2_step)
        ps1 = psi1_step
        ps2 = psi2_step
    return -p100 / 2 / pi


def poly_by_angle(kx, ny, kz):
    """Calculate y polarization
    taking in account only angle between <uk|uk+dk>"""
    ps1 = u1(kx, 0, kz)
    ps2 = u2(kx, 0, kz)
    p100 = 0
    for nky in range(1, ny):
        ky = 2 * pi * nky / (ny - 1)
        psi1_step = u1(kx, ky, kz)
        psi2_step = u2(kx, ky, kz)
        p100 += np.angle(np.conj(ps1) * psi1_step + np.conj(ps2) * psi2_step)
        ps1 = psi1_step
        ps2 = psi2_step
    return -p100 / 2 / pi


def wccz_low(kx, ky, nz):
    """WCC in z direction of lower band at kx, ky point"""
    ps1 = u1_prl(kx, ky, 0)
    ps2 = u2_prl(kx, ky, 0)
    wcc = 0
    for nkz in range(1, nz):
        kz = 2 * pi * nkz / (nz - 1)
        psi1_step = u1_prl(kx, ky, kz)
        psi2_step = u2_prl(kx, ky, kz)
        wcc += np.angle(np.conj(ps1) * psi1_step + np.conj(ps2) * psi2_step)
        ps1 = psi1_step
        ps2 = psi2_step
    return -wcc / 2 / pi


def wccz_up(kx, ky, nz):
    """WCC in z direction of upper band at kx, ky point"""
    ps1 = u1_prl(kx, ky, 0)
    ps2 = u2_prl(kx, ky, 0)
    wcc = 0
    for nkz in range(1, nz):
        kz = 2 * pi * nkz / (nz - 1)
        psi1_step = u1_prl(kx, ky, kz)
        psi2_step = u2_prl(kx, ky, kz)
        wcc += np.angle(ps1 * np.conj(psi1_step) + ps2 * np.conj(psi2_step))
        ps1 = psi1_step
        ps2 = psi2_step
    return -wcc / 2 / pi


def pol001_GMKG_num(nk, nz):
    """Pz on the lines GMKG"""
    p = np.empty(3 * nk - 2)
    for ik in range(nk):
        k = pi * ik / (nk - 1)
        p[ik] = -polz_numeric(k, k, nz)
        p[nk + ik - 1] = -polz_numeric(pi, pi - k, nz)
        p[2 * nk + ik - 2] = -polz_numeric(pi - k, 0, nz)
        # if p[ik] < 0:
        #     p[ik] += 1
        # if p[ik + nk - 1] < 0:
        #     p[ik + nk - 1] += 1
        # if p[ik + 2 * nk - 2] < 0:
        #     p[ik + 2 * nk - 2] += 1
    return p


def pol001_GMKG(nk, nz):
    """Pz on the lines GMKG"""
    p = np.empty(3 * nk - 2)
    for ik in range(nk):
        k = pi * ik / (nk - 1)
        p[ik] = -polz_by_angle(k, k, nz)
        p[nk + ik - 1] = -polz_by_angle(pi, pi - k, nz)
        p[2 * nk + ik - 2] = -polz_by_angle(pi - k, 0, nz)
    return p


def wcc001_low_MGXM(nk, nz):
    """z WCC of occ state on the lines MYGXM"""
    wcc = np.empty(3 * nk - 2)
    for ik in range(nk):
        k = pi * ik / (nk - 1)
        wcc[ik] = wccz_low(pi - k, pi - k, nz)
        wcc[nk + ik - 1] = wccz_low(k, 0, nz)
        wcc[2 * nk + ik - 2] = wccz_low(pi, k, nz)
    return wcc


def wcc001_up_MGXM(nk, nz):
    """z WCC of unocc state on the lines MYGXM"""
    wcc = np.empty(3 * nk - 2)
    for ik in range(nk):
        k = pi * ik / (nk - 1)
        wcc[ik] = wccz_up(pi - k, pi - k, nz)
        wcc[nk + ik - 1] = wccz_up(k, 0, nz)
        wcc[2 * nk + ik - 2] = wccz_up(pi, k, nz)
    return wcc


def pol100_CKMCLM(nk, nx):
    """Px on the lines CKMCLM"""
    p = np.empty(5 * nk - 4)
    for ik in range(nk):
        k = pi * ik / (nk - 1)
        p[ik] = -polx_by_angle(nx, 0, k)
        p[nk + ik - 1] = -polx_by_angle(nx, k, pi)
        p[2 * nk + ik - 2] = -polx_by_angle(nx, pi - k, pi - k)
        p[3 * nk + ik - 3] = -polx_by_angle(nx, k, 0)
        p[4 * nk + ik - 4] = -polx_by_angle(nx, pi, k)
    return p


def pol100_CMKC(nk, nx):
    """Px on the lines CMKC"""
    p = np.empty(3 * nk - 2)
    for ik in range(nk):
        k = pi * ik / (nk - 1)
        p[ik] = -polx_by_angle(nx, k, k)
        p[nk + ik - 1] = -polx_by_angle(nx, pi - k, pi)
        p[2 * nk + ik - 2] = -polx_by_angle(nx, 0, pi - k)
    return p


def pol010_CMKC(nk, ny):
    """Py on the lines CMKC"""
    p = np.empty(3 * nk - 2)
    for ik in range(nk):
        k = pi * ik / (nk - 1)
        p[ik] = -poly_by_angle(k, ny, k)
        p[nk + ik - 1] = -poly_by_angle(pi, ny, pi - k)
        p[2 * nk + ik - 2] = -poly_by_angle(pi - k, ny, 0)
    return p


def pol100_2d(nk, nx):
    """Px on a (ky,kz) plane"""
    p = np.empty((nk, nk))
    for iky in range(nk):
        ky = 2 * pi * iky / (nk - 1) - pi
        for ikz in range(nk):
            kz = 2 * pi * ikz / (nk - 1) - pi
            p[iky, ikz] = -polx_by_angle(nx, ky, kz)
    return p


def pol010_2d(nk, ny):
    """Py on a (kx,kz) plane"""
    p = np.empty((nk, nk))
    for ikx in range(nk):
        kx = 2 * pi * ikx / (nk - 1) - pi
        for ikz in range(nk):
            kz = 2 * pi * ikz / (nk - 1) - pi
            p[ikx, ikz] = -poly_by_angle(kx, ny, kz)
    return p


def GMKG_plot():
    """Plot polarization along GMKG"""
    K_part = np.linspace(0, 1, Nkgrid)
    K_plot = np.concatenate((K_part * sqrt(2), sqrt(2) + K_part[1:],
                             sqrt(2) + 1 + K_part[1:]))
    # Sizes for paper
    fig = plt.figure(figsize=[12, 7])
    ax = fig.add_axes([0.06, 0.11, 0.92, 0.83])
    fs = 36
    fss = 30
    ls = 3.5
    ps = 30
    ax.yaxis.set_label_coords(-0.03, 0.5)

    # Sizes for poster
    # fig = plt.figure(figsize=[3.5, 5])
    # ax = fig.add_axes([0.13, 0.13, 0.82, 0.82])
    # fs = 40
    # fss = 35
    # axs = 2.5
    # ls = 5
    # ps = 20
    # ax.set_xlabel(r'$k_x$', fontsize=fs)
    # ax.xaxis.set_label_coords(0.5, -0.08)
    # ax.yaxis.set_label_coords(-0.1, 0.5)
    ax.set_ylabel(r'$\bar{z}$', rotation='horizontal', fontsize=fs)
    # ylabel place for paper
    # ax.yaxis.set_label_coords(-0.17, 0.35)

    # ylabel place for poster
    x_ticks = [0, sqrt(2), sqrt(2) + 1, sqrt(2) + 2]
    ax.tick_params(labelsize=fss, width=ls)
    ax.tick_params(axis='x', labelsize=fs)
    ax.tick_params(axis='y', pad=3)
    ax.set_xlim(0, sqrt(2) + 2)
    # ax.set_ylim(0, 1)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels([r'$\Gamma$', r'$M$', r'$K$', r'$\Gamma$'])
    # ax.set_yticks([0, 1])
    # ax.set_yticklabels([r'$0$', r'$1$'])
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(ls)
    settings = {'marker': '.', 'linewidth': ls, 'markersize': ps}
    # ax.plot(K_plot, P100_h05, color='red',
    #         linestyle='solid', label='$\chi=-2$',
    #         **settings)  # np.linspace(0, 3, 3 * Nkgrid - 2)
    ax.plot(K_plot, P001_hmin2, color='blue',
            linestyle='dashed', label='$\chi=1$', **settings)
    for xc in x_ticks:
        ax.axvline(x=xc, color='k', linewidth=ls)
    plt.legend(fontsize=fss, loc=(0.01, 0.7))
    # plt.text(0.5, 0.8, '$k_y=0$', fontsize=fss)
    # plt.savefig('Images/HybridWannierFunctions/WCC_halfBZ_chi12_GMKG.png',
    #             bbox_inches=None)
    plt.show()


def CKMCLM_plot():
    """Plot polarization along CKMCLM"""
    K_part = np.linspace(0, 1, Nkgrid)
    K_plotx = np.concatenate((K_part, 1 + K_part[1:],
                              2 + sqrt(2) * K_part[1:],
                              2 + sqrt(2) + K_part[1:],
                              3 + sqrt(2) + K_part[1:]))
    x_ticks = [0, 1, 2, 2 + sqrt(2), 3 + sqrt(2), 4 + sqrt(2)]
    fig1 = plt.figure()
    ax1 = fig1.add_axes([0.1, 0.05, 0.8, 0.8])
    ax1.set_xticks(x_ticks)
    ax1.set_xticklabels([r'$C$', r'$K$', r'$M$', r'$C$', r'$L$', r'$M$'])
    ax1.plot(K_plotx, P100_h15)
    for xc in x_ticks:
        ax1.axvline(x=xc, color='k', linewidth=0.5)
    plt.show()


def twod_plot():
    """Plot polarization on a plane"""
    fig = plt.figure()
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    im = ax.imshow(P010_h15_2d)
    ax.set_xticks([0, Nkgrid - 1])
    ax.set_xticklabels([r'$-\pi$', r'$\pi$'])
    ax.set_yticks([0, Nkgrid - 1])
    ax.set_yticklabels([r'$-\pi$', r'$\pi$'])
    cbar = ax.figure.colorbar(im, ax=ax)
    plt.show()


def slab_plot_MGXM():
    """Plot WCC along MGXM for a slab with finite number of layers"""
    K_part = np.linspace(0, 1, Nkgrid)
    K_plot = np.concatenate((K_part * sqrt(2), sqrt(2) + K_part[1:],
                             sqrt(2) + 1 + K_part[1:]))

    # Sizes for paper
    fig = plt.figure(figsize=[3, 3])
    ax = fig.add_axes([0.11, 0.13, 0.82, 0.83])
    fs = 25
    fss = 22
    lss = 1.4
    ls = 2.

    ax.yaxis.set_label_coords(-0.03, 0.5)

    x_ticks = [0, sqrt(2), sqrt(2) + 1, sqrt(2) + 2]
    ax.tick_params(labelsize=fss, width=lss)
    ax.tick_params(axis='x', labelsize=fs)
    ax.tick_params(axis='y', pad=3)
    ax.set_xlim(0, sqrt(2) + 2)
    ax.set_ylim(-1.2, N_layers + 0.2)
    y_ticks = [0, 1, 2, 3]

    ax.set_yticks(y_ticks)
    ax.set_yticklabels([r'1', r'2', r'3', r'4'])
    # ax.set_yticklabels([])

    ax.set_xticks(x_ticks)
    ax.set_xticklabels([r'$M$', '$\Gamma$', r'$X$', r'$M$'])

    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(lss)
    # settings = {'marker': '.', 'linewidth': ls, 'markersize': ps}
    for ind in range(N_layers):
        ax.plot(K_plot, WCCz_m1 + ind, color='black',
                linestyle='solid', linewidth=lss)
        ax.scatter(x_ticks, np.ones(len(x_ticks)) * ind,
                   s=120, facecolors='b', edgecolors='b')
    # ax.scatter(sqrt(2), 4, s=120, facecolors='none', edgecolors='b')
    ax.plot([0, 2 + sqrt(2)], [-0.5, -0.5], 'k--')
    ax.plot([0, 2 + sqrt(2)], [3.5, 3.5], 'k--')
    for xc in x_ticks:
        ax.axvline(x=xc, color='k', linewidth=lss)
    # plt.show()
    plt.savefig('Images/Polarization/Slab/WCC_low_slab5_m-10.png',
                transparent=True)


t = 1

Nk = 50
Nkgrid = 40
# Nkavg = 200
# Kz = np.linspace(0, 2 * pi, Nk)

#        Average polarization
# h = 0.0
# P010_avg = 0
# for ikx in range(Nkavg):
#     for ikz in range(Nkavg):
#         Kx = ikx * 2 * pi / Nkavg
#         Kz = ikz * 2 * pi / Nkavg
#         P010_avg += poly_by_angle(Kx, Nk, Kz) / Nkavg**2
# # P010_avg = P010_avg
#
# print(P010_avg)

h = 0.5
# P001_h05 = pol001_GMKG(Nkgrid, Nk)
# P100_h05 = pol100_CKMCLM(Nkgrid, Nk)
# P100_h05_2d = pol100_2d(Nkgrid, Nk)
# h = 1.5
h = 1.5
# P001_h15 = pol001_GMKG_num(Nkgrid, Nk)
# print(P001_h15[Nkgrid - 1])
# print(P001_h15[0])
# print(P001_h15[-1])
# P100_h15 = pol100_CKMCLM(Nkgrid, Nk)
# P010_h15 = -pol010_CMKC(Nkgrid, Nk)
# P100_h15_2d = pol100_2d(Nkgrid, Nk)
# P010_h15_2d = pol010_2d(Nkgrid, Nk)

N_layers = 4
m = -10
WCCz_m1 = wcc001_low_MGXM(Nkgrid, Nk)

slab_plot_MGXM()
# GMKG_plot()
# CKMCLM_plot()
# twod_plot()

# plt.figure()
# plt.imshow(P001)
# # plt.plot(Kz[0: Nk - 1], np.real((psi1[:Nk - 1] - psi1[1:]))
# #          * np.imag(psi1[:Nk - 1]))
# plt.colorbar()
# plt.show()
