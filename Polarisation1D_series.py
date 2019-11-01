"""
 Created by alexandra at 23.07.19 15:09

 Calculate the polarisation of 1D chain on Gamma, K, M, Gamma line
 for a serie of h parameter
"""

import numpy as np
from math import pi, sqrt
import matplotlib.pyplot as plt
import seaborn as sns


def u1(kx, ky, kz):
    """First component of eigenvector"""
    lamb = np.sqrt(
        np.power(np.sin(kx), 2) + t ** 2 * np.power(np.sin(ky), 2)
        + np.power(np.sin(kz), 2) + np.power(
            np.cos(kx) + np.cos(ky) + np.cos(kz) + h, 2))
    return np.divide(np.sin(kz) - 1j * (
            np.cos(kx) + np.cos(ky) + np.cos(kz) + h), lamb)


def u2(kx, ky, kz):
    """Second component of eigenvector"""
    lamb = np.sqrt(
        np.power(np.sin(kx), 2) + t ** 2 * np.power(np.sin(ky), 2)
        + np.power(np.sin(kz), 2) + np.power(
            np.cos(kx) + np.cos(ky) + np.cos(kz) + h, 2))
    return np.divide(-np.sin(kx) + 1j * t * np.sin(ky), lamb)


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


def pol_by_angle(kx, ky, nz):
    """Calculate polarization taking in account only angle between <uk|uk+dk>"""
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


def pol_GMKG(nk, nz):
    """Pz on the lines GMKG"""
    p = np.empty(3 * nk - 2)
    for ik in range(nk):
        k = pi * ik / (nk - 1)
        p[ik] = -pol_by_angle(k, k, nz)
        p[nk + ik - 1] = -pol_by_angle(pi, pi - k, nz)
        p[2 * nk + ik - 2] = -pol_by_angle(pi - k, 0, nz)
    return p


t = 1

Nk = 1000
Nkgrid = 10

hchi2 = [0, 0.5, 0.9]
Nhchi2 = len(hchi2)

hchi1 = [1.1, 1.5, 2]
Nhchi1 = len(hchi1)


P001chi2 = np.empty((3 * Nkgrid - 2, Nhchi2))
P001chi1 = np.empty((3 * Nkgrid - 2, Nhchi1))

for ih in range(Nhchi2):
    h = hchi2[ih]
    P001chi2[:, ih] = pol_GMKG(Nkgrid, Nk)
for ih in range(Nhchi1):
    h = hchi1[ih]
    P001chi1[:, ih] = pol_GMKG(Nkgrid, Nk)

K_part = np.linspace(0, 1, Nkgrid)
K_plot = np.concatenate((K_part * sqrt(2), sqrt(2) + K_part[1:],
                         sqrt(2) + 1 + K_part[1:]))

# Sizes for paper
fig = plt.figure(figsize=[12, 7])
sns.set_palette(sns.color_palette(['salmon', 'crimson', 'darkred',
                                   'darkblue', 'blue', 'dodgerblue']))

ax = fig.add_axes([0.06, 0.11, 0.92, 0.83])
fs = 36
fss = 30
fsss = 27
ls = 3.5
ps = 25
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
ax.set_ylim(0, 1)
ax.set_xticks(x_ticks)
ax.set_xticklabels([r'$\Gamma$', r'$M$', r'$K$', r'$\Gamma$'])
ax.set_yticks([0, 1])
ax.set_yticklabels([r'$0$', r'$1$'])
for axis in ['top', 'bottom', 'left', 'right']:
    ax.spines[axis].set_linewidth(ls)

settings = {'marker': '.', 'linewidth': ls, 'markersize': ps}

ax.plot(K_plot, P001chi2[:, 0],
        linestyle='solid', label='$h=0$', color='salmon', **settings)
ax.plot(K_plot, P001chi2[:, 1],
        linestyle='solid', label='$h=0.5$', color='crimson', **settings)
ax.plot(K_plot, P001chi2[:, 2],
        linestyle='solid', label='$h=0.9$', color='darkred', **settings)

ax.plot(K_plot, P001chi1[:, 0],
        linestyle='--', label='$h=1.1$', color='darkblue', **settings)
ax.plot(K_plot, P001chi1[:, 1],
        linestyle='--', label='$h=1.5$', color='blue', **settings)
ax.plot(K_plot, P001chi1[:, 2],
        linestyle='--', label='$h=2$', color='dodgerblue', **settings)

for xc in x_ticks:
    ax.axvline(x=xc, color='k', linewidth=ls)
plt.legend(fontsize=fsss, loc='upper left')
# plt.text(0.5, 0.8, '$k_y=0$', fontsize=fss)
# plt.savefig('Images/HybridWannierFunctions/WCC_halfBZ_h-15_GMKG.png',
#             bbox_inches=None)
plt.show()
# P001 = np.empty((Nkgrid, Nkgrid))
# for ikx in range(Nkgrid):
#     for iky in range(Nkgrid):
#         Kx = 2 * pi * ikx / (Nkgrid - 1)
#         Ky = 2 * pi * iky / (Nkgrid - 1)
#
#         P001[ikx, iky] = polariz(Kx, Ky, Nk)
#
#         # Kx = pi
#         # Ky = pi
#         # psi1 = u1(Kx, Ky, Kz)
#         # psi2 = u2(Kx, Ky, Kz)
#         #
#         # P001[ikx, iky] = np.real(1j * np.sum(np.conj(psi1[0:Nk - 1])
#         #                          * (psi1[1:Nk] - psi1[0:Nk - 1])
#         #                          + np.conj(psi2[0:Nk - 1])
#         #                          * (psi2[1:Nk] - psi2[0:Nk - 1])) / 2 / pi)
#
# print(np.sum(P001) / Nkgrid**2)
# plt.figure()
# plt.imshow(P001)
# # plt.plot(Kz[0: Nk - 1], np.real((psi1[:Nk - 1] - psi1[1:]))
# #          * np.imag(psi1[:Nk - 1]))
# plt.colorbar()
# plt.show()
