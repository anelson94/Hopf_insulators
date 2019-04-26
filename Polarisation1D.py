"""
 Created by alexandra at 21.01.19 17:22
 
 Calculate the polarisation of 1D chain at points 0, pi
"""

import numpy as np
from math import pi
import matplotlib.pyplot as plt


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


h = 0
t = 1

Nk = 1000
Nkgrid = 10
# Kz = np.linspace(0, 2 * pi, Nk)

P001_00 = pol_by_angle(0, 0, Nk)
print(P001_00)

P001 = np.empty(Nkgrid)
Ky = 0
for ikx in range(Nkgrid):
    Kx = pi * ikx / (Nkgrid - 1)
    P001[ikx] = -pol_by_angle(Kx, Ky, Nk)

fig = plt.figure(figsize=[1.05, 1.35])
ax = fig.add_axes([0.2, 0.2, 0.7, 0.7])
fs = 10
fss = 9
ls = 1
# ax.set_xlabel(r'$k_x$', fontsize=fs)
# ax.xaxis.set_label_coords(0.5, -0.08)
ax.set_ylabel(r'$\bar{z}$', rotation='horizontal', fontsize=fs)
ax.yaxis.set_label_coords(-0.17, 0.35)
ax.tick_params(labelsize=fss, width=ls)
ax.set_xlim(0, pi)
ax.set_ylim(0, 1)
ax.set_xticks([0, pi])
ax.set_xticklabels([r'$\Gamma$', r'$K$'])
ax.set_yticks([0, 1])
ax.set_yticklabels([r'$0$', r'$1$'])
for axis in ['top', 'bottom', 'left', 'right']:
    ax.spines[axis].set_linewidth(ls)
ax.plot(np.linspace(0, pi, Nkgrid), P001, color='red', marker='.',
        linestyle='solid')
# plt.text(0.5, 0.8, '$k_y=0$', fontsize=fss)
plt.savefig('Images/HybridWannierFunctions/WCC_halfBZ_h0_GK.png',
            bbox_inches=None)
# plt.show()
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
