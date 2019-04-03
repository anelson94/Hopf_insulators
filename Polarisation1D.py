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


h = 2
t = 1

Nk = 8000
Nkgrid = 100
Kz = np.linspace(0, 2 * pi, Nk)
P001 = np.empty((Nkgrid, Nkgrid))
for ikx in range(Nkgrid):
    for iky in range(Nkgrid):
        Kx = 2 * pi * ikx / (Nkgrid - 1)
        Ky = 2 * pi * iky / (Nkgrid - 1)

        # Kx = pi
        # Ky = pi
        psi1 = u1(Kx, Ky, Kz)
        psi2 = u2(Kx, Ky, Kz)
        #
        P001[ikx, iky] = np.real(1j * np.sum(np.conj(psi1[0:Nk - 1])
                                 * (psi1[1:Nk] - psi1[0:Nk - 1])
                                 + np.conj(psi2[0:Nk - 1])
                                 * (psi2[1:Nk] - psi2[0:Nk - 1])) / 2 / pi)

print(np.sum(P001) / Nkgrid**2)
plt.figure()
plt.imshow(P001)
# plt.plot(Kz[0: Nk - 1], np.real((psi1[:Nk - 1] - psi1[1:]))
#          * np.imag(psi1[:Nk - 1]))
plt.colorbar()
plt.show()
