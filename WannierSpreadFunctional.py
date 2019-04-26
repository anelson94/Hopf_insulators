"""
 Created by alexandra at 16.04.19 17:36

 Calculate the Wannier spread functional using analytical solution
 to analize the localization of Wannier functions
"""

import numpy as np
from math import pi
import json


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


def m_bshift(shift_axis, eigv1, eigv2):
    """Calculate the M(k,b) for the shift in (bx, by, bz) direction for all
    (kx, ky, kz)"""
    bra_1 = np.conj(eigv1)
    bra_2 = np.conj(eigv2)
    ket_1 = np.roll(eigv1, -1, axis=shift_axis)
    ket_2 = np.roll(eigv2, -1, axis=shift_axis)
    return bra_1 * ket_1 + bra_2 * ket_2


def m_angle(b):
    """Calculate the angle for all complex values of overlap matrix M
    in b direction"""
    return np.angle(Mdict[b])


def rb(b):
    """Calculate b*r as a function of b from M-overlap matrix"""
    return -np.sum(Mangledict[b]) / Nk**2 / (2 * pi)


def omega_d():
    """Calculate gauge dependent part of Wannier spread functional"""
    omega = 0
    for ib in range(3):
        undersum = (np.power(Mangledict[ib] + (2 * pi / Nk) * rdict[ib], 2))
        omega += np.sum(undersum) / Nk / (2 * pi)**2
    return omega


def omega_i():
    """Calculate gauge independent part of Wannier spread functional"""
    omega = 0
    for ib in range(3):
        undersum = 1 - np.conj(Mdict[ib]) * Mdict[ib]
        omega += np.sum(undersum) / Nk / (2 * pi)**2
    return omega


h = 2
t = 1
Nk = 200

# Set the meshgrid
Kx = np.linspace(0, 2*pi, Nk + 1)
Ky = np.linspace(0, 2*pi, Nk + 1)
Kz = np.linspace(0, 2*pi, Nk + 1)
# Include the border of the BZ only once
Kx = Kx[0:-1]
Ky = Ky[0:-1]
Kz = Kz[0:-1]

[KKx, KKy, KKz] = np.meshgrid(Kx, Ky, Kz, indexing='ij')

# Calculate eigenvector on a grid
U1 = u1(KKx, KKy, KKz)
U2 = u2(KKx, KKy, KKz)

# ImlogMbx = m_bshift(0, U1, U2)
# ImlogMby = m_bshift(1, U1, U2)
# ImlogMbz = m_bshift(2, U1, U2)

# Create a dictionary of the overlap matrices for all K on grid
# Keys of the dictionary correspond to different b
Mdict = {ib: m_bshift(ib, U1, U2) for ib in range(3)}

# The dictionary of the angles of the matrices M
Mangledict = {ib: m_angle(ib) for ib in range(3)}

# Calculate b*r for each b and write in the dictionary
rdict = {ib: rb(ib) for ib in range(3)}

print(rdict)

OmI = omega_i()
OmD = omega_d()

# with open(
#         'Results/SpreadFunctional/'
#         'WannierSpreadFunctional_h{0}t{1}_N{2}.txt'.format(
#          h, t, Nk), 'wb') as f:
#     json.dump(rdict, f)

print(OmD, OmI)
