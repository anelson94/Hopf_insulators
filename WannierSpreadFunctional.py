"""
 Created by alexandra at 16.04.19 17:36

 Calculate the Wannier spread functional using analytical solution
 to analize the localization of Wannier functions

 20.06.19 Add steepest decent method for omega_D minimization
"""

import numpy as np
from math import pi
# import json


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
    return np.angle(Mdict[b]) + 0.0j


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


def steepestdecent(om, alpha, a, r):
    """Steepest decent method for spread funct minimization
    with step alpha/3
    absolute error a
    and relative error r"""
    omnew = minimstep(om, alpha)
    count = 1
    while abs((omnew - om) / om) > r and abs(omnew) > a and count < 3:
        om = omnew
        omnew = minimstep(om, alpha)
        count += 1
        print(count)  # , omnew, Mangledict[0][29, 5, 78], np.abs(Mdict[0][29, 5, 78]))
    return omnew


def minimstep(om, alpha):
    """Step in minimization procedure with multiplier alpha/3"""
    global Mdict, rdict, Mangledict
    g = 0
    for ib in range(3):
        g = g + 1j * 4 * (Mangledict[ib] + (2 * pi / Nk) * rdict[ib])
    print(np.sum(np.abs(g)) / Nk**2)
    print(np.sum(np.abs(om_deriv())) / Nk**2)

    om = om - alpha / 3 * np.sum(np.power(np.abs(g), 2)) / Nk / (2 * pi)**2
    u_unitary = np.exp(alpha / 3 * g)  # 1 - alpha / 3 * g
    for ib in range(3):
        Mdict[ib] = (np.conjugate(u_unitary) * Mdict[ib]
                     * np.roll(u_unitary, -1, axis=ib))
        # mdict_check = Mdict[ib] + alpha / 3 * (-g + np.roll(g, -1, axis=ib)) * Mdict[ib]
        # print(np.max(np.abs(Mdict[ib] - mdict_check)))
        Mangledict[ib] = m_angle(ib)
        rdict[ib] = rb(ib)
    om_check = omega_d()
    print(om, om_check)
    return om


def om_deriv():
    deriv = np.empty((Nk, Nk, Nk), dtype=complex)
    for idkx in range(Nk):
        for idky in range(Nk):
            for idkz in range(Nk):
                dw = 0.01
                mangle = Mangledict
                mangle[0][idkx, idky, idkz] += (
                    -mangle[0][idkx, idky, idkz] * dw * 1j)
                mangle[0][idkx - 1, idky, idkz] += (
                    mangle[0][idkx, idky, idkz] * dw * 1j)
                mangle[1][idkx, idky, idkz] += (
                        -mangle[1][idkx, idky, idkz] * dw * 1j)
                mangle[1][idkx, idky - 1, idkz] += (
                        mangle[1][idkx, idky, idkz] * dw * 1j)
                mangle[2][idkx, idky, idkz] += (
                        -mangle[2][idkx, idky, idkz] * dw * 1j)
                mangle[2][idkx, idky, idkz - 1] += (
                        mangle[2][idkx, idky, idkz] * dw * 1j)

                r = {ib: rb(ib) for ib in range(3)}
                deriv[idkx, idky, idkz] = (
                        (omega_d_loc(mangle, r) - omega_d()) / dw)

    return deriv


def omega_d_loc(mangle, r):
    omega = 0
    for ib in range(3):
        undersum = (np.power(mangle[ib] + (2 * pi / Nk) * r[ib], 2))
        omega += np.sum(undersum) / Nk / (2 * pi) ** 2
    return omega


h = 3.1
t = 1
Nk = 50

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

# Calculate initial spread functionals
OmI = omega_i()
OmD = omega_d()

# Steepest decent:
# Alpha = 0.2
# epsa = 0.000001
# epsr = 0.0001
# OmDmin = steepestdecent(OmD, Alpha, epsa, epsr)
OmDmin = OmD

# with open(
#         'Results/SpreadFunctional/'
#         'WannierSpreadFunctional_h{0}t{1}_N{2}.txt'.format(
#          h, t, Nk), 'wb') as f:
#     json.dump(rdict, f)

print(OmD, OmDmin, OmI)
