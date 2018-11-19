# -*- coding: utf-8 -*-
"""
Created on Wed Sep  5 18:36:26 2018

@author: aleksandra
"""

import numpy as np
from math import pi
import pickle
import scipy.linalg

"""
Construct Wannier Functions from initial eigenstate of Hopf Hamiltonian
"""


def fourtransf_occ(xx, yy, zz):
    """
    Fourier transformation of two components of occupied Bloch function
    correspond to two localized Wannier functions.
    """
    wf_occ = np.empty((2, 1), dtype=complex)
    u2 = - np.multiply(np.divide(np.sin(kkx) - 1j * t * np.sin(kky), np.sqrt(
        np.power(np.sin(kkx), 2) + t ** 2 * np.power(np.sin(kky), 2) +
        np.power(np.sin(kkz), 2) +
        np.power(np.cos(kkx) + np.cos(kky) + np.cos(kkz) + h, 2))),
                       np.exp(1j * (kkx * xx + kky * yy + kkz * zz)))

    u1 = np.multiply(np.divide(np.sin(kkz) -
                               1j * (np.cos(kkx) + np.cos(kky) +
                                     np.cos(kkz) + h),
                               np.sqrt(
                                   np.power(np.sin(kkx), 2) + t ** 2 * np.power(
                                       np.sin(kky), 2) +
                                   np.power(np.sin(kkz), 2) +
                                   np.power(np.cos(kkx) + np.cos(kky) + np.cos(
                                       kkz) + h, 2))),
                     np.exp(1j * (kkx * xx + kky * yy + kkz * zz)))

    ufourier = np.fft.fftn(u1) / Nk ** 3
    wf_occ[0] = ufourier[0, 0, 0]  # Consider only WF(R=0)
    ufourier = np.fft.fftn(u2) / Nk ** 3
    wf_occ[1] = ufourier[0, 0, 0]
    return np.squeeze(wf_occ)


def fourtransf_val(xx, yy, zz):
    """
    Fourier transformation of two components of valence Bloch function
    correspond to two localized Wannier functions.
    """
    wf_val = np.empty((2, 1), dtype=complex)
    u1 = np.multiply(np.divide(np.sin(kkx) + 1j * t * np.sin(kky), np.sqrt(
        np.power(np.sin(kkx), 2) + t ** 2 * np.power(np.sin(kky), 2) +
        np.power(np.sin(kkz), 2) +
        np.power(np.cos(kkx) + np.cos(kky) + np.cos(kkz) + h, 2))),
                     np.exp(1j * (kkx * xx + kky * yy + kkz * zz)))

    u2 = np.multiply(np.divide(np.sin(kkz) +
                               1j * (np.cos(kkx) + np.cos(kky) +
                                     np.cos(kkz) + h),
                               np.sqrt(
                                   np.power(np.sin(kkx), 2) + t ** 2 * np.power(
                                       np.sin(kky), 2) +
                                   np.power(np.sin(kkz), 2) +
                                   np.power(np.cos(kkx) + np.cos(kky) + np.cos(
                                       kkz) + h, 2))),
                     np.exp(1j * (kkx * xx + kky * yy + kkz * zz)))

    ufourier = np.fft.fftn(u1) / Nk ** 3
    wf_val[0] = ufourier[0, 0, 0]  # Consider only WF(R=0)
    ufourier = np.fft.fftn(u2) / Nk ** 3
    wf_val[1] = ufourier[0, 0, 0]
    return np.squeeze(wf_val)


def fourtransf_occ_1d(xx):
    """
    Calculate Hybrid WF, make only 1d Fourier transform
    """
    ky0 = 0
    kz0 = 0
    wf_occ_1d = np.empty((2, 1), dtype=complex)
    u2 = - np.multiply(np.divide(np.sin(kx) - 1j * t * np.sin(ky0), np.sqrt(
        np.power(np.sin(kx), 2) + t ** 2 * np.power(np.sin(ky0), 2) +
        np.power(np.sin(kz0), 2) +
        np.power(np.cos(kx) + np.cos(ky0) + np.cos(kz0) + h, 2))),
        np.exp(1j * kx * xx))

    u1 = np.multiply(np.divide((np.sin(kz0) - 1j * (
        np.cos(kx) + np.cos(ky0) + np.cos(kz0) + h)), np.sqrt(
        np.power(np.sin(kx), 2) + t ** 2 * np.power(np.sin(ky0), 2) +
        np.power(np.sin(kz0), 2) +
        np.power(np.cos(kx) + np.cos(ky0) + np.cos(kz0) + h, 2))),
        np.exp(1j * kx * xx))

    # ufourier = np.fft.fft(u1) / Nk
    # wf_occ_1d[0] = ufourier[Nk//2 - 1]  # Consider only WF(R=0)
    # ufourier = np.fft.fft(u2) / Nk
    # wf_occ_1d[1] = ufourier[Nk//2 - 1]
    wf_occ_1d[0] = np.sum(u1) / Nk
    wf_occ_1d[1] = np.sum(u2) / Nk
    return np.squeeze(wf_occ_1d)


def scalar_mult(a, b):
    """Define the scalar product of two vectors"""
    mult = np.sum(np.conj(a) * b)
    return mult


def normalize(s_overlap, a_vec):
    """
    Normalization of the vector a by overlap metrix S
    a -> S^(-1/2)a
    """
    a_vec = np.dot(scipy.linalg.sqrtm(scipy.linalg.inv(s_overlap)), a_vec)
    return a_vec


def wannier_1d(n_x1d, n_unitcell):
    """Calculate Wannier function depending on 1 varible (others are 0)"""
    # The function of only one variable, e.g. (x)
    # (100)
    WFoccx = np.empty((n_x1d, 2), dtype=complex)
    WFvalx = np.empty((n_x1d, 2), dtype=complex)
    # (010)
    WFoccy = np.empty((n_x1d, 2), dtype=complex)
    WFvaly = np.empty((n_x1d, 2), dtype=complex)
    # (001)
    WFoccz = np.empty((n_x1d, 2), dtype=complex)
    WFvalz = np.empty((n_x1d, 2), dtype=complex)
    # (110) direction
    WFoccxy = np.empty((n_x1d, 2), dtype=complex)
    WFvalxy = np.empty((n_x1d, 2), dtype=complex)
    # (111) direction
    WFoccxyz = np.empty((n_x1d, 2), dtype=complex)
    WFvalxyz = np.empty((n_x1d, 2), dtype=complex)

    # For different 1d directions calculate WFs as a Fourier transform of
    # Bloch solutions

    for nx in range(n_x1d):
        x = nx * n_unitcell / n_x1d
        print('nx=', nx)
        WFoccx[nx, :] = fourtransf_occ(x, 0, 0)
        # WFvalx[nx, :] = fourtransf_val(x, 0, 0)
        # WFoccy[nx, :] = fourtransf_occ(0, x, 0)
        # WFvaly[nx, :] = fourtransf_val(0, x, 0)
        # WFoccz[nx, :] = fourtransf_occ(0, 0, x)
        # WFvalz[nx, :] = fourtransf_val(0, 0, x)

        # WFoccxy[nx, :] = fourtransf_occ(x, x, 0)
        # WFvalxy[nx, :] = fourtransf_val(x, x, 0)

        # WFoccxyz[nx, :] = fourtransf_occ(x, x, x)
        # WFvalxyz[nx, :] = fourtransf_val(x, x, x)

    with open('WannierLoc1d.pickle', 'wb') as f:
        pickle.dump([WFoccx, WFvalx, WFoccy, WFvaly, WFoccz, WFvalz,
                     WFoccxy, WFvalxy, WFoccxyz, WFvalxyz,
                     n_x1d, n_unitcell], f)


def wannier_3d(n_x3d):
    """Calculate Wannier function depending on 3 variable (x, y, z)"""
    # Set the size of WF matrices - number of points (x, y, z)
    # The 3d function of (x, y, z)
    wf_occ3d = np.empty((n_x3d, n_x3d, n_x3d, 2), dtype=complex)
    wf_val3d = np.empty((n_x3d, n_x3d, n_x3d, 2), dtype=complex)

    # For (x, y, z) in 4 BZs calculate Wannier functions
    # as a Fourier transform of analytical Bloch solutions

    for nx in range(n_x3d):
        x = (nx - (n_x3d - 1) / 2) / (n_x3d-1) * 4
        print('nx=', nx)
        for ny in range(n_x3d):
            y = (ny - (n_x3d - 1) / 2) / (n_x3d-1) * 4
            print('ny=', ny)
            for nz in range(n_x3d):
                z = (nz - (n_x3d - 1) / 2) / (n_x3d-1) * 4
                wf_occ3d[nx, ny, nz, :] = fourtransf_occ(x, y, z)
                wf_val3d[nx, ny, nz, :] = fourtransf_val(x, y, z)

    # Write WFs into file
    with open('WannierLoc.pickle', 'wb') as f:
        pickle.dump([wf_occ3d, wf_val3d, n_x3d], f)


def wannier_hybrid(n_xhybr, n_hybr_uc):
    wf_hybr = np.empty((n_xhybr, 2), dtype=complex)

    for nx in range(n_xhybr):
        x = nx * n_hybr_uc / n_xhybr
        wf_hybr[nx, :] = fourtransf_occ_1d(x)

    # Write Hybrid Wannier function into file
    with open('HybridWannierLoc.pickle', 'wb') as f:
        pickle.dump([wf_hybr, n_xhybr, n_hybr_uc], f)


t = 1
h = 0
Nk = 1024

kx = np.linspace(0, 2 * pi, Nk)
ky = np.linspace(0, 2 * pi, Nk)
kz = np.linspace(0, 2 * pi, Nk)

# Cartesian coordinates, indexing for correct order x,y,z (not y,x,z)
[kkx, kky, kkz] = np.meshgrid(kx, ky, kz, indexing='ij')

Nx1d = 210
Nunitcell = 30
Nx3d = 21       # odd
Nxhybr = 400
Nhybr_uc = 80

wannier_hybrid(Nxhybr, Nhybr_uc)