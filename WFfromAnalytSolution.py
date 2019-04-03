# -*- coding: utf-8 -*-
"""
Created on Wed Sep  5 18:36:26 2018

@author: aleksandra
"""

import numpy as np
from math import pi
import pickle

"""
Construct Wannier Functions from initial eigenstate of Hopf Hamiltonian
"""


def fourtransf_occ(nkx, nky, nkz):
    """
    Fourier transformation of two components of occupied Bloch function
    correspond to two localized Wannier functions.
    """
    wf_occ = np.empty((nkx, nky, nkz, 2), dtype=complex)

    kx = np.linspace(0, 2 * pi, nkx)
    ky = np.linspace(0, 2 * pi, nky)
    kz = np.linspace(0, 2 * pi, nkz)

    # Cartesian coordinates, indexing for correct order x,y,z (not y,x,z)
    [kkx, kky, kkz] = np.meshgrid(kx, ky, kz, indexing='ij')

    u2 = - (np.divide(np.sin(kkx) - 1j * t * np.sin(kky), np.sqrt(
            np.power(np.sin(kkx), 2) + t ** 2 * np.power(np.sin(kky), 2)
            + np.power(np.sin(kkz), 2)
            + np.power(np.cos(kkx) + np.cos(kky) + np.cos(kkz) + h, 2))))

    u1 = (np.divide(np.sin(kkz) - 1j * (np.cos(kkx) + np.cos(kky)
          + np.cos(kkz) + h), np.sqrt(
          np.power(np.sin(kkx), 2) + t ** 2 * np.power(np.sin(kky), 2)
          + np.power(np.sin(kkz), 2)
          + np.power(np.cos(kkx) + np.cos(kky) + np.cos(kkz) + h, 2))))

    wf_occ[:, :, :, 0] = np.fft.fftn(u1) / nkx / nky / nkz
    wf_occ[:, :, :, 1] = np.fft.fftn(u2) / nkx / nky / nkz
    return wf_occ


def fourtransf_val(nkx, nky, nkz):
    """
    Fourier transformation of two components of valence Bloch function
    correspond to two localized Wannier functions.
    """
    wf_val = np.empty((nkx, nky, nkz, 2), dtype=complex)

    kx = np.linspace(0, 2 * pi, nkx)
    ky = np.linspace(0, 2 * pi, nky)
    kz = np.linspace(0, 2 * pi, nkz)

    # Cartesian coordinates, indexing for correct order x,y,z (not y,x,z)
    [kkx, kky, kkz] = np.meshgrid(kx, ky, kz, indexing='ij')

    u1 = (np.divide(np.sin(kkx) + 1j * t * np.sin(kky), np.sqrt(
          np.power(np.sin(kkx), 2) + t ** 2 * np.power(np.sin(kky), 2)
          + np.power(np.sin(kkz), 2)
          + np.power(np.cos(kkx) + np.cos(kky) + np.cos(kkz) + h, 2))))

    u2 = (np.divide(np.sin(kkz) + 1j * (np.cos(kkx) + np.cos(kky)
          + np.cos(kkz) + h), np.sqrt(
          np.power(np.sin(kkx), 2) + t ** 2 * np.power(np.sin(kky), 2)
          + np.power(np.sin(kkz), 2)
          + np.power(np.cos(kkx) + np.cos(kky) + np.cos(kkz) + h, 2))))

    wf_val[:, :, :, 0] = np.fft.fftn(u1) / nkx / nky / nkz
    wf_val[:, :, :, 1] = np.fft.fftn(u2) / nkx / nky / nkz
    return wf_val


def bloch_occ(nkx, nky, nkz):
    """
    Construct occupied bloch function for further fourier transform
    """

    kx = np.linspace(0, 2 * pi, nkx)
    ky = np.linspace(0, 2 * pi, nky)
    kz = np.linspace(0, 2 * pi, nkz)

    # Cartesian coordinates, indexing for correct order x,y,z (not y,x,z)
    [kkx, kky, kkz] = np.meshgrid(kx, ky, kz, indexing='ij')

    u2 = - np.multiply(np.divide(np.sin(kkx) - 1j * t * np.sin(kky), np.sqrt(
        np.power(np.sin(kkx), 2) + t ** 2 * np.power(np.sin(kky), 2)
        + np.power(np.sin(kkz), 2)
        + np.power(np.cos(kkx) + np.cos(kky) + np.cos(kkz) + h, 2))),
        np.exp(-1j * 6 * kkz))

    u1 = np.multiply(np.divide(np.sin(kkz) - 1j * (np.cos(kkx) + np.cos(kky)
                               + np.cos(kkz) + h), np.sqrt(
        np.power(np.sin(kkx), 2) + t ** 2 * np.power(np.sin(kky), 2)
        + np.power(np.sin(kkz), 2)
        + np.power(np.cos(kkx) + np.cos(kky) + np.cos(kkz) + h, 2))),
        np.exp(-1j * 6 * kkz))
    return u1, u2


def fourtransf_occ_1d(nk):
    """
    Calculate Hybrid WF, make only 1d Fourier transform
    """
    wf_occ_1d = np.empty((nk, 2), dtype=complex)
    kx0 = np.linspace(0, 2 * pi, nk)
    ky0 = 0
    kz0 = 0

    u2 = - (np.divide(np.sin(kx0) - 1j * t * np.sin(ky0), np.sqrt(
            np.power(np.sin(kx0), 2) + t ** 2 * np.power(np.sin(ky0), 2)
            + np.power(np.sin(kz0), 2)
            + np.power(np.cos(kx0) + np.cos(ky0) + np.cos(kz0) + h, 2))))

    u1 = (np.divide((np.sin(kz0) - 1j * (np.cos(kx0) + np.cos(ky0)
          + np.cos(kz0) + h)), np.sqrt(
          np.power(np.sin(kx0), 2) + t ** 2 * np.power(np.sin(ky0), 2)
          + np.power(np.sin(kz0), 2)
          + np.power(np.cos(kx0) + np.cos(ky0) + np.cos(kz0) + h, 2))))

    wf_occ_1d[:, 0] = np.fft.fft(u1) / nk
    wf_occ_1d[:, 1] = np.fft.fft(u2) / nk
    return wf_occ_1d


def wannier_1d(n_1d, nkx, nky, nkz):
    """Calculate Wannier function depending on 1 varible (others are 0)"""
    # The function of only one variable, e.g. (x) (100)
    wf_occ_x = fourtransf_occ(nkx, nky, nkz)[0:n_1d, 0, 0, :]

    with open('WannierLoc1d.pickle', 'wb') as f:
        pickle.dump([wf_occ_x, n_1d], f)


def wannier_hybrid(n_xhybr, nk):
    """Calculate Hybrid Wannier function at n_xhybr integer points"""
    wf_hybr = fourtransf_occ_1d(nk)[0:n_xhybr, :]

    # Write Hybrid Wannier function into file
    with open('HybridWannierLoc.pickle', 'wb') as f:
        pickle.dump([wf_hybr, n_xhybr], f)


# Parameters
t = 1
h = 0
# Large number of points in fft for direction with R dependence
Nksmall = 50
Nk = 100000

# Number of R points
Nx1d = 20
Nx3d = 21       # odd
Nxhybr = 30

# Specify which function we want to calculate (Hybrid or normal)
# Specify in which direction we want to check the dependance
wannier_1d(Nx1d, Nk, Nksmall, Nksmall)

# !!! Check Fourier
# uocc1, uocc2 = bloch_occ(Nk, Nksmall, Nksmall)
# print(uocc1.shape)
# wf1 = np.sum(uocc1) / Nksmall**2 / Nk
# wf2 = np.sum(uocc2) / Nksmall**2 / Nk
# print(wf1)
#
# print(np.sqrt(np.conj(wf1) * wf1 + np.conj(wf2) * wf2))
