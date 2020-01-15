"""
 Created by alexandra at 15.01.20 15:55
 
 A module with hopf hamiltonians
"""


import numpy as np


def ham_mrw(m, kx, ky, kz):
    """hamiltonian of Moore, Ran and Wen"""
    # Pauli matrices
    sigmax = np.array([[0, 1], [1, 0]])
    sigmay = np.array([[0, -1j], [1j, 0]])
    sigmaz = np.array([[1, 0], [0, -1]])

    # Pauli matrices for calculations at all (kx, ky, kz)
    sigmax = sigmax[np.newaxis, np.newaxis, np.newaxis, :, :]

    sigmay = sigmay[np.newaxis, np.newaxis, np.newaxis, :, :]

    sigmaz = sigmaz[np.newaxis, np.newaxis, np.newaxis, :, :]

    # hopf hamiltonian is a mapping function from T^3 to S^2.
    # It has two energy states, one of them occupied.

    hx = (np.sin(kx) * np.sin(kz)
          + np.sin(ky) * (np.cos(kx) + np.cos(ky) + np.cos(kz) - 3 + m))
    hy = (- np.sin(ky) * np.sin(kz)
          + np.sin(kx) * (np.cos(kx) + np.cos(ky) + np.cos(kz) - 3 + m))
    hz = (np.sin(kx)**2 + np.sin(ky)**2
          - np.sin(kz)**2 - (np.cos(kx) + np.cos(ky) + np.cos(kz) - 3 + m)**2)

    hx = hx[:, :, :, np.newaxis, np.newaxis]
    hy = hy[:, :, :, np.newaxis, np.newaxis]
    hz = hz[:, :, :, np.newaxis, np.newaxis]

    return hx * sigmax + hy * sigmay + hz * sigmaz


def ham_sym_breaking(m, alpha, kx, ky, kz):
    """hamiltonian of Moore, Ran and Wen"""
    # Pauli matrices
    sigmax = np.array([[0, 1], [1, 0]])
    sigmay = np.array([[0, -1j], [1j, 0]])
    sigmaz = np.array([[1, 0], [0, -1]])

    # Pauli matrices for calculations at all (kx, ky, kz)
    sigmax = sigmax[np.newaxis, np.newaxis, np.newaxis, :, :]

    sigmay = sigmay[np.newaxis, np.newaxis, np.newaxis, :, :]

    sigmaz = sigmaz[np.newaxis, np.newaxis, np.newaxis, :, :]

    # hopf hamiltonian is a mapping function from T^3 to S^2.
    # It has two energy states, one of them occupied.

    hx = (np.sin(kx) * np.sin(kz)
          + np.sin(ky) * (np.cos(kx) + np.cos(ky) + np.cos(kz) - 3 + m))
    hy = (- np.sin(ky) * np.sin(kz)
          + np.sin(kx) * (np.cos(kx) + np.cos(ky) + np.cos(kz) - 3 + m))
    hz = (np.sin(kx) ** 2 + np.sin(ky) ** 2
          - np.sin(kz) ** 2 - (
                      np.cos(kx) + np.cos(ky) + np.cos(kz) - 3 + m) ** 2)

    hx = hx[:, :, :, np.newaxis, np.newaxis]
    hy = hy[:, :, :, np.newaxis, np.newaxis]
    hz = hz[:, :, :, np.newaxis, np.newaxis]

    return hx * sigmax + hy * sigmay + hz * sigmaz

