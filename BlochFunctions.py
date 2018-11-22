"""
 Created by alexandra at 19.11.18 11:18

Check the results from paper
He and Vanderbilt, Phys Rev Lett 86, no. 23 (2001): 5341–44
"""

import numpy as np
import pickle
from math import pi


def potential(g):
    """periodiacal potential in reciprocal space"""
    v0 = -10
    b = 0.3
    pot = v0 * pi * np.exp(- b**2 * g**2 / 4)
    return pot


def potmatrix(n_g):
    """constact matrix of potentials with size n_g"""
    matr = [[potential(2 * pi * (G1-G2)) for G1 in range(n_g)]
            for G2 in range(n_g)]
    return matr


def hamiltmatrix(n_g, k_point):
    """for each k a matrix of hamiltonian to find coefficients c(k,G)"""
    row_g = 2 * pi * (np.asarray(range(n_g)) - (n_g - 1) / 2)
    matr = potmatrix(n_g) + np.diag(np.power(k_point + row_g, 2))
    return matr


def blochperiodic(k_point, x_row, n_g):
    """construct periodic part of Bloch function"""
    # obtain coefficients
    energy, ck_g = np.linalg.eig(hamiltmatrix(n_g, k_point))
    # G points for exponent e^iGx
    row_g = 2 * pi * (np.asarray(range(n_g)) - (n_g - 1) / 2)
    # the size is n_g * Nx
    ck_g_cut = ck_g[:, 0]
    ck_g_cut = ck_g_cut[:, np.newaxis]
    x_ = x_row[np.newaxis, :]
    row_g_ = row_g[:, np.newaxis]
    # set exponent
    exp_g = np.exp(1j * x_ * row_g_)
    # obtain coefficient columns for k point as a function of x
    ck = np.sum(np.multiply(ck_g_cut, exp_g), axis=0) / n_g
    return ck


def paraltransport(u):
    n_k = np.ma.size(u, 0)
    usmooth = np.empty(u.shape, dtype=complex)
    # initial value for smooth function is equal to original
    # function
    usmooth[0, :] = u[0, :]
    for idk in range(n_k - 1):
        m_old = np.conj(usmooth[idk, :]) * u[idk + 1, :]
        usmooth[idk + 1, :] = u[idk + 1, :] * np.exp(-1j * np.angle(m_old))
    lamb = np.conj(usmooth[0, :]) * usmooth[n_k - 1, :]
    print(lamb.shape)
    nks = np.linspace(0, n_k - 1, n_k)
    nks = nks[:, np.newaxis]
    # Distribute the multiplier among functions at kx in [0, 2pi]
    usmooth = np.multiply(usmooth, np.power(lamb, - nks / (n_k - 1)))
    return usmooth


# number of points in x - space
Nx = 300
# number f unit cells
x_cell = 60
# number of k points
Nk = 200
# number of G points
NG = 601

# k = 2
# xx = 0.3
# # !!! Chech ck for one x
# E, ck_G = np.linalg.eig(hamiltmatrix(NG, k))
# # G points for exponent e^iGx
# row_G = 2 * pi * (np.asarray(range(NG)) - (NG - 1) / 2)
# ck_G = ck_G[:, 0]
# ck_G = paraltransport(ck_G, NG)
# # set exponent
# exp_G = np.exp(1j * xx * row_G)
# # obtain coefficient columns for k point as a function of x
# ck = np.sum(ck_G * exp_G) / NG
# psi = ck * np.exp(1j * k * xx)
# print(psi)

# x = np.array([0, 0.3, 1.3, 2, 3])
# uk1 = blochperiodic(1, x, NG)
# uk2 = blochperiodic(1 + 2 * pi, x, NG)
# Psi1 = uk1 * np.exp(1j * 0 * x)
# Psi2 = uk2 * np.exp(1j * 2 * pi * x)
# print(uk1)
# print(uk2)

# print(blochperiodic(1, np.array([0, 0.3, 1, 2, 3]), NG))

# !!!! main loop
# x vector
x = np.asarray(range(Nx)) / Nx * x_cell
# Wannier = np.zeros(10)
# x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
uk0 = blochperiodic(0, x, NG)
Wannier = uk0
uk = np.empty((Nk, Nx), dtype=complex)  # columns: one x, rows: one k
for nk in range(Nk):
    "for each k calculate coeffitients and then remember them in a matrix"
    k = 2 * np.pi * nk / (Nk - 1)
    print(nk)
    # periodic part of Bloch funcion
    uk[nk, :] = blochperiodic(k, x, NG)

uk_smooth = paraltransport(uk)

k_row = np.linspace(0, 2 * pi, Nk)
# calculate Wannier function as a sum over k
Wannier = np.sum(uk_smooth[0:Nk - 1, :]
                 * np.exp(1j * k_row[0:Nk - 1, np.newaxis] * x[np.newaxis, :]),
                 axis=0)

Wannier = Wannier / Nk

with open('WannieFromBloch.pickle', 'wb') as f:
    pickle.dump([Wannier, Nx, x_cell, NG], f)
