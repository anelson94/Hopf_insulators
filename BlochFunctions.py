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
    index = np.argmin(energy)
    # G points for exponent e^iGx
    row_g = 2 * pi * (np.asarray(range(n_g)) - (n_g - 1) / 2)
    # the size is n_g * Nx
    ck_g_cut = ck_g[:, index]
    ck_g_cut = ck_g_cut[:, np.newaxis]
    x_ = x_row[np.newaxis, :]
    row_g_ = row_g[:, np.newaxis]
    # set exponent
    exp_g = np.exp(1j * x_ * row_g_)
    # obtain coefficient columns for k point as a function of x
    ck = np.sum(np.multiply(ck_g_cut, exp_g), axis=0)
    return ck


def paraltransport(u):
    n_k = np.ma.size(u, 0)
    # Normalization
    u_abs = np.sqrt(np.sum(np.conj(u) * u, axis=-1))
    u = np.divide(u, u_abs[:, np.newaxis])
    usmooth = np.empty(u.shape, dtype=complex)
    # initial value for smooth function is equal to original
    # function
    usmooth[0, :] = u[0, :]
    for idk in range(n_k - 1):
        # the overlap is integral over unit cell of <u'(k)|u(k+dk)>
        m_old = (np.sum(np.conj(usmooth[idk, 0: round(Nx / x_cell)])
                        * u[idk + 1, 0: round(Nx / x_cell)], axis=-1)
                 / Nx * x_cell)  # 0: round(Nx / x_cell)   * x_cell
        # make the overlap in smooth function real
        usmooth[idk + 1, :] = u[idk + 1, :] * np.exp(-1j * np.angle(m_old))
    # calculate how the function in k = 0 differs from k = 2pi
    # it should be e^(i 2pi x)
    xx = np.asarray(range(round(Nx / x_cell))) / Nx * x_cell  # / x_cell
    lamb = (np.sum(np.conj(usmooth[0, 0: round(Nx / x_cell)])
                   * usmooth[n_k - 1, 0: round(Nx / x_cell)]
                   * np.exp(1j * 2 * pi * xx[np.newaxis, :]), axis=-1)
            / Nx * x_cell)  # 0: round(Nx / x_cell)   * x_cell
    nks = np.linspace(0, n_k - 1, n_k)
    nks = nks[:, np.newaxis]
    # Distribute the multiplier among functions at kx in [0, 2pi]
    usmooth = np.multiply(usmooth,
                          np.exp(-1j * np.angle(lamb) * nks / (n_k - 1)))
    return usmooth


# number of points in x - space
Nx = 600
# number f unit cells
x_cell = 15
# number of k points
Nk = 400
# number of G points
NG = 401

# !!!! main loop
# x vector
x = np.asarray(range(Nx)) / Nx * x_cell
uk = np.empty((Nk, Nx), dtype=complex)  # columns: one x, rows: one k
for nk in range(Nk):
    "for each k calculate coeffitients and then remember them in a matrix"
    k = 2 * np.pi * nk / (Nk - 1)
    print(nk)
    # periodic part of Bloch funcion
    uk[nk, :] = blochperiodic(k, x, NG)

# Parallel trnsport to maka u(k) smooth
u_smooth = paraltransport(uk)
k = np.linspace(0, 2 * pi, Nk)
# Calculate Bloch function
psi = u_smooth * np.exp(1j * k[:, np.newaxis] * x[np.newaxis, :])

# calculate Wannier function as a sum over k
Wannier = np.sum(psi[0:Nk - 1, :], axis=0) / (Nk - 1)


with open('WannieFromBloch.pickle', 'wb') as f:
    pickle.dump([Wannier, Nx, x_cell, NG], f)

with open('SmoothBloch.pickle', 'wb') as f:
    pickle.dump([psi], f)
