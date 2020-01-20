"""
 Created by alexandra at 15.01.20 15:55
 
 A module with hopf hamiltonians
"""


import numpy as np
from math import pi


def scalarprod(a, b):
    """Scalar product of two stackes of wavefunctions of the same size
    Returns a stack of <a[i,j,...,:]| b[i,j,...,:]>"""
    prod = np.sum(np.conj(a) * b, axis=-1)
    return prod


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

    hx = 2 * (np.sin(kx) * np.sin(kz)
              + np.sin(ky) * (np.cos(kx) + np.cos(ky) + np.cos(kz) - 3 + m))
    hy = 2 * (- np.sin(ky) * np.sin(kz)
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

    hx = 2 * (np.sin(kx) * np.sin(kz)
              + np.sin(ky) * (np.cos(kx) + np.cos(ky) + np.cos(kz) - 3 + m))
    hy = 2 * (- np.sin(ky) * np.sin(kz)
              + np.sin(kx) * (np.cos(kx) + np.cos(ky) + np.cos(kz) - 3 + m))
    hz = (np.sin(kx) ** 2 + np.sin(ky) ** 2
          - np.sin(kz) ** 2 - (
                      np.cos(kx) + np.cos(ky) + np.cos(kz) - 3 + m) ** 2)

    hx = hx[:, :, :, np.newaxis, np.newaxis]
    hy = hy[:, :, :, np.newaxis, np.newaxis]
    hz = hz[:, :, :, np.newaxis, np.newaxis]

    return hx * sigmax + hy * sigmay + hz * sigmaz


def parallel_transport_1d(u, n):
    """Perform parallel transport of vector len=n in 1 direction for a
    stack of vectors"""
    for nk in range(0, n - 1):
        m_old = scalarprod(u[..., nk, :], u[..., nk + 1, :])
        u[..., nk + 1, :] = (u[..., nk + 1, :]
                             * np.exp(-1j * np.angle(m_old[..., np.newaxis])))

    return u


def smooth_gauge(u):
    """Make parallel transport for eig-f u to get the smooth gauge in 3D BZ"""
    nx, ny, nz, vect = u.shape

    # First of all make parallel transport in kx direction for ky=kz=0
    u[:, 0, 0, :] = parallel_transport_1d(u[:, 0, 0, :], nx)

    # Make function periodic in kx direction
    # The function gains the multiplier
    lamb = scalarprod(u[0, 0, 0, :], u[nx - 1, 0, 0, :])

    nxs = np.linspace(0, nx - 1, nx) / (nx - 1)
    # Distribute the multiplier among functions at kx in [0, 2pi]
    u[:, 0, 0, :] = u[:, 0, 0, :] * np.exp(
        - 1j * np.angle(lamb) * nxs[:, np.newaxis])

    # For all kx make parallel transport along ky
    u[:, :, 0, :] = parallel_transport_1d(u[:, :, 0, :], ny)

    # The function gains the multiplier
    lamb2 = scalarprod(u[:, 0, 0, :], u[:, ny - 1, 0, :])

    # Get the phase of lambda
    langle2 = np.angle(lamb2)

    # Construct smooth phase of lambda (without 2pi jumps)
    for nkx in range(0, nx - 1):
        if np.abs(langle2[nkx + 1] - langle2[nkx]) > pi:
            langle2[nkx + 1: nx] = (
                    langle2[nkx + 1: nx]
                    - np.sign(langle2[nkx + 1] - langle2[nkx]) * (2 * pi))

    nys = np.linspace(0, ny - 1, ny) / (ny - 1)
    # Distribute the multiplier among functions at ky in [0, 2pi]
    u[:, :, 0, :] = (u[:, :, 0, :] * np.exp(
        - 1j * langle2[:, np.newaxis, np.newaxis]
        * nys[np.newaxis, :, np.newaxis]))

    # For all kx, ky make parallel transport along kz
    u = parallel_transport_1d(u, nz)

    lamb3 = scalarprod(u[:, :, 0, :], u[:, :, nz - 1, :])
    langle3 = np.angle(lamb3)

    # Langle3 = np.where(Langle3 < 0, Langle3 + 2 * pi, Langle3)

    # First make the lambda phase smooth along x-axis
    for nkx in range(0, nx - 1):
        jump = (np.abs(langle3[nkx + 1, :] - langle3[nkx, :])
                > pi * np.ones(ny))
        langlechange = (jump * np.sign(langle3[nkx + 1, :] - langle3[nkx, :])
                        * (2 * pi))
        langle3[nkx + 1: nx, :] = (langle3[nkx + 1: nx, :] -
                                   langlechange[np.newaxis, :])

    # Then make the phase smooth along y-axis similar for all x
    for nky in range(0, ny - 1):
        if np.abs(langle3[0, nky + 1] - langle3[0, nky]) > pi:
            langle3[:, nky + 1: ny] = (
                    langle3[:, nky + 1: ny]
                    - np.sign(langle3[0, nky + 1] - langle3[0, nky]) * (2 * pi))

    nzs = np.linspace(0, nz - 1, nz) / (nz - 1)
    # Distribute the multiplier among functions at kz in [0, 2pi]
    u = (u * np.exp(- 1j * langle3[:, :, np.newaxis, np.newaxis]
                    * nzs[np.newaxis, np.newaxis, :, np.newaxis]))

    return u


def hopf_invariant(u):
    """Calculate Hopf invariant for the band u"""
    # Constract the overlaps between neighbor points in all possible directions
    uxy1 = scalarprod(u[:-1, :-1, :-1, :], u[1:, :-1, :-1, :])
    uxy2 = scalarprod(u[1:, :-1, :-1, :], u[1:, 1:, :-1, :])
    uxy3 = scalarprod(u[1:, 1:, :-1, :], u[:-1, 1:, :-1, :])

    uyz1 = scalarprod(u[:-1, :-1, :-1, :], u[:-1, 1:, :-1, :])
    uyz2 = scalarprod(u[:-1, 1:, :-1, :], u[:-1, 1:, 1:, :])
    uyz3 = scalarprod(u[:-1, 1:, 1:, :], u[:-1, :-1, 1:, :])

    uzx1 = scalarprod(u[:-1, :-1, :-1, :], u[:-1, :-1, 1:, :])
    uzx2 = scalarprod(u[:-1, :-1, 1:, :], u[1:, :-1, 1:, :])
    uzx3 = scalarprod(u[1:, :-1, 1:, :], u[1:, :-1, :-1, :])

    # use the formula for F and A in terms of overlaps and calculate
    # sum_i(A_i*F_i)
    underhopf = (
            np.angle(uxy1 * uxy2 * uxy3 * np.conj(uyz1)) * np.angle(uzx1)
            + np.angle(uyz1 * uyz2 * uyz3 * np.conj(uzx1)) * np.angle(uxy1)
            + np.angle(uzx1 * uzx2 * uzx3 * np.conj(uxy1)) * np.angle(uyz1))

    # Hopf invariant is a sum of A*F over the whole BZ
    return - np.sum(underhopf)/(2*pi)**2


