"""
 Created by alexandra at 15.01.20 15:55
 
 A module with hopf hamiltonians
"""


import numpy as np
from math import pi
from time import time as tm
import matplotlib.pyplot as plt



def scalarprod(a, b):
    """Scalar product of two stackes of wavefunctions of the same size
    Returns a stack of <a[i,j,...,:]| b[i,j,...,:]>"""
    prod = np.sum(np.conj(a) * b, axis=-1)
    return prod


# Functions to define Hamiltonians for different models

def ham(kx, ky, kz, model, **kwargs):
    """General Hamiltonian function"""
    # Get arrays of hx, hy, hz vectors for all momenta in BZ (kx, ky, kz)
    if callable(model):
        model = model(kx, ky, kz, **kwargs)

    # Pauli matrices
    sigmax = np.array([[0, 1], [1, 0]])
    sigmay = np.array([[0, -1j], [1j, 0]])
    sigmaz = np.array([[1, 0], [0, -1]])

    # Extract hx, hy, hz from the dict and broadcast to match with
    # Pauli matrices
    hx = model['hx'][..., np.newaxis, np.newaxis]
    hy = model['hy'][..., np.newaxis, np.newaxis]
    hz = model['hz'][..., np.newaxis, np.newaxis]

    return hx * sigmax + hy * sigmay + hz * sigmaz


def model_mrw(kx, ky, kz, m=1):
    """Moore Ran Wen model of Hamiltonian"""
    # Define coefficients in front of Pauli matrices
    hx = 2 * (np.sin(kx) * np.sin(kz)
              + np.sin(ky) * (np.cos(kx) + np.cos(ky) + np.cos(kz) - 3 + m))
    hy = 2 * (- np.sin(ky) * np.sin(kz)
              + np.sin(kx) * (np.cos(kx) + np.cos(ky) + np.cos(kz) - 3 + m))
    hz = (np.sin(kx) ** 2 + np.sin(ky) ** 2
          - np.sin(kz) ** 2 - (
                      np.cos(kx) + np.cos(ky) + np.cos(kz) - 3 + m) ** 2)
    # Return them as a dict
    return {'hx': hx, 'hy': hy, 'hz': hz}


def model_mrw_norm(kx, ky, kz, m=1):
    """Moore Ran Wen model of Hamiltonian"""
    # Define coefficients in front of Pauli matrices
    hx = 2 * (np.sin(kx) * np.sin(kz)
              + np.sin(ky) * (np.cos(kx) + np.cos(ky) + np.cos(kz) - 3 + m))
    hy = 2 * (- np.sin(ky) * np.sin(kz)
              + np.sin(kx) * (np.cos(kx) + np.cos(ky) + np.cos(kz) - 3 + m))
    hz = (np.sin(kx) ** 2 + np.sin(ky) ** 2
          - np.sin(kz) ** 2 - (
                      np.cos(kx) + np.cos(ky) + np.cos(kz) - 3 + m) ** 2)

    # Define normalization
    lamb = (np.sin(kx) ** 2 + np.sin(ky) ** 2
            + np.sin(kz) ** 2 + (
                      np.cos(kx) + np.cos(ky) + np.cos(kz) - 3 + m) ** 2)
    # Return normalized coefficients as a dict
    return {'hx': hx / lamb, 'hy': hy / lamb, 'hz': hz / lamb}


def model_edgeconst(kx, ky, kz):
    """Model of a Hamiltonian that is constant on the edge of the BZ"""
    # Pauli matrices
    sigmax = np.array([[0, 1], [1, 0]])
    sigmay = np.array([[0, -1j], [1j, 0]])
    sigmaz = np.array([[1, 0], [0, -1]])

    # Define psi, theta and phi angles to transform cube to a sphere
    # This allows to get constant values on the edges of the cube.
    # TODO understand how
    k_norm = np.sqrt(kx ** 2 + ky ** 2 + kz ** 2)
    psi = np.maximum(np.abs(kx),
                     np.maximum(np.abs(ky), np.abs(kz)))

    # remove 0 elements from norm for further division
    k_norm_fixed = np.where(k_norm > 0, k_norm, 1)  # delta_k / 2
    # Where k_norm = 0 define theta = pi/2
    theta = np.where(k_norm > 0, np.arccos(kz / k_norm_fixed), pi / 2)

    # remove 0 elements from kx for further division
    kx_fixed = np.where(np.abs(kx) > 0, kx, 1)
    # Where kx = 0 define phi = pi/2 with a pi shift for negative ky
    phi = np.where(
        np.abs(kx) > 0,
        np.arctan(ky / kx_fixed) + pi * np.heaviside(-kx, 0),
        pi / 2 + pi * np.heaviside(-ky, 0)
    )

    # Define 4 components of a vector on S3 (Re and Im parts of c1, c2)
    c1 = np.cos(psi) + 1j * np.sin(psi) * np.cos(theta)
    c2 = np.sin(psi) * np.sin(theta) * (np.cos(phi) + 1j * np.sin(phi))

    # Define coefficients in front of Pauli matrices
    c = np.stack((c1, c2), axis=-1)

    hx = np.sum(np.matmul(np.conj(c), sigmax) * c, axis=-1)
    hy = np.sum(np.matmul(np.conj(c), sigmay) * c, axis=-1)
    hz = np.sum(np.matmul(np.conj(c), sigmaz) * c, axis=-1)

    # Return them as a dict
    return {'hx': hx, 'hy': hy, 'hz': hz}


def model_mrw_maps(kx, ky, kz, m=1):
    """Moore Ran Wen model constructed from maps T3->S3->S2"""
    # Define a vector on S3
    c1, c2, c3, c4 = map_t3s3_mrw(kx, ky, kz, m)
    # Define a vector on S2
    hx, hy, hz = map_s3s2(c1, c2, c3, c4)
    return {'hx': hx, 'hy': hy, 'hz': hz}


def model_mrw_maps_rotated(kx, ky, kz, m=1, alpha=pi/2):
    """Rotated Moore Ran Wen model constructed from maps T3->S3->S2"""
    # Define a vector on S3
    c1, c2, c3, c4 = map_t3s3_mrw(kx, ky, kz, m)
    # Rotate a vector on S3
    c1, c2, c3, c4 = s3_rotate(c1, c2, c3, c4, alpha)
    # Define a vector on S2
    hx, hy, hz = map_s3s2(c1, c2, c3, c4)
    return {'hx': hx, 'hy': hy, 'hz': hz}


def model_edgeconst_maps(kx, ky, kz):
    """Model that is constant on the edge of the BZ
    from maps T3->S3->S2"""
    # Define a vector on S3
    c1, c2, c3, c4 = map_t3s3_edgeconst(kx, ky, kz)
    # Define a vector on S2
    hx, hy, hz = map_s3s2(c1, c2, c3, c4)
    return {'hx': hx, 'hy': hy, 'hz': hz}


def model_edgeconst_maps_rotated(kx, ky, kz, alpha):
    """Rotated model that is constant on the edge of the BZ
    from maps T3->S3->S2"""
    # Define a vector on S3
    c1, c2, c3, c4 = map_t3s3_edgeconst(kx, ky, kz)
    # Rotate a vector on S3
    c1, c2, c3, c4 = s3_rotate(c1, c2, c3, c4, alpha)
    # Define a vector on S2
    hx, hy, hz = map_s3s2(c1, c2, c3, c4)
    return {'hx': hx, 'hy': hy, 'hz': hz}


def map_t3s3_mrw(kx, ky, kz, m):
    """Moore Ran Wen map from T3 to S3"""
    c1 = np.sin(kx)
    c2 = np.sin(ky)
    c3 = np.sin(kz)
    c4 = np.cos(kx) + np.cos(ky) + np.cos(kz) + m - 3
    return c1, c2, c3, c4


def map_t3s3_edgeconst(kx, ky, kz):
    """Map from T3 to S3 that is constant on the edge of the BZ"""
    # Define psi, theta and phi angles to transform cube to a sphere
    # This allows to get constant values on the edges of the cube.
    k_norm = np.sqrt(kx ** 2 + ky ** 2 + kz ** 2)
    psi = np.maximum(np.abs(kx),
                     np.maximum(np.abs(ky), np.abs(kz)))

    # remove 0 elements from norm for further division
    k_norm_fixed = np.where(k_norm > 0, k_norm, 1)
    # Where k_norm = 0 define theta = pi/2
    theta = np.where(k_norm > 0, np.arccos(kz / k_norm_fixed), pi / 2)

    # remove 0 elements from kx for further division
    kx_fixed = np.where(np.abs(kx) > 0, kx, 1)
    # Where kx = 0 define phi = pi/2 with a pi shift for negative ky
    phi = np.where(
        np.abs(kx) > 0,
        np.arctan(ky / kx_fixed) + pi * np.heaviside(-kx, 0),
        pi / 2 + pi * np.heaviside(-ky, 0)
    )
    # Define 4 components of a vector on S3
    c1 = np.cos(psi)
    c2 = np.sin(psi) * np.cos(theta)
    c3 = np.sin(psi) * np.sin(theta) * np.cos(phi)
    c4 = np.sin(psi) * np.sin(theta) * np.sin(phi)
    return c1, c2, c3, c4


def s3_rotate(c1, c2, c3, c4, alpha):
    """Rotate coordinates on S3 by angles alpha"""
    # Not yet generic formula. Rotate c1 and c3
    c1_rot = c1 * np.cos(alpha) - c3 * np.sin(alpha)
    c2_rot = c2
    c3_rot = c1 * np.sin(alpha) + c3 * np.cos(alpha)
    c4_rot = c4
    return c1_rot, c2_rot, c3_rot, c4_rot


def map_s3s2(c1, c2, c3, c4):
    """Normalized Hopf map from S3 to S2"""
    # Define complex 2-vector from a 4-vector on S3
    z1 = c1 + 1j * c2
    z2 = c3 + 1j * c4

    # Pauli matrices
    sigmax = np.array([[0, 1], [1, 0]])
    sigmay = np.array([[0, -1j], [1j, 0]])
    sigmaz = np.array([[1, 0], [0, -1]])

    # Calculate coefficients of a vector on S2 using formula
    # h_i=c^+ sigma_i c
    z = np.stack((z1, z2), axis=-1)

    hx = np.real(np.sum(np.matmul(np.conj(z), sigmax) * z, axis=-1))
    hy = np.real(np.sum(np.matmul(np.conj(z), sigmay) * z, axis=-1))
    hz = np.real(np.sum(np.matmul(np.conj(z), sigmaz) * z, axis=-1))

    # Normalization coefficient
    lamb = np.sqrt(hx**2 + hy**2 + hz**2)

    # Return normalized components of a vector on S2
    return hx / lamb, hy / lamb, hz / lamb


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

    # Return a Hamiltonian 2*2 matrix for a k-grid
    return hx * sigmax + hy * sigmay + hz * sigmaz


def ham_bzedge_constant(kx, ky, kz):
    """Construct a Hamiltonian that is constant on the edge of the BZ"""
    # Pauli matrices
    sigmax = np.array([[0, 1], [1, 0]])
    sigmay = np.array([[0, -1j], [1j, 0]])
    sigmaz = np.array([[1, 0], [0, -1]])

    k_norm = np.sqrt(kx ** 2 + ky ** 2 + kz ** 2)
    psi = np.maximum(np.abs(kx),
                     np.maximum(np.abs(ky), np.abs(kz)))

    # delta_k = kx[1, 0, 0] - kx[0, 0, 0]

    # remove 0 elements from norm for further division
    k_norm_fixed = np.where(k_norm > 0, k_norm, 1)  # delta_k / 2
    theta = np.where(k_norm > 0, np.arccos(kz/k_norm_fixed), pi/2)

    kx_fixed = np.where(np.abs(kx) > 0, kx, 1)
    phi = np.where(
        np.abs(kx) > 0,
        np.arctan(ky / kx_fixed) + pi * np.heaviside(-kx, 0),
        pi / 2 + pi * np.heaviside(-ky, 0)
    )

    c1 = np.cos(psi) + 1j * np.sin(psi) * np.cos(theta)
    c2 = np.sin(psi) * np.sin(theta) * (np.cos(phi) + 1j * np.sin(phi))

    c = np.stack((c1, c2), axis=-1)

    hx = np.sum(np.matmul(np.conj(c), sigmax) * c, axis=-1)
    hy = np.sum(np.matmul(np.conj(c), sigmay) * c, axis=-1)
    hz = np.sum(np.matmul(np.conj(c), sigmaz) * c, axis=-1)

    # Return a Hamiltonian 2*2 matrix for a k-grid
    return (hx[..., np.newaxis, np.newaxis] * sigmax
            + hy[..., np.newaxis, np.newaxis] * sigmay
            + hz[..., np.newaxis, np.newaxis] * sigmaz)


def parallel_transport_1d(u, n):
    """Perform parallel transport of vector len=n in 1 direction for a
    stack of vectors

    Each next element is multiplied by a phase overlap with previous element
    with reverted sign. A smooth phase is obtained in the chosen direction."""
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

    # A vector that helps to distribute the multiplier along BZ
    nxs = np.linspace(0, nx - 1, nx) / (nx - 1)
    # Distribute the multiplier among functions at kx in [0, 2pi]
    # Each point gets portion different by lamb / (nx - 1) from the
    # previous point
    u[:, 0, 0, :] = u[:, 0, 0, :] * np.exp(
        - 1j * np.angle(lamb) * nxs[:, np.newaxis])

    # For all kx make parallel transport along ky
    u[:, :, 0, :] = parallel_transport_1d(u[:, :, 0, :], ny)

    # The function gains the multiplier
    lamb2 = scalarprod(u[:, 0, 0, :], u[:, ny - 1, 0, :])

    # Get the phase of lambda
    langle2 = np.angle(lamb2)

    # Construct smooth phase of lambda (without 2pi jumps)
    # Add 2pi shift at each jump
    for nkx in range(0, nx - 1):
        if np.abs(langle2[nkx + 1] - langle2[nkx]) > pi:
            langle2[nkx + 1: nx] = (
                    langle2[nkx + 1: nx]
                    - np.sign(langle2[nkx + 1] - langle2[nkx]) * (2 * pi))

    # A vector that helps to distribute the multiplier along BZ
    nys = np.linspace(0, ny - 1, ny) / (ny - 1)
    # Distribute the multiplier among functions at ky in [0, 2pi]
    u[:, :, 0, :] = (u[:, :, 0, :] * np.exp(
        - 1j * langle2[:, np.newaxis, np.newaxis]
        * nys[np.newaxis, :, np.newaxis]))

    # For all kx, ky make parallel transport along kz
    u = parallel_transport_1d(u, nz)

    # The function gains a multiplier
    lamb3 = scalarprod(u[:, :, 0, :], u[:, :, nz - 1, :])
    # Define a phase
    langle3 = np.angle(lamb3)

    # Langle3 = np.where(Langle3 < 0, Langle3 + 2 * pi, Langle3)

    # First make the lambda phase smooth (remove 2pi jumps) along x-axis
    for nkx in range(0, nx - 1):
        jump = (np.abs(langle3[nkx + 1, :] - langle3[nkx, :])
                > pi * np.ones(ny))
        langlechange = (jump * np.sign(langle3[nkx + 1, :] - langle3[nkx, :])
                        * (2 * pi))
        langle3[nkx + 1: nx, :] = (langle3[nkx + 1: nx, :] -
                                   langlechange[np.newaxis, :])

    # Then make the phase smooth (remove 2pi jumps) along y-axis similar
    # for all x
    for nky in range(0, ny - 1):
        if np.abs(langle3[0, nky + 1] - langle3[0, nky]) > pi:
            langle3[:, nky + 1: ny] = (
                    langle3[:, nky + 1: ny]
                    - np.sign(langle3[0, nky + 1] - langle3[0, nky]) * (2 * pi))

    # A vector that helps to distribute the multiplier along BZ
    nzs = np.linspace(0, nz - 1, nz) / (nz - 1)
    # Distribute the multiplier among functions at kz in [0, 2pi]
    u = (u * np.exp(- 1j * langle3[:, :, np.newaxis, np.newaxis]
                    * nzs[np.newaxis, np.newaxis, :, np.newaxis]))

    return u


def hopf_invariant(u):
    """Calculate Hopf invariant for the band u"""
    # Construct the overlaps between neighbor points in all possible directions
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

    # Hopf invariant is a sum of A*F over the whole BZ. '-' sign is a
    # matter of definition
    return - np.sum(underhopf)/(2*pi)**2


def mesh_make(nx, ny, nz):
    """Make a 3D mesh for coordinate vectors with (nx, ny, nz) points"""
    # Define k vector components in the interval (-pi, pi)
    kx = np.linspace(-pi, pi, nx)
    ky = np.linspace(-pi, pi, ny)
    kz = np.linspace(-pi, pi, nz)

    # Return a meshgrid
    return np.meshgrid(kx, ky, kz, indexing='ij')


def main():
    """Test functions"""
    nx = 101
    kx, ky, kz = mesh_make(nx, nx, nx)
    hamilt = ham(kx, ky, kz, model_edgeconst_maps)
    hamilt_test = ham(kx, ky, kz, model_edgeconst)
    print(np.allclose(hamilt, hamilt_test))
    # hamilt = ham_bzedge_constant(kx, ky, kz)
    # e, u = np.linalg.eigh(hamilt)
    # uocc = u[..., 0]
    #
    # u_smooth = smooth_gauge(uocc)


if __name__ == '__main__':
    main()
