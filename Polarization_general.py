"""
 Created by alexandra at 15.01.20 13:14

 Calculate the polarization of 1D cut of the BZ for a generalized
 Hopf Hamiltonian
"""

import numpy as np
from math import pi
import hopfham
from HopfChernNumbers import gap_check
import matplotlib.pyplot as plt


def projector(u):
    """Construct a projector from a wavefunction
    (array in kx, ky, kz directions)"""
    proj = (np.conj(u[:, :, :, :, np.newaxis])
            * u[:, :, :, np.newaxis, :])
    print(proj.shape)
    return proj


def polarization_z(pr):
    """Polarization of a 1D cut of a BZ in kz direction"""
    nz = pr.shape[2]
    all_pr = np.identity(2, dtype=complex)
    all_pr = all_pr[np.newaxis, np.newaxis, :, :]
    for idz in range(nz):
        all_pr = np.matmul(all_pr, pr[:, :, idz, :, :])
    pol = np.angle(np.trace(all_pr, axis1=-2, axis2=-1)) / 2 / pi
    pol = np.where(pol > 0, pol, pol + 1)

    return pol


def main():
    m = 5
    nx = 100
    ny = 100
    nz = 100
    plotpol = 0

    kx = np.linspace(0, 2 * pi, nx)
    ky = np.linspace(0, 2 * pi, ny)
    kz = np.linspace(0, 2 * pi, nz)

    kkx, kky, kkz = np.meshgrid(kx, ky, kz, indexing='ij')
    hamilt = hopfham.ham_mrw(m, kkx, kky, kkz)

    [e, u] = np.linalg.eigh(hamilt)

    # Occupied states correspond to smaller eigenvalues
    uocc = u[:, :, :, :, 0]

    gap_check(e[:, :, :, 0], 0.001)

    pr = projector(uocc)

    pol_z = polarization_z(pr)
    print(np.max(pol_z) - np.min(pol_z))
    if plotpol == 1:
        plt.imshow(pol_z)
        plt.colorbar()
        plt.show()


if __name__ == '__main__':
    main()
