"""
 Created by alexandra at 03.02.20 18:27

 Work with a Hopf model with Hamiltonian being constant on the edge of the BZ
"""

import numpy as np
import hopfham
from HopfChernNumbers import gap_check
import matplotlib.pyplot as plt
from math import pi


def plot_spectrum_2dcut(energy):
    plt.figure()
    plt.imshow(energy)
    plt.colorbar()
    plt.show()


def plot_spectrum_1dcut(energy, nk):
    k = np.linspace(-pi, pi, nk)
    plt.figure()
    plt.plot(k, energy)
    plt.plot(k, np.zeros((nk, 1)), 'k')
    plt.show()


def check_constant_edge(*h_list):
    """Check if the Hamiltonian is constant on the edge"""
    for h in h_list:
        h_ref = h[0, 0, 0]
        print(h_ref)
        print(
            np.allclose(h[0, :, :], h_ref)
            and np.allclose(h[-1, :, :], h_ref)
            and np.allclose(h[:, 0, :], h_ref)
            and np.allclose(h[:, -1, :], h_ref)
            and np.allclose(h[:, :, 0], h_ref)
            and np.allclose(h[:, :, -1], h_ref)
              )


def main():
    nx = 100
    ny = 100
    nz = 100
    n_half = round(nx / 2)
    m = 1

    kx, ky, kz = hopfham.mesh_make(nx, ny, nz)
    # hamilt = hopfham.ham_bzedge_constant(kx, ky, kz)
    # hamilt = hopfham.ham_mrw(m, kx, ky, kz)
    ham_dict = hopfham.model_edgeconst_maps_rotated(kx, ky, kz, 1.9)
    check_constant_edge(ham_dict['hx'], ham_dict['hy'], ham_dict['hz'])


    # hamilt = hopfham.ham(kx, ky, kz, hopfham.model_edgeconst_maps_rotated,
    #                      alpha=0)
    #
    # [e, u] = np.linalg.eigh(hamilt)
    #
    # gap_check(e[..., 1], 0.1)
    # uocc = u[..., 0]
    # uocc_smooth = hopfham.smooth_gauge(uocc)
    #
    # chi = hopfham.hopf_invariant(uocc_smooth)
    # print(chi)

    # plot_spectrum_2dcut(np.imag(hamilt[25, :, :, 0, 1]))
    # plot_spectrum_2dcut(e[20, ..., 1])
    # plot_spectrum_2dcut(e[:, 0, ..., 1])
    # plot_spectrum_2dcut(e[:, -1, ..., 1])
    # plot_spectrum_2dcut(e[..., 0, 1])
    # plot_spectrum_2dcut(e[..., -1, 1])
    # plot_spectrum_1dcut(e[n_half, n_half, :, 1], nx)


if __name__ == '__main__':
    main()
