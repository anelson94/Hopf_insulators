# -*- coding: utf-8 -*-
"""
Created on Thu Jun 28 13:34:39 2018

@author: aleksandra

Calculate Chern numbers of 2D cuts of the BZ of the Hopf insulator
"""

import numpy as np
from math import pi
import matplotlib.pyplot as plt
import hopfham


def gap_check(e, delta):
    """Check if spectrum has a gap less than 2*delta and return the positions"""
    small_gap = np.isclose(e, 0, atol=delta)

    print('Gap closes at points:')
    where_small = np.where(small_gap)
    print(list(zip(where_small[0], where_small[1], where_small[2])))


def chern_number(u):
    """Calculate Chern numbers of band u: Cx, Cy, Cz as functions of kx, ky, kz
    correspondingly"""

    ovlp_x = np.sum(
        np.conj(u[:-1, :-1, :-1, :]) * u[1:, :-1, :-1, :], axis=-1)
    ovlp_y = np.sum(
        np.conj(u[:-1, :-1, :-1, :]) * u[:-1, 1:, :-1, :], axis=-1)
    ovlp_z = np.sum(
        np.conj(u[:-1, :-1, :-1, :]) * u[:-1, :-1, 1:, :], axis=-1)
    ovlp_xy = np.sum(
        np.conj(u[1:, :-1, :-1, :])
        * u[1:, 1:, :-1, :], axis=-1)
    ovlp_xz = np.sum(
        np.conj(u[1:, :-1, :-1, :])
        * u[1:, :-1, 1:, :], axis=-1)
    ovlp_yx = np.sum(
        np.conj(u[:-1, 1:, :-1, :])
        * u[1:, 1:, :-1, :], axis=-1)
    ovlp_yz = np.sum(
        np.conj(u[:-1, 1:, :-1, :])
        * u[:-1, 1:, 1:, :], axis=-1)
    ovlp_zx = np.sum(
        np.conj(u[:-1, :-1, 1:, :])
        * u[1:, :-1, 1:, :], axis=-1)
    ovlp_zy = np.sum(
        np.conj(u[:-1, :-1, 1:, :])
        * u[:-1, 1:, 1:, :], axis=-1)

    chern_x = - np.sum(np.angle(ovlp_y
                                * ovlp_yz
                                * ovlp_zy.conjugate()
                                * ovlp_z.conjugate()),
                       axis=(1, 2)) / (2 * pi)
    chern_y = - np.sum(np.angle(ovlp_z
                                * ovlp_zx
                                * ovlp_xz.conjugate()
                                * ovlp_x.conjugate()),
                       axis=(0, 2)) / (2 * pi)
    chern_z = - np.sum(np.angle(ovlp_x
                                * ovlp_xy
                                * ovlp_yx.conjugate()
                                * ovlp_y.conjugate()),
                       axis=(0, 1)) / (2 * pi)
    return chern_x, chern_y, chern_z


def main():
    m = 1
    nx = 100
    ny = 100
    nz = 100

    kx = np.linspace(0, 2*pi, nx)
    ky = np.linspace(0, 2*pi, ny)
    kz = np.linspace(0, 2*pi, nz)

    kkx, kky, kkz = np.meshgrid(kx, ky, kz, indexing='ij')
    hamilt = hopfham.ham_mrw(m, kkx, kky, kkz)

    [e, u] = np.linalg.eigh(hamilt)

    # Occupied states correspond to smaller eigenvalues
    uocc = u[:, :, :, :, 0]

    # Check gap in the spectrum
    gap_check(e[:, :, :, 0], 0.001)

    c_x, c_y, c_z = chern_number(uocc)
    print(c_x.shape, kx.shape)

    #
    figy, ax = plt.subplots(1, 3)
    ax[0].plot(kx[:-1], c_x)
    ax[1].plot(ky[:-1], c_y)
    ax[2].plot(kz[:-1], c_z)
    plt.show()


if __name__ == '__main__':
    main()
