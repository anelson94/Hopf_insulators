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
import pickle


def projector(u):
    """Construct a projector from a wavefunction
    (array in kx, ky, kz directions)"""
    proj = (np.conj(u[:, :, :, :, np.newaxis])
            * u[:, :, :, np.newaxis, :])
    print(proj.shape)
    return proj


def polarization_z(pr, between01):
    """Polarization of a 1D cut of a BZ in kz direction"""
    nz = pr.shape[2]
    all_pr = np.identity(2, dtype=complex)
    all_pr = all_pr[np.newaxis, np.newaxis, :, :]
    for idz in range(nz):
        all_pr = np.matmul(all_pr, pr[:, :, idz, :, :])
    pol = np.angle(np.trace(all_pr, axis1=-2, axis2=-1)) / 2 / pi
    if between01 == 1:
        pol = np.where(pol > 0, pol, pol + 1)

    return pol


def polarization_x(pr, between01):
    """Polarization of a 1D cut of a BZ in kx direction"""
    nx = pr.shape[0]
    all_pr = np.identity(2, dtype=complex)
    all_pr = all_pr[np.newaxis, np.newaxis, :, :]
    for idx in range(nx):
        all_pr = np.matmul(all_pr, pr[idx, :, :, :, :])
    pol = np.angle(np.trace(all_pr, axis1=-2, axis2=-1)) / 2 / pi
    if between01 == 1:
        pol = np.where(pol > 0, pol, pol + 1)

    return pol


def polarization_xz(pr, between01):
    """Polarization of a 1D cut of a BZ in kx+kz direction.
    The output is a 1d array."""
    nx = pr.shape[0]
    nz = pr.shape[2]
    if nx != nz:
        print('Dimensions in x and z directions should match')
        pass
    all_pr = np.identity(2, dtype=complex)
    all_pr = all_pr[np.newaxis, :, :]
    for id in range(nx):
        all_pr = np.matmul(all_pr, pr[id, :, id, :, :])
    pol = np.angle(np.trace(all_pr, axis1=-2, axis2=-1)) / 2 / pi
    if between01 == 1:
        pol = np.where(pol > 0, pol, pol + 1)

    return pol


def polarization_yz(pr, between01):
    """Polarization of a 1D cut of a BZ in ky+kz direction.
    The output is a 1d array."""
    ny = pr.shape[1]
    nz = pr.shape[2]
    if ny != nz:
        print('Dimensions in y and z directions should match')
        pass
    all_pr = np.identity(2, dtype=complex)
    all_pr = all_pr[np.newaxis, :, :]
    for id in range(ny):
        all_pr = np.matmul(all_pr, pr[:, id, id, :, :])
    pol = np.angle(np.trace(all_pr, axis1=-2, axis2=-1)) / 2 / pi
    if between01 == 1:
        pol = np.where(pol > 0, pol, pol + 1)

    return pol


def polarization_y(pr, between01):
    """Polarization of a 1D cut of a BZ in ky direction"""
    ny = pr.shape[1]
    all_pr = np.identity(2, dtype=complex)
    all_pr = all_pr[np.newaxis, np.newaxis, :, :]
    for idy in range(ny):
        all_pr = np.matmul(all_pr, pr[:, idy, :, :, :])
    pol = np.angle(np.trace(all_pr, axis1=-2, axis2=-1)) / 2 / pi
    if between01 == 1:
        pol = np.where(pol > 0, pol, pol + 1)

    return pol


def sym_break_draft(ampl, kx, ky, kz):
    """Try different symmetry breaking terms in Hamiltonian"""
    nx, ny, nz = kx.shape

    sigmax = np.array([[0, 1], [1, 0]])
    sigmay = np.array([[0, -1j], [1j, 0]])
    sigmaz = np.array([[1, 0], [0, -1]])

    # Pauli matrices for calculations at all (kx, ky, kz)
    sigmax = sigmax[np.newaxis, np.newaxis, np.newaxis, :, :]
    sigmay = sigmay[np.newaxis, np.newaxis, np.newaxis, :, :]
    sigmaz = sigmaz[np.newaxis, np.newaxis, np.newaxis, :, :]

    zero_k = np.zeros((nx, ny, nz))

    hx = ampl * (np.sin(kz) + np.cos(kz)) / 2  # 0.5 * np.cos(kx) +
    hy = ampl * np.cos(kz)  # 0.3 * np.cos(ky) * np.cos(kx) +
    hz = zero_k  # 0.2 * np.cos(kx) + 0.3 * np.sin(ky) + 0.4 * np.sin(kz)

    hx = hx[:, :, :, np.newaxis, np.newaxis]
    hy = hy[:, :, :, np.newaxis, np.newaxis]
    hz = hz[:, :, :, np.newaxis, np.newaxis]

    return hx * sigmax + hy * sigmay + hz * sigmaz


def hamiltonian_checkes(hamilt):
    """Check that old and new codes give the same results"""
    with open('HopfHamiltonian.pickle', 'rb') as f:
        hamilt_old = pickle.load(f)

    print('Two hamiltonians are equal:', np.allclose(hamilt, hamilt_old))
    print('Maximal difference between two Hamiltonians:',
          np.max(np.abs(hamilt - hamilt_old)))

    [e, u] = np.linalg.eigh(hamilt)

    [e2, u2] = np.linalg.eigh(hamilt)

    print('Two eigh calculations are equal:', np.allclose(u, u2))

    # with open('Hopfeigen.pickle', 'rb') as f:
    #     [E_old, u_old] = pickle.load(f)
    [E_old, u_old] = np.linalg.eigh(hamilt_old)
    uocc_old = u_old[:, :, :, :, 0]

    uocc = u[:, :, :, :, 0]
    uocc_smooth = hopfham.smooth_gauge(uocc)

    with open('Hopfsmoothstates.pickle', 'rb') as f:
        usmooth_old = pickle.load(f)

    print(np.allclose(uocc_smooth, usmooth_old))
    print(np.allclose(uocc, uocc_old))

    phase_diff = np.angle(uocc_smooth[0, 0, 0, 0]) - np.angle(usmooth_old[0, 0, 0, 0])
    usmooth_old = usmooth_old * np.exp(1j * phase_diff)
    print(phase_diff)
    print(u.shape, u_old.shape)
    print(np.allclose(uocc_smooth, usmooth_old))


def plot_pol_1d(n, pol, label):
    """Plot 1d cut of a polarization"""
    k = np.linspace(-pi, pi, n)
    plt.figure()
    plt.plot(k, pol, label=label)
    plt.plot(k, np.zeros((n, 1)), c='black')
    plt.legend()


def plot_pol_2d(pol, label):
    """Plot 2d imshow of a polarization"""
    plt.figure()
    plt.imshow(pol)
    plt.title(label)
    plt.colorbar()


def plot_spectrum(n, e):
    """Plot a 1d cut of a spectrum"""
    k = np.linspace(-pi, pi, n)
    plt.figure()
    plt.plot(k, e)


def main():
    nx = 101
    ny = 101
    nz = 101
    nx_half = round((nx - 1) / 2)
    ny_half = round((ny - 1) / 2)
    nz_half = round((nz - 1) / 2)
    plotpol = 0
    plot_spectrum = 0
    between01 = 0
    hamiltceck = 0
    breaksymmetry = 0
    invariant_calc = 0
    gapcheck = 1
    plot1d = ['px', 'py', 'pz', 'pxz', 'pyz']
    plot2d = ['px', 'pz', 'py']

    pol_dict = {'px': polarization_x, 'py': polarization_y,
                'pz': polarization_z,
                'pxz': polarization_xz, 'pyz': polarization_yz}

    n_dict = {'px': ny, 'py': nx, 'pz': nx, 'pxz': ny, 'pyz': nx}
    cut_dict = {'px': nz_half, 'py': nz_half, 'pz': ny_half}

    # mrw model
    # ham_args = {'model': hopfham.model_mrw, 'm': 1}
    # normalized mrw model
    # ham_args = {'model': hopfham.model_mrw_norm, 'm': 1}
    # mrw model from maps
    # ham_args = {'model': hopfham.model_mrw_maps, 'm': 1}
    # rotated mrw model from maps
    # ham_args = {'model': hopfham.model_mrw_maps_rotated, 'm': 1, 'alpha': pi/2}
    # rotated edge constant model from maps
    ham_args = {'model': hopfham.model_edgeconst_maps_rotated, 'alpha': pi/4}
    # edge constant model
    # ham_args = {'model': hopfham.model_edgeconst}

    kkx, kky, kkz = hopfham.mesh_make(nx, ny, nz)
    hamilt = hopfham.ham(kkx, kky, kkz, **ham_args)
    [e, u] = np.linalg.eigh(hamilt)
    # Occupied states correspond to smaller eigenvalues
    uocc = u[:, :, :, :, 0]

    if plot_spectrum == 1:
        kx = np.linspace(-pi, pi, nx)
        plt.figure()
        plt.plot(kx, e[:, 0, 0, 1])

    if invariant_calc == 1:
        # Smoothen the eigenstates
        uocc_smooth = hopfham.smooth_gauge(uocc)
        # Calculate the Hopf invariant
        print(hopfham.hopf_invariant(uocc_smooth))

    if gapcheck == 1:
        # Check if the spectrum is gapped
        gap_check(e[:, :, :, 0], 0.01)

    pr = projector(uocc)

    for p in plot1d:
        pol = pol_dict[p](pr, between01)
        if len(pol.shape) > 1:
            plot_pol_1d(n_dict[p], pol[:, cut_dict[p]], p)
        else:
            plot_pol_1d(n_dict[p], pol, p)

    for p in plot2d:
        pol = pol_dict[p](pr, between01)
        plot_pol_2d(pol, p)

    plt.show()

    # Make for loop to vary some parameter. n_ampl = 1 -> only one calculation
    n_ampl = 0
    ampl_start = 0.5
    ampl_delta = 0.1
    pol_z = np.empty((nx, ny, n_ampl), dtype=complex)
    for idx in range(n_ampl):
        ampl = ampl_start + idx * ampl_delta
        hamilt = hopfham.ham(kkx, kky, kkz, **ham_args)
        if breaksymmetry == 1:
            hamilt += sym_break_draft(ampl, kkx, kky, kkz)

        if hamiltceck == 1:
            hamiltonian_checkes(hamilt)

        [e, u] = np.linalg.eigh(hamilt)

        # Occupied states correspond to smaller eigenvalues
        uocc = u[:, :, :, :, 0]
        uocc_smooth = hopfham.smooth_gauge(uocc)
        print(hopfham.hopf_invariant(uocc_smooth))

        gap_check(e[:, :, :, 0], 0.01)

        pr = projector(uocc)

        pol_z[:, :, idx] = polarization_z(pr, between01)
    if plotpol == 2:
        pol_x = polarization_x(pr, between01)
        print(np.max(pol_x), np.min(pol_x))
    # print(np.max(pol_z) - np.min(pol_z))
    if plotpol == 1:
        kx = np.linspace(-pi, pi, nx)
        plt.plot(kx, pol_z[:, nx_half, :])
        plt.plot(kx, np.zeros((nx, 1)), c='black')
        plt.legend([round(ampl_start + idx * ampl_delta, 1)
                    for idx in range(n_ampl)], loc='upper right')
    if plotpol == 2:
        kx = np.linspace(-pi, pi, nx)
        fig, ax = plt.subplots(1, 2)
        ax[0].plot(kx, pol_z[:, 50])
        ax[0].plot(kx, np.zeros((nx, 1)), c='black')
        # ax[1].plot(kz, pol_x[:, 50])
        # ax[1].plot(kz, np.zeros((nx, 1)), c='black')
        # plt.plot(kx, np.ones((nx, 1)), c='black')
        imsh = ax[1].imshow(pol_x)
        plt.colorbar(imsh, ax=ax[1])
        # plt.imshow(pol_z)
        # plt.colorbar()
    if plot_spectrum == 1:
        kx = np.linspace(-pi, pi, nx)
        plt.figure()
        plt.plot(kx, e[:, 0, 0, 1])

    plt.show()


if __name__ == '__main__':
    main()
