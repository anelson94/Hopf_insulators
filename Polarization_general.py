"""
 Created by alexandra at 15.01.20 13:14

 Calculate the polarization of 1D cut of the BZ for a generalized
 Hopf Hamiltonian
"""

import numpy as np
from math import pi, ceil
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
    for idd in range(nx):
        all_pr = np.matmul(all_pr, pr[idd, :, idd, :, :])
    pol = np.angle(np.trace(all_pr, axis1=-2, axis2=-1)) / 2 / pi
    if between01 == 1:
        pol = np.where(pol > 0, pol, pol + 1)

    return pol


def polarization_xz_arbratio_1d(pr, n, m, between01):
    """Polarization of a 1D cut of a BZ in (n*kx, m*kz) direction.
    n, m must be coprime
    The output is a 1d array."""
    nx = pr.shape[0]
    nz = pr.shape[2]
    if nx != nz:
        print('Dimensions in x and z directions should match')
        pass
    all_pr = np.identity(2, dtype=complex)
    all_pr = all_pr[np.newaxis, :, :]
    for idd in range(nx):
        idx = (n * idd) % nx
        idz = (m * idd) % nz
        all_pr = np.matmul(all_pr, pr[idx, :, idz, :, :])
    pol = np.angle(np.trace(all_pr, axis1=-2, axis2=-1)) / 2 / pi
    if between01 == 1:
        pol = np.where(pol > 0, pol, pol + 1)

    return pol


def polarization_xz_arbratio(pr, n, m, between01, griddir='kz'):
    """Polarization of a 1D cut of a BZ in (n*kx, m*kz) direction.
    n, m must be coprime
    Output is on a (ky, kz) grid"""
    nx, ny, nz, _1, _2 = pr.shape
    if nx != nz:
        print('Dimensions in x and z directions should match')
        pass
    # Choose which grid to take
    if griddir == 'kz':
        # ky, kz grid
        nz_ratio = ceil(nz / n)
        pol = np.empty((ny, nz_ratio))
        for idbz in range(nz_ratio):
            all_pr_1d = np.identity(2, dtype=complex)
            all_pr_1d = all_pr_1d[np.newaxis, :, :]
            for idd in range(nx):
                idx = (n * idd) % nx
                idz = (m * idd + idbz) % nz
                all_pr_1d = np.matmul(all_pr_1d, pr[idx, :, idz, :, :])
            pol[:, idbz] = np.angle(np.trace(all_pr_1d,
                                             axis1=-2, axis2=-1)) / 2 / pi
    elif griddir == 'kx':
        # ky, kx/2 grid
        nx_ratio = ceil(nx / m)
        pol = np.empty((nx_ratio, ny))
        for idbx in range(nx_ratio):
            all_pr_1d = np.identity(2, dtype=complex)
            all_pr_1d = all_pr_1d[np.newaxis, :, :]
            for idd in range(nx):
                idx = (n * idd + idbx) % nx
                idz = (m * idd) % nz
                all_pr_1d = np.matmul(all_pr_1d, pr[idx, :, idz, :, :])
            pol[idbx, :] = np.angle(np.trace(all_pr_1d,
                                             axis1=-2, axis2=-1)) / 2 / pi

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
    for ind in range(ny):
        all_pr = np.matmul(all_pr, pr[:, ind, ind, :, :])
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


def eigs_get(nx, ny, nz, **ham_args):
    """Construct a Hamiltonian and get its eigenstates and eigenvalues"""
    kkx, kky, kkz = hopfham.mesh_make(nx, ny, nz)
    hamilt = hopfham.ham(kkx, kky, kkz, **ham_args)
    return np.linalg.eigh(hamilt)


def hamiltonian_checkes(hamilt):
    """Check that old and new codes give the same results"""
    with open('HopfHamiltonian.pickle', 'rb') as f:
        hamilt_old = pickle.load(f)

    print('Two hamiltonians are equal:', np.allclose(hamilt, hamilt_old))
    print('Maximal difference between two Hamiltonians:',
          np.max(np.abs(hamilt - hamilt_old)))

    [_e, u] = np.linalg.eigh(hamilt)

    [_e2, u2] = np.linalg.eigh(hamilt)

    print('Two eigh calculations are equal:', np.allclose(u, u2))

    # with open('Hopfeigen.pickle', 'rb') as f:
    #     [e_old, u_old] = pickle.load(f)
    [_e_old, u_old] = np.linalg.eigh(hamilt_old)
    uocc_old = u_old[:, :, :, :, 0]

    uocc = u[:, :, :, :, 0]
    uocc_smooth = hopfham.smooth_gauge(uocc)

    with open('Hopfsmoothstates.pickle', 'rb') as f:
        usmooth_old = pickle.load(f)

    print(np.allclose(uocc_smooth, usmooth_old))
    print(np.allclose(uocc, uocc_old))

    phase_diff = (np.angle(uocc_smooth[0, 0, 0, 0])
                  - np.angle(usmooth_old[0, 0, 0, 0]))
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
    # Set size of the grid
    nx = 101
    ny = 101
    nz = 101
    nx_half = round((nx - 1) / 2)
    ny_half = round((ny - 1) / 2)
    nz_half = round((nz - 1) / 2)

    # Parameters that define what to calculate:
    plot_spectr = 0  # whether to plot energy spectrum
    between01 = 1  # whether to bring polarization function to [0, 1] interval
    hamiltceck = 0  # whether to run hamiltonians_checks function to compare
    # different ways to construct hamiltonian
    invariant_calc = 0  # whether calculate the Hopf invariant
    gapcheck = 0  # wether to check the existence of the gap
    # List the directions in which the polarization is calculated
    plot1d = []  # 'px', 'py', 'pz', 'pxz', 'pyz'
    plot2d = []
    n_alpha = 0  # number of ham rotation angles in the for loop
    # for polarization in fractional direction.
    # if n_alpha == 0 no calculation is performed
    n_ampl = 0  # number of different amplitudes of the symmetry
    # breaking potential for z-polarization calculation
    # if n_ampl == 0 no calculation is performed
    n, m = 1, 2  # rational direction of the polarization (n*kx, m*kz)
    # If both values are zero no calculation is performed

    #
    # For each direction of polarization specify a function
    pol_dict = {'px': polarization_x, 'py': polarization_y,
                'pz': polarization_z,
                'pxz': polarization_xz, 'pyz': polarization_yz}

    # For each direction of polarization specify direction of 1d plot
    n_dict = {'px': ny, 'py': nx, 'pz': nx, 'pxz': ny, 'pyz': nx}
    # For each direction of polarization specify the place where to cut
    # for 1d plot
    cut_dict = {'px': nz_half, 'py': nz_half, 'pz': ny_half}

    # Hamiltonian models and parameters:
    # mrw model
    # ham_args = {'model': hopfham.model_mrw, 'm': 1}
    # normalized mrw model
    # ham_args = {'model': hopfham.model_mrw_norm, 'm': 1}
    # mrw model from maps
    # ham_args = {'model': hopfham.model_mrw_maps, 'm': 1}
    # rotated mrw model from maps
    ham_args = {'model': hopfham.model_mrw_maps_rotated, 'm': 1,
                'alpha': 0.4636476}
    # rotated edge constant model from maps
    # ham_args = {'model': hopfham.model_edgeconst_maps_rotated, 'alpha': pi/4}
    # edge constant model
    # ham_args = {'model': hopfham.model_edgeconst}

    # Calculate the eigenstates of the chosen model
    e, u = eigs_get(nx, ny, nz, **ham_args)
    # Choose the occupied state
    uocc = u[:, :, :, :, 0]
    # Plot energy spectrum for ky=kz=-pi cut and conductance band
    if plot_spectr == 1:
        kx = np.linspace(-pi, pi, nx)
        plt.figure()
        plt.plot(kx, e[:, 0, 0, 1])

    # Calculate the Hopf invariant
    if invariant_calc == 1:
        # Smoothen the eigenstates
        uocc_smooth = hopfham.smooth_gauge(uocc)
        # Calculate the Hopf invariant
        print(hopfham.hopf_invariant(uocc_smooth))

    if gapcheck == 1:
        # Check if the spectrum is gapped
        gap_check(e[:, :, :, 0], 0.01)

    # Calculate the projectors
    pr = projector(uocc)

    # For all directions from the list plot1d calculate the polarization and
    # plot a 1d cut
    for p in plot1d:
        pol = pol_dict[p](pr, between01)
        if len(pol.shape) > 1:
            plot_pol_1d(n_dict[p], pol[:, cut_dict[p]], p)
        else:
            plot_pol_1d(n_dict[p], pol, p)

    # For all directions from the list plot2d calculate the polarization and
    # plot a 2d color plot
    for p in plot2d:
        pol = pol_dict[p](pr, between01)
        plot_pol_2d(pol, p)

    # Check fractional direction polarization
    if n != 0 or m != 0:
        pol = polarization_xz_arbratio(pr, n, m, between01)
        # plot_pol_1d(ny, pol[:, nz_half], 'p_x,0z')
        # pol_check = polarization_z(pr, between01)
        # plot_pol_1d(ny, pol_check[:, nz_half], 'p_x')
        plot_pol_2d(pol, 'p_{}x,{}z'.format(n, m))
        # plot_pol_2d(pol_check[nx_half:, :], 'p_x')

        plt.show()

    # For loop to calculate (nx, mz) polarization for models rotated by
    # different angles
    # Check if the minimum value in the middle of the rBZ turns to 0 at some
    # angle
    # Result: no
    if n_alpha != 0:
        alpha_list = []  # write all the angles in the list
        min_list = []  # write all the minimum values in the list
        for idalpha in range(n_alpha):
            # for each angle caluclate eigenstate
            alpha = 0.665 + idalpha * 0.005
            alpha_list.append(alpha)
            ham_args = {'model': hopfham.model_mrw_maps_rotated, 'm': 1,
                        'alpha': alpha}
            [_e, u] = eigs_get(nx, ny, nz, **ham_args)
            # Occupied states correspond to smaller eigenvalues
            uocc = u[:, :, :, :, 0]
            # calculate all projectors
            pr = projector(uocc)
            # calculate polarization in (x, 2z) direction
            between01 = 1  # want all the values to be positive to find the
            # minimum in the center
            # second direction of the grid is kz by default
            pol = polarization_xz_arbratio(pr, 1, 2, between01)
            # calculate the minimum value in a square around
            # the center of the BZ
            min_list.append(np.min(pol[ny_half - 20:ny_half + 20,
                                   nz_half - 20:nz_half + 20]))
            # plot_pol_2d(pol, 'p_x,2z')
            # plt.show()

        # print minimum value of polarization
        # for each angle of the model rotation
        alpha_list = [round(elem, 3) for elem in alpha_list]
        min_list = [round(elem, 5) for elem in min_list]
        min_dict = {key: value for key, value in zip(alpha_list, min_list)}
        print(min_dict)

    # Make for loop to vary amplitude of a symmetry breaking term
    # n_ampl = 1 -> only one calculation
    # Set n_ampl = 0 to not perform this calculation
    if n_ampl != 0:
        ampl_start = 0.5
        ampl_delta = 0.1
        pol_z = np.empty((nx, ny, n_ampl), dtype=complex)
        kkx, kky, kkz = hopfham.mesh_make(nx, ny, nz)
        for idx in range(n_ampl):
            ampl = ampl_start + idx * ampl_delta
            hamilt = hopfham.ham(kkx, kky, kkz, **ham_args)
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

        kx = np.linspace(-pi, pi, nx)
        plt.plot(kx, pol_z[:, nx_half, :])
        plt.plot(kx, np.zeros((nx, 1)), c='black')
        plt.legend([round(ampl_start + idx * ampl_delta, 1)
                    for idx in range(n_ampl)], loc='upper right')
        plt.show()


if __name__ == '__main__':
    main()
