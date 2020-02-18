"""
 Created by alexandra at 14.02.20 12:04

 Module to calculate Berry curvature of the Hopf insulator
"""

import numpy as np
from math import pi
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import hopfham


def berrycurv_x(u):
    """Calculate Berry curvature in x direction of a wavefunction u.
    u is nx x ny x nz x 2"""
    nx, ny, nz, vect = u.shape
    # Calculate overlaps on the plaquette
    # <u(k)|u(k+dky)>
    u1 = np.sum(np.conj(u[:-1, :-1, :-1, :]) * u[:-1, 1:, :-1, :], axis=-1)
    # <u(k+dky)|u(k+dky+dkz)>
    u2 = np.sum(np.conj(u[:-1, 1:, :-1, :]) * u[:-1, 1:, 1:, :], axis=-1)
    # <u(k+dky+dkz)|u(k+dkz)>
    u3 = np.sum(np.conj(u[:-1, 1:, 1:, :]) * u[:-1, :-1, 1:, :], axis=-1)
    # <u(k+dkz)|u(k)>
    u4 = np.sum(np.conj(u[:-1, :-1, 1:, :]) * u[:-1, :-1, :-1, :], axis=-1)

    # The phase of the product of overlaps with a normalization factor
    # gives the Berry curvature
    return np.angle(u1 * u2 * u3 * u4) * (ny - 1) * (nz - 1) / (2 * pi) ** 2


def berrycurv_y(u):
    """Calculate Berry curvature in y direction of a wavefunction u.
    u is nx x ny x nz x 2"""
    nx, ny, nz, vect = u.shape
    # Calculate analogously to berrycurv_x but for (dkz, dkx) plaquette
    u1 = np.sum(np.conj(u[:-1, :-1, :-1, :]) * u[:-1, :-1, 1:, :], axis=-1)
    u2 = np.sum(np.conj(u[:-1, :-1, 1:, :]) * u[1:, :-1, 1:, :], axis=-1)
    u3 = np.sum(np.conj(u[1:, :-1, 1:, :]) * u[1:, :-1, :-1, :], axis=-1)
    u4 = np.sum(np.conj(u[1:, :-1, :-1, :]) * u[:-1, :-1, :-1, :], axis=-1)

    return np.angle(u1 * u2 * u3 * u4) * (nx - 1) * (nz - 1) / (2 * pi) ** 2


def berrycurv_z(u):
    """Calculate Berry curvature in z direction of a wavefunction u.
    u is nx x ny x nz x 2"""
    nx, ny, nz, vect = u.shape
    # Calculate analogously to berrycurv_x but for (dkx, dky) plaquette
    u1 = np.sum(np.conj(u[:-1, :-1, :-1, :]) * u[1:, :-1, :-1, :], axis=-1)
    u2 = np.sum(np.conj(u[1:, :-1, :-1, :]) * u[1:, 1:, :-1, :], axis=-1)
    u3 = np.sum(np.conj(u[1:, 1:, :-1, :]) * u[:-1, 1:, :-1, :], axis=-1)
    u4 = np.sum(np.conj(u[:-1, 1:, :-1, :]) * u[:-1, :-1, :-1, :], axis=-1)

    return np.angle(u1 * u2 * u3 * u4) * (nx - 1) * (ny - 1) / (2 * pi) ** 2


def dirandnorm(x, y):
    """Define the directions and module of vector field"""
    r = np.sqrt(x**2 + y**2)
    return r, x / r, y / r


def plot_berrycurv_xy(bx, by, bz, kz_cut, fig_params):
    """Plot the Berry curvature at a 2d cut of the BZ"""
    # Read the grid size from the berry arrays
    nx, ny, nz = bx.shape
    # Create a grid in (kx, ky) rBZ
    kx = np.linspace(-pi, pi, nx + 1)
    ky = np.linspace(-pi, pi, ny + 1)
    kxgrid, kygrid = np.meshgrid(kx[:-1], ky[:-1], indexing='ij')

    # Calculate norm and directions of the x, y components of the Berry curv
    bxy_abs, bx_angle, by_angle = dirandnorm(bx[:, :, kz_cut], by[:, :, kz_cut])

    # Create the figure object
    fig = plt.figure(figsize=fig_params['figsize'])
    ax = fig.add_axes(fig_params['axessize'])
    ax.set_xlabel('$k_x$', size=fig_params['fontsize'])
    ax.set_ylabel('$k_y$', size=fig_params['fontsize'], rotation=0)
    ax.xaxis.set_label_coords(fig_params['xlabel_x'], fig_params['xlabel_y'])
    ax.yaxis.set_label_coords(fig_params['ylabel_x'], fig_params['ylabel_y'])
    ax.set_xticks([-pi, 0, pi])
    ax.set_xticklabels(('$-\pi$', '$0$', '$\pi$'), 
                       size=fig_params['fonttickssize'])
    ax.set_yticks([-pi, 0, pi])
    ax.set_yticklabels(('$-\pi$', '$0$', '$\pi$'), 
                       size=fig_params['fonttickssize'])
    plt.xlim(-pi, pi)
    plt.ylim(-pi, pi)
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(fig_params['linewidth'])
    ax.tick_params(width=fig_params['linewidth'])

    # Define a colormap for x, y components of the Berry curvature
    cmap_bxy = colors.LinearSegmentedColormap.from_list("", 
                                                        ["lightgray", "red"])

    # Plot x, y components of the Berry curvature as a stream
    # Plot only every (st)th point
    # Color of the arrows defines their magnitude
    st = fig_params['streamplot_step']  # step to plot stream
    bxy_plot = ax.streamplot(
        kygrid[::st, ::st], kxgrid[::st, ::st],
        by_angle[::st, ::st], bx_angle[::st, ::st],
        color=bxy_abs[::st, ::st], cmap=cmap_bxy,
        density=fig_params['stream_density'],
        linewidth=fig_params['streamlinewidth'],
        arrowsize=fig_params['streamarrowsize'])

    # Define a colormap for z component of the Berry curvature
    cmap_bz = colors.LinearSegmentedColormap.from_list(
        "", ["royalblue", "white", (0.3, 0.5, 0.3)])
    # Other colors: "mediumseagreen" "olivedrab"
    # Plot the z component of the Berry curvature as a background color
    bz_plot = ax.imshow(bz[:, :, kz_cut],
                        extent=[-pi, pi, -pi, pi], cmap=cmap_bz)

    # Show colorbars on the figure
    cbar = fig.colorbar(bxy_plot.lines, pad=fig_params['colorbar1pad'])
    cbar.ax.set_xlabel('    $(F_x^2+F_y^2)^{1/2}$', size=fig_params['fontsize'],
                       rotation=0)
    cbar.ax.tick_params(labelsize=fig_params['fonttickssize'],
                        width=fig_params['linewidth'])
    cbar.outline.set_linewidth(fig_params['linewidth'])
    cbar2 = fig.colorbar(bz_plot, pad=fig_params['colorbar2pad'])
    # Set limits to z component colorbar to make all plots having similar color
    # map
    cbar2.set_clim(-2, 2)
    cbar2.ax.set_xlabel('$F_z$', size=fig_params['fontsize'])
    cbar2.ax.tick_params(labelsize=fig_params['fonttickssize'],
                         width=fig_params['linewidth'])
    cbar2.outline.set_linewidth(fig_params['linewidth'])
    for axis in ['top', 'bottom', 'left', 'right']:
        cbar.ax.spines[axis].set_linewidth(fig_params['linewidth'])
    plt.show()


def chern(b, area_ratio):
    """Calculate the Chern number of the berry curvature b
    The area is defined by b, while area_ration defines the ration
    to the full cut of the BZ"""
    n1, n2 = b.shape
    return -np.sum(b) / n1 / n2 * 2 * pi * area_ratio


def main():
    # Parameters for the Berry curvature plotting
    fig_params = {'figsize': (10, 7), 'axessize': [0.05, 0.08, 0.93, 0.88],
                  'fontsize': 25, 'fonttickssize': 20, 'linewidth': 2,
                  'xlabel_x': 0.9, 'xlabel_y': -0.02, 
                  'ylabel_x': -0.05, 'ylabel_y': 0.85,
                  'streamplot_step': 8, 'stream_density': 0.6,
                  'streamlinewidth': 3, 'streamarrowsize': 3,
                  'colorbar1pad': 0.05, 'colorbar2pad': 0.05}

    # Grid size of the BZ
    nx = 201
    ny = 201
    nz = 201

    # Create the grid and the model
    model_params = {'m': 3}
    kx, ky, kz = hopfham.mesh_make(nx, ny, nz)
    hamilt = hopfham.ham(kx, ky, kz, hopfham.model_mrw, **model_params)
    # Find eigenenergies and eigenstates
    e, u = np.linalg.eigh(hamilt)

    # Calculate 3 components of the Berry curvature on the grid for the
    # occupied eigenstate
    bx = berrycurv_x(u[:, :, :, :, 0])
    by = berrycurv_y(u[:, :, :, :, 0])
    bz = berrycurv_z(u[:, :, :, :, 0])

    # Define the middle cut of the BZ
    kx_half = int((nx - 1)/2)
    ky_half = int((ny - 1)/2)

    # Plot Berry curvature on a 2d cut of BZ

    # kx_cut = int((nx - 1)/2) + 50

    # print(bz[kx_cut, kx_cut, kx_cut], bz[kx_cut, kx_cut, kx_cut+50])
    # plot_berrycurv_xy(bx, by, bz, kx_cut, fig_params)

    # Calculate Chern numbers at the half BZ cuts

    # Check that chern number over the BZ is 0
    print(chern(bx[kx_half, :, :], 1))
    print(chern(bx[-1, :, :], 1))

    # ky in [-pi, 0], kz in [-pi,pi], kx = pi
    c1 = chern(bx[-1, :ky_half, :], 0.5)
    print(c1)
    # ky in [0, pi], kz in [-pi, pi], kx = 0
    c2 = chern(bx[kx_half, ky_half:, :], 0.5)
    print(c2)


if __name__ == '__main__':
    main()
