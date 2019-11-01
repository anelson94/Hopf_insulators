"""
 Created by alexandra at 26.04.19 12:37

 Calculate and plot the Berry curvature at critical point h
 with the Weyl splitting constatnt A
"""

import numpy as np
from math import pi
import pickle
import cmath
import matplotlib.pyplot as plt
import matplotlib.colors as colors


def dirandnorm(x, y):
    """Define the directions and module of vector field"""
    r = np.sqrt(np.power(x, 2) + np.power(y, 2))
    return r, x / r, y / r


Nx = 201
Ny = 201
Nz = 201

kx = np.linspace(0, 2 * pi, Nx)
ky = np.linspace(0, 2 * pi, Ny)
kz = np.linspace(0, 2 * pi, Nz)

# Import eigenstates of Hopf Humiltonian
# Have possibility to add splitting term
with open('HopfeigenWeyl.pickle', 'rb') as f:
    [E, u] = pickle.load(f)

# Occupied states correspond to smaller eigenvalues
uOcc = u[:, :, :, :, 0]
TrFx = np.zeros((Nx - 1, Nz - 1))
TrFy = np.zeros((Nx - 1, Nz - 1))
TrFz = np.zeros((Nx - 1, Nz - 1))

# Calculate 3 components of Berry flux as a function of kx, kz at ky=ky(nky)
# Use numerical calculation method
nky = 0
for nkx in range(0, Nx - 1):
    kkx = kx[nkx]
    for nkz in range(0, Nz - 1):
        kkz = kz[nkz]
        U1x = np.dot(np.conj(uOcc[nkx, nky - 1, nkz, :]),
                     uOcc[nkx, nky + 1, nkz, :])
        U2x = np.dot(np.conj(uOcc[nkx, nky + 1, nkz, :]),
                     uOcc[nkx, nky + 1, nkz + 1, :])
        U3x = np.dot(np.conj(uOcc[nkx, nky + 1, nkz + 1, :]),
                     uOcc[nkx, nky - 1, nkz + 1, :])
        U4x = np.dot(np.conj(uOcc[nkx, nky - 1, nkz + 1, :]),
                     uOcc[nkx, nky - 1, nkz, :])
        TrFx[nkx, nkz] = - ((cmath.log(U1x * U2x * U3x * U4x)).imag
                            * Ny * Nz / (2 * pi) ** 2) / 2

        U1z = np.dot(np.conj(uOcc[nkx, nky - 1, nkz, :]),
                     uOcc[nkx + 1, nky - 1, nkz, :])
        U2z = np.dot(np.conj(uOcc[nkx + 1, nky - 1, nkz, :]),
                     uOcc[nkx + 1, nky + 1, nkz, :])
        U3z = np.dot(np.conj(uOcc[nkx + 1, nky + 1, nkz, :]),
                     uOcc[nkx, nky + 1, nkz, :])
        U4z = np.dot(np.conj(uOcc[nkx, nky + 1, nkz, :]),
                     uOcc[nkx, nky - 1, nkz, :])
        TrFz[nkx, nkz] = - ((cmath.log(U1z * U2z * U3z * U4z)).imag
                            * Nx * Ny / (2 * pi) ** 2) / 2

        # U1y = np.dot(np.conj(uOcc[nkx, nky, nkz, :]),
        #              uOcc[nkx, nky, nkz + 1, :])
        # U2y = np.dot(np.conj(uOcc[nkx, nky, nkz + 1, :]),
        #              uOcc[nkx + 1, nky, nkz + 1, :])
        # U3y = np.dot(np.conj(uOcc[nkx + 1, nky, nkz + 1, :]),
        #              uOcc[nkx + 1, nky, nkz, :])
        # U4y = np.dot(np.conj(uOcc[nkx + 1, nky, nkz, :]),
        #              uOcc[nkx, nky, nkz, :])
        # TrFy[nkx, nkz] = - ((cmath.log(U1y * U2y * U3y * U4y)).imag
        #                     * Nx * Nz / (2 * pi) ** 2)


# Change [0, 2pi] -> [-pi, pi]
TrFx = np.concatenate((TrFx[int(int(round((Nx - 1) / 2))):Nx - 1, :],
                      TrFx[1:int(round((Nx - 1) / 2)), :]), axis=0)
# TrFy = np.concatenate((TrFy[int(round((Nx - 1) / 2)):Nx - 1, :],
#                       TrFy[1:int(round((Nx - 1) / 2)), :]), axis=0)
# TrFz = np.concatenate((TrFz[int(round((Nx - 1) / 2)):Nx - 1, :],
#                       TrFzFz[1:int(round((Nx - 1) / 2)), :]), axis=0)

TrFx = np.concatenate((TrFx[:, int(round((Ny - 1) / 2)):Ny - 1],
                      TrFx[:, 1:int(round((Ny - 1) / 2))]), axis=1)
# TrFy = np.concatenate((TrFy[:, int(round((Ny - 1) / 2)):Ny - 1],
#                       TrFy[:, 1:int(round((Ny - 1) / 2))]), axis=1)
# TrFz = np.concatenate((TrFz[:, int(round((Ny - 1) / 2)):Ny - 1],
#                       TrFz[:, 1:int(round((Ny - 1) / 2))]), axis=1)

TrFz = TrFz[1:, 1:]
print(TrFx.shape)
print(TrFz.shape)
# Plot the vector field
kxpipi = np.linspace(-pi, pi, Nx)
kzpipi = np.linspace(0, 2 * pi, Ny)
kxgrid, kzgrid = np.meshgrid(kxpipi[0:Nx - 1], kzpipi[0:Ny - 1], indexing='ij')
F, Fxdir, Fzdir = dirandnorm(TrFx, TrFz)

fig = plt.figure(figsize=(8, 6.3))
ax = fig.add_axes([0.07, 0.09, 0.89, 0.87])
ax.xaxis.set_label_coords(0.9, -0.02)
ax.yaxis.set_label_coords(-0.05, 0.85)
ax.set_ylabel('$k_x$', size=25, rotation=0)
ax.set_xlabel('$k_z$', size=25)
ax.set_xticks([0, pi, 2 * pi])
ax.set_xticklabels(('$-\pi$', '$0$', '$\pi$'), size=20)
ax.set_yticks([-pi, 0, pi])
ax.set_yticklabels(('$-\pi$', '$0$', '$\pi$'), size=20)
plt.xlim(0, 2 * pi)
plt.ylim(-pi, pi)
for axis in ['top', 'bottom', 'left', 'right']:
    ax.spines[axis].set_linewidth(2)
ax.tick_params(width=2)
cmapQ = colors.LinearSegmentedColormap.from_list("", ["lightgray", "red"])
Q = ax.streamplot(
    kzgrid[::8, ::8], kxgrid[::8, ::8], Fzdir[::8, ::8], Fxdir[::8, ::8],
    color=F[::8, ::8], norm=colors.LogNorm(vmin=F.min(), vmax=F.max()),
    cmap=cmapQ, density=0.72, linewidth=3, arrowsize=2.5)
# Q = ax.quiver(kxgrid[::4, ::4], kzgrid[::4, ::4],
#               Fxdir[::4, ::4],
#               Fzdir[::4, ::4],
#               F[::4, ::4], pivot='mid', cmap='copper', units='x',
#               norm=colors.LogNorm(vmin=F.min(), vmax=F.max()),
#               width=0.03, headwidth=3., headlength=4., scale=6.)
cbar = fig.colorbar(Q.lines, pad=0.1)  #
cbar.ax.set_xlabel('    $(F_x^2+F_y^2)^{1/2}$', size=20, rotation=0)
cbar.ax.tick_params(labelsize=15, width=2)
for axis in ['top', 'bottom', 'left', 'right']:
    cbar.ax.spines[axis].set_linewidth(2)
plt.show()
