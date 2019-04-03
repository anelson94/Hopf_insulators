"""
 Created by alexandra at 18.01.19 11:08

 Construct the vector field of Berry curvature on a cut of the BZ by xy plane
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
TrFx = np.zeros((Nx - 1, Ny - 1))
TrFy = np.zeros((Nx - 1, Ny - 1))
TrFz = np.zeros((Nx - 1, Ny - 1))

# Calculate 3 components of Berry flux as a function of kx, ky at kz=kz(nkz)
# Use numerical calculation method
nkz = 0
for nkx in range(0, Nx - 1):
    kkx = kx[nkx]
    for nky in range(0, Ny - 1):
        kky = ky[nky]
        U1x = np.dot(np.conj(uOcc[nkx, nky, nkz, :]),
                     uOcc[nkx, nky + 1, nkz, :])
        U2x = np.dot(np.conj(uOcc[nkx, nky + 1, nkz, :]),
                     uOcc[nkx, nky + 1, nkz + 1, :])
        U3x = np.dot(np.conj(uOcc[nkx, nky + 1, nkz + 1, :]),
                     uOcc[nkx, nky, nkz + 1, :])
        U4x = np.dot(np.conj(uOcc[nkx, nky, nkz + 1, :]),
                     uOcc[nkx, nky, nkz, :])
        TrFx[nkx, nky] = - ((cmath.log(U1x * U2x * U3x * U4x)).imag
                            * Ny * Nz / (2 * pi) ** 2)

        U1z = np.dot(np.conj(uOcc[nkx, nky, nkz, :]),
                     uOcc[nkx + 1, nky, nkz, :])
        U2z = np.dot(np.conj(uOcc[nkx + 1, nky, nkz, :]),
                     uOcc[nkx + 1, nky + 1, nkz, :])
        U3z = np.dot(np.conj(uOcc[nkx + 1, nky + 1, nkz, :]),
                     uOcc[nkx, nky + 1, nkz, :])
        U4z = np.dot(np.conj(uOcc[nkx, nky + 1, nkz, :]),
                     uOcc[nkx, nky, nkz, :])
        TrFz[nkx, nky] = - ((cmath.log(U1z * U2z * U3z * U4z)).imag
                            * Nx * Ny / (2 * pi) ** 2)

        U1y = np.dot(np.conj(uOcc[nkx, nky, nkz, :]),
                     uOcc[nkx, nky, nkz + 1, :])
        U2y = np.dot(np.conj(uOcc[nkx, nky, nkz + 1, :]),
                     uOcc[nkx + 1, nky, nkz + 1, :])
        U3y = np.dot(np.conj(uOcc[nkx + 1, nky, nkz + 1, :]),
                     uOcc[nkx + 1, nky, nkz, :])
        U4y = np.dot(np.conj(uOcc[nkx + 1, nky, nkz, :]),
                     uOcc[nkx, nky, nkz, :])
        TrFy[nkx, nky] = - ((cmath.log(U1y * U2y * U3y * U4y)).imag
                            * Nx * Nz / (2 * pi) ** 2)


# Change [0, 2pi] -> [-pi, pi]
TrFx = np.concatenate((TrFx[int(int(round((Nx - 1) / 2))):Nx - 1, :],
                      TrFx[1:int(round((Nx - 1) / 2)), :]), axis=0)
TrFy = np.concatenate((TrFy[int(round((Nx - 1) / 2)):Nx - 1, :],
                      TrFy[1:int(round((Nx - 1) / 2)), :]), axis=0)
TrFz = np.concatenate((TrFz[int(round((Nx - 1) / 2)):Nx - 1, :],
                      TrFz[1:int(round((Nx - 1) / 2)), :]), axis=0)

TrFx = np.concatenate((TrFx[:, int(round((Ny - 1) / 2)):Ny - 1],
                      TrFx[:, 1:int(round((Ny - 1) / 2))]), axis=1)
TrFy = np.concatenate((TrFy[:, int(round((Ny - 1) / 2)):Ny - 1],
                      TrFy[:, 1:int(round((Ny - 1) / 2))]), axis=1)
TrFz = np.concatenate((TrFz[:, int(round((Ny - 1) / 2)):Ny - 1],
                      TrFz[:, 1:int(round((Ny - 1) / 2))]), axis=1)

# Plot the vector field
kxpipi = np.linspace(-pi, pi, Nx)
kypipi = np.linspace(-pi, pi, Ny)
kxgrid, kygrid = np.meshgrid(kxpipi[0:Nx - 1], kypipi[0:Ny - 1], indexing='ij')
F, Fxdir, Fydir = dirandnorm(TrFx, TrFy)

fig = plt.figure(figsize=(10, 7.2))
ax = fig.add_axes([0.105, 0.10, 0.89, 0.87])
# ax.set_title(' \n Berry curvature in $(xy)$ plane, A=0, h=2\n at $k_z = 0$',
# size=25)
ax.set_xlabel('$k_x$', size=25)
ax.set_ylabel('$k_y$', size=25)
ax.set_xticks([-pi, 0, pi])
ax.set_xticklabels(('$-\pi$', '$0$', '$\pi$'), size=20)
ax.set_yticks([-pi, 0, pi])
ax.set_yticklabels(('$-\pi$', '$0$', '$\pi$'), size=20)
plt.xlim(-pi, pi)
plt.ylim(-pi, pi)
for axis in ['top', 'bottom', 'left', 'right']:
    ax.spines[axis].set_linewidth(2)
ax.tick_params(width=2)
# Q = ax.quiver(kxgrid[::8, ::8], kygrid[::8, ::8],
#               Fxdir[::8, ::8],
#               Fydir[::8, ::8],
#               F[::8, ::8], pivot='mid', cmap='copper', units='x',
#               norm=colors.LogNorm(vmin=F.min(), vmax=F.max()),
#               width=0.03, headwidth=3., headlength=4., scale=6.)
cmapQ = colors.LinearSegmentedColormap.from_list("", ["lightgray", "red"])
Q = ax.streamplot(
    kygrid[::8, ::8], kxgrid[::8, ::8], Fydir[::8, ::8], Fxdir[::8, ::8],
    color=F[::8, ::8], cmap=cmapQ, density=0.6, linewidth=3, arrowsize=3)
cmapFz = colors.LinearSegmentedColormap.from_list(
    "", [(0.2, 0.5, 0.5), "white", "navy"])
Fz = ax.imshow(TrFz, extent=[-pi, pi, -pi, pi], cmap=cmapFz)
# Fz = ax.imshow(TrFz, extent=[0, 2 * pi, 0, 2 * pi], cmap='BrBG')
cbar = fig.colorbar(Q.lines, pad=0.03)
cbar.ax.set_xlabel('    $(\Omega_x^2+\Omega_y^2)^{1/2}$', size=25, rotation=0)
cbar.ax.tick_params(labelsize=20, width=2)
cbar2 = fig.colorbar(Fz, pad=0.03)
cbar2.ax.set_xlabel('$\Omega_z$', size=25)
cbar2.ax.tick_params(labelsize=20, width=2)
for axis in ['top', 'bottom', 'left', 'right']:
    cbar.ax.spines[axis].set_linewidth(2)
plt.show()
