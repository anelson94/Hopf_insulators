"""
 Created by alexandra at 18.01.19 11:08

 Construct the vector field of Berry curvature around two Weyl points
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


Nx = 200
Ny = 201
Nz = 200

kx = np.linspace(0, 2 * pi, Nx)
ky = np.linspace(0, 2 * pi, Ny)
kz = np.linspace(0, 2 * pi, Nz)

# Import eigenstates of Hopf Humiltonian
with open('HopfeigenWeyl.pickle', 'rb') as f:
    [E, u] = pickle.load(f)

# Occupied states correspond to smaller eigenvalues
uOcc = u[:, :, :, :, 0]
TrFx = np.zeros((Nx - 1, Nz - 1))
TrFz = np.zeros((Nx - 1, Nz - 1))

# Calculate Berry flux in x and z direction as a function of kx, kz
# Use numerical calculation method
nky = 100  # ky = pi
for nkx in range(0, Nx - 1):
    kkx = kx[nkx]
    for nkz in range(0, Nz - 1):
        kky = ky[nky]
        U1x = np.dot(np.conj(uOcc[nkx, nky - 1, nkz, :]),
                     uOcc[nkx, nky + 1, nkz, :])
        U2x = np.dot(np.conj(uOcc[nkx, nky + 1, nkz, :]),
                     uOcc[nkx, nky + 1, nkz + 1, :])
        U3x = np.dot(np.conj(uOcc[nkx, nky + 1, nkz + 1, :]),
                     uOcc[nkx, nky - 1, nkz + 1, :])
        U4x = np.dot(np.conj(uOcc[nkx, nky - 1, nkz + 1, :]),
                     uOcc[nkx, nky - 1, nkz, :])
        TrFx[nkx, nkz] = - ((cmath.log(U1x * U2x * U3x * U4x)).imag
                            * Ny / 2 * Nz / (2 * pi) ** 2)

        U1z = np.dot(np.conj(uOcc[nkx, nky - 1, nkz, :]),
                     uOcc[nkx + 1, nky - 1, nkz, :])
        U2z = np.dot(np.conj(uOcc[nkx + 1, nky - 1, nkz, :]),
                     uOcc[nkx + 1, nky + 1, nkz, :])
        U3z = np.dot(np.conj(uOcc[nkx + 1, nky + 1, nkz, :]),
                     uOcc[nkx, nky + 1, nkz, :])
        U4z = np.dot(np.conj(uOcc[nkx, nky + 1, nkz, :]),
                     uOcc[nkx, nky - 1, nkz, :])
        TrFz[nkx, nkz] = - ((cmath.log(U1z * U2z * U3z * U4z)).imag
                            * Nx * Ny / 2 / (2 * pi) ** 2)


# Plot the vector field
kxgrid, kzgrid = np.meshgrid(kx[0:Nx - 1], kz[0:Nz - 1], indexing='ij')
F, Fxdir, Fzdir = dirandnorm(TrFx, TrFz)
Fabs = np.abs(F)
fig, ax = plt.subplots()
ax.set_title('Berry curvature in $(xz)$ plane, A=1, h=3')
ax.set_xlabel('$k_x$')
ax.set_ylabel('$k_z$')
ax.set_xticks([0, pi, 2 * pi])
ax.set_xticklabels(('0', '$\pi$', '$2\pi$'))
ax.set_yticks([0, pi, 2 * pi])
ax.set_yticklabels(('0', '$\pi$', '$2\pi$'))
plt.xlim(0, 2 * pi)
plt.ylim(0, 2 * pi)
Q = ax.quiver(kxgrid[::8, ::8], kzgrid[::8, ::8],
              Fxdir[::8, ::8],
              Fzdir[::8, ::8],
              F[::8, ::8], pivot='mid', cmap='copper', units='x',
              norm=colors.LogNorm(vmin=F.min(), vmax=F.max()),
              width=0.03, headwidth=3., headlength=4., scale=5.)

cbar = fig.colorbar(Q, extend='max')
cbar.set_label('Norm of the vectors', rotation=270)
cbar.ax.tick_params(labelsize=7)
plt.show()
