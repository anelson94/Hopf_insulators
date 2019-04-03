"""
 Created by alexandra at 12.02.19 15:20

 Calculate the polarization in z direction as a function of (kx, ky)
 using the projectors method
"""

import numpy as np
from math import pi
import matplotlib.pyplot as plt
import pickle


def projector(u):
    """Construct a projector from a wavefunction (array in kx, ky directions)"""
    proj = np.empty((Nx, Ny, 2, 2), dtype=complex)
    proj[:, :, 0, 0] = np.conj(u[:, :, 0]) * u[:, :, 0]
    proj[:, :, 0, 1] = np.conj(u[:, :, 0]) * u[:, :, 1]
    proj[:, :, 1, 0] = np.conj(u[:, :, 1]) * u[:, :, 0]
    proj[:, :, 1, 1] = np.conj(u[:, :, 1]) * u[:, :, 1]
    return proj


def projector_x(u):
    """Construct a projector from a wavefunction (array in ky, kz directions)"""
    proj = np.empty((Ny, Nz, 2, 2), dtype=complex)
    proj[:, :, 0, 0] = np.conj(u[:, :, 0]) * u[:, :, 0]
    proj[:, :, 0, 1] = np.conj(u[:, :, 0]) * u[:, :, 1]
    proj[:, :, 1, 0] = np.conj(u[:, :, 1]) * u[:, :, 0]
    proj[:, :, 1, 1] = np.conj(u[:, :, 1]) * u[:, :, 1]
    return proj


def projector_y(u):
    """Construct a projector from a wavefunction (array in kx, kz directions)"""
    proj = np.empty((Nx, Nz, 2, 2), dtype=complex)
    proj[:, :, 0, 0] = np.conj(u[:, :, 0]) * u[:, :, 0]
    proj[:, :, 0, 1] = np.conj(u[:, :, 0]) * u[:, :, 1]
    proj[:, :, 1, 0] = np.conj(u[:, :, 1]) * u[:, :, 0]
    proj[:, :, 1, 1] = np.conj(u[:, :, 1]) * u[:, :, 1]
    return proj


# Import eigenstates of Hopf Humiltonian
# with open('Hopfgeneigen.pickle', 'rb') as f:
with open('HopfeigenWeyl.pickle', 'rb') as f:
    [E, u] = pickle.load(f)
uocc = u[:, :, :, :, 0]

Nx = 100
Ny = 100
Nz = 1000

Proj_all = np.identity(2, dtype=complex)
Proj_all = Proj_all[np.newaxis, np.newaxis, :, :]
for idz in range(Nz):
    Proj_all = np.matmul(Proj_all, projector(uocc[:, :, idz, :]))
# for idy in range(Ny):
#     Proj_all = np.matmul(Proj_all, projector_y(uocc[:, idy, :, :]))

# print(Proj_all)
[eig, func] = np.linalg.eig(Proj_all)
# P001_1 = np.log(eig[0])
P001_2 = -np.angle(eig[:, :, 1] + eig[:, :, 0]) / 2 / pi
P001_2 = np.where(P001_2 < 0, P001_2, P001_2 - 1)

print(np.sum(P001_2) / Nx / Nz)
plt.figure()
plt.imshow(P001_2)
# plt.plot(Kz[0: Nk - 1], np.real((psi1[:Nk - 1] - psi1[1:]))
#          * np.imag(psi1[:Nk - 1]))
plt.colorbar()
plt.show()
