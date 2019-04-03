"""
 Created by alexandra at 15.02.19 12:28

 Calculate the Berry flux through the half-plane of the BZ cut along kz axis
 We expect to have a Berry curvature tube around the origin
"""

import numpy as np
from math import pi, sqrt
import pickle
import cmath


def berrycurvy(nx, ny, nz):
    """Calculate y component of Berry curvature at point nx, ny, nz"""
    u1y = np.dot(np.conj(uOcc[nx, ny, nz, :]),
                 uOcc[nx, ny, nz + 1, :])
    u2y = np.dot(np.conj(uOcc[nx, ny, nz + 1, :]),
                 uOcc[nx + 1, ny, nz + 1, :])
    u3y = np.dot(np.conj(uOcc[nx + 1, ny, nz + 1, :]),
                 uOcc[nx + 1, ny, nz, :])
    u4y = np.dot(np.conj(uOcc[nx + 1, ny, nz, :]),
                 uOcc[nx, ny, nz, :])

    return (cmath.log(u1y * u2y * u3y * u4y)).imag * Nx * Nz / (2 * pi) ** 2


def berrycurvx(nx, ny, nz):
    """Calculate x component of Berry curvature at point nx, ny, nz"""
    u1x = np.dot(np.conj(uOcc[nx, ny, nz, :]),
                 uOcc[nx, ny + 1, nz, :])
    u2x = np.dot(np.conj(uOcc[nx, ny + 1, nz, :]),
                 uOcc[nx, ny + 1, nz + 1, :])
    u3x = np.dot(np.conj(uOcc[nx, ny + 1, nz + 1, :]),
                 uOcc[nx, ny, nz + 1, :])
    u4x = np.dot(np.conj(uOcc[nx, ny, nz + 1, :]),
                 uOcc[nx, ny, nz, :])

    return (cmath.log(u1x * u2x * u3x * u4x)).imag * Ny * Nz / (2 * pi) ** 2


Nx = 601
Ny = 601
Nz = 51

kx = np.linspace(0, 2 * pi, Nx)
ky = np.linspace(0, 2 * pi, Ny)
kz = np.linspace(0, 2 * pi, Nz)

# Import eigenstates of Hopf Humiltonian
with open('Hopfeigen.pickle', 'rb') as f:
    [E, u] = pickle.load(f)

# Occupied states correspond to smaller eigenvalues
uOcc = u[:, :, :, :, 0]

# TrFyFlux = 0
# nky = 0
# for nkx in range(0, int((Nx - 1) / 2)):
#     for nkz in range(0, Nz - 1):
#         TrFyFlux = TrFyFlux - berrycurvy(nkx, nky, nkz) / (Nx * Nz) * 2 * pi

TrFxyFlux = 0
for nky in range(0, int((Ny - 1) / 2)):
    nkx = nky
    for nkz in range(0, Nz - 1):
        TrFxyFlux = TrFxyFlux - (
            berrycurvy(nkx, nky, nkz) / (Ny * Nz) * 2 * pi
            - berrycurvx(nkx, nky, nkz) / (Nx * Nz) * 2 * pi) / sqrt(2)

print(TrFxyFlux)
