# -*- coding: utf-8 -*-
"""
Created on Tue Jun 26 10:08:57 2018

@author: aleksandra
"""

# Calculate Wannier functions for each kx, ky separately

# from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
# from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from math import pi
import pickle
# import math
import cmath

# Import parameters for Hopf Hamiltonian from file params.py
import params

t = params.t
h = params.h

Nx = params.Nx
Ny = params.Ny
Nz = params.Nz

# Import eigenstates of Hopf Humiltonian
with open('Hopfeigen.pickle', 'rb') as f:
    [Ek, uk] = pickle.load(f)

# Make parallel transport along kz, then calculate Hybrid Wannier Centers for
# each kx, ky separately
zAveragek = np.empty([Nx, Ny])
for nkx in range(0, Nx):
    for nky in range(0, Ny):
        uOcc = uk[nkx, nky, :, :, 0]
        usmooth = np.empty([Nz, 2], dtype=complex)
        usmooth[0, :] = uOcc[0, :]
        Mprod = 1
        for nkz in range(0, Nz - 1):
            Mold = np.dot(np.conj(usmooth[nkz, :]), uOcc[nkz + 1, :])
            usmooth[nkz + 1, :] = (uOcc[nkz + 1, :]
                                   * cmath.exp(-1j * np.angle(Mold)))
            Mprod = Mprod * abs(Mold)
        Lamb = np.dot(np.conj(usmooth[0, :]), usmooth[-1, :])
        zAveragek[nkx, nky] = -1/Nz * np.angle(Mprod/Lamb)

# with open('HybridWannierCenters.pickle', 'rb') as f:
#     zAverage = pickle.load(f)
#
# print(zAverage[1, :]-zAveragek[1, :])
#
# print(zAveragek[0, :]-zAveragek[-1, :])
# print(zAveragek[:, 0]-zAveragek[:, -1])
        
kx = np.linspace(0, 2*pi, Nx)
ky = np.linspace(0, 2*pi, Ny)

# Cartesian coordinates
[kkx, kky] = np.meshgrid(kx, ky, indexing='ij')

# figy = plt.figure()
# plt.plot(ky,zAveragek[:,50])
# plt.show()
#
# figz = plt.figure()
# plt.plot(kz,zAveragek[50,:])
# plt.show()
fig = plt.figure()
plt.imshow(zAverage, cmap=cm.coolwarm)
plt.colorbar()
# ax = fig.add_subplot(111, projection='3d')
# Axes3D.plot_surface(ax, kky, kkz, zAveragek, rstride=1, cstride=1,
#                     cmap=cm.coolwarm)
plt.show()
