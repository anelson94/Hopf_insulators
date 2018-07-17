# -*- coding: utf-8 -*-
"""
Created on Tue Jul 17 11:41:42 2018

@author: aleksandra
"""

# Calculate Hopf invariant using analytical eigenstates of H from Aris note

import numpy as np
from math import pi
import matplotlib.pyplot as plt

# Import parameters for Hopf Hamiltonian from file params.py
import params

t = params.t
h = params.h

Nx = params.Nx
Ny = params.Ny
Nz = params.Nz 

kx = np.linspace(0, 2*pi, Nx)
ky = np.linspace(0, 2*pi, Ny)
kz = np.linspace(0, 2*pi, Nz)

# Cartesian coordinates, indexing for correct order x,y,z (not y,x,z)
[kkx, kky, kkz] = np.meshgrid(kx, ky, kz, indexing = 'ij')

# z1 and z2 functions for eigenstate construction
lamb = np.divide(1, np.power(np.sin(kkx), 2) + t**2 * np.power(np.sin(kky), 2) + 
                 np.power(np.sin(kkz), 2) + 
                 np.power(np.cos(kkx) + np.cos(kky) + np.cos(kkz) + h, 2))

# Analytic expression for Hopf integrand AF
HopffromMathematica = np.multiply(np.power(lamb, 2), 
                                  2 * t * (np.multiply(np.cos(kky), np.cos(kkz)) + 
                                  np.multiply(np.cos(kkx), np.cos(kky)) + 
                                  np.multiply(np.cos(kkx), np.cos(kkz)) +
                                  h * np.multiply(np.multiply(np.cos(kkx), 
                                                              np.cos(kky)), 
                                                  np.cos(kkz))))

# After summation we get Hopf invariant
Hopf = (np.sum(HopffromMathematica[0:Nx-1, 0:Ny-1, 0:Nz-1]) * 
        (2*pi)/((Nx-1) * (Ny-1) * (Nz-1)))

# Plot integrant at different kz
plt.imshow(HopffromMathematica[:, :, 70], cmap='RdBu')
plt.colorbar()
plt.show()

print(Hopf)

# Numerical calculation
# !!! Something is wrong

#z1 = np.multiply(np.sqrt(lamb), (np.sin(kkx) + 1j * t * np.sin(kky)))
#z2 = np.multiply(np.sqrt(lamb), (np.sin(kkz) + 
#             1j * (np.cos(kkx) + np.cos(kky) + np.cos(kkz) + h)))
# 
#z1x = np.zeros((Nx, Ny, Nz), dtype = complex)
#z1y = np.zeros((Nx, Ny, Nz), dtype = complex)
#z1z = np.zeros((Nx, Ny, Nz), dtype = complex)
#z2x = np.zeros((Nx, Ny, Nz), dtype = complex)
#z2y = np.zeros((Nx, Ny, Nz), dtype = complex)
#z2z = np.zeros((Nx, Ny, Nz), dtype = complex)
#
#z1x[0:Nx-1, 0:Ny-1, 0:Nz-1] = z1[1:Nx, 0:Ny-1, 0:Nz-1] - z1[0:Nx-1, 0:Ny-1, 0:Nz-1]
#z1y[0:Nx-1, 0:Ny-1, 0:Nz-1] = z1[0:Nx-1, 1:Ny, 0:Nz-1] - z1[0:Nx-1, 0:Ny-1, 0:Nz-1]
#z1z[0:Nx-1, 0:Ny-1, 0:Nz-1] = z1[0:Nx-1, 0:Ny-1, 1:Nz] - z1[0:Nx-1, 0:Ny-1, 0:Nz-1]
#
#z2x[0:Nx-1, 0:Ny-1, 0:Nz-1] = z2[1:Nx, 0:Ny-1, 0:Nz-1] - z2[0:Nx-1, 0:Ny-1, 0:Nz-1]
#z2y[0:Nx-1, 0:Ny-1, 0:Nz-1] = z2[0:Nx-1, 1:Ny, 0:Nz-1] - z2[0:Nx-1, 0:Ny-1, 0:Nz-1]
#z2z[0:Nx-1, 0:Ny-1, 0:Nz-1] = z2[0:Nx-1, 0:Ny-1, 1:Nz] - z2[0:Nx-1, 0:Ny-1, 0:Nz-1]
#
#z1x[Nx-1, :, :] = z1x[0, :, :]
#z1x[:, Ny-1, :] = z1x[:, 0, :]
#z1x[:, :, Nz-1] = z1x[:, :, 0]
#
#z1y[Nx-1, :, :] = z1y[0, :, :]
#z1y[:, Ny-1, :] = z1y[:, 0, :]
#z1y[:, :, Nz-1] = z1y[:, :, 0]
#
#z1z[Nx-1, :, :] = z1z[0, :, :]
#z1z[:, Ny-1, :] = z1z[:, 0, :]
#z1z[:, :, Nz-1] = z1z[:, :, 0]
#
#z2x[Nx-1, :, :] = z2x[0, :, :]
#z2x[:, Ny-1, :] = z2x[:, 0, :]
#z2x[:, :, Nz-1] = z2x[:, :, 0]
#
#z2y[Nx-1, :, :] = z2y[0, :, :]
#z2y[:, Ny-1, :] = z2y[:, 0, :]
#z2y[:, :, Nz-1] = z2y[:, :, 0]
#
#z2z[Nx-1, :, :] = z2z[0, :, :]
#z2z[:, Ny-1, :] = z2z[:, 0, :]
#z2z[:, :, Nz-1] = z2z[:, :, 0]
##z1 = z1[0:Nx-1, 0:Ny-1, 0:Nz-1]
##z2 = z2[0:Nx-1, 0:Ny-1, 0:Nz-1]
#
#Ax = 1j * (np.multiply(z1, np.conj(z1x)) + np.multiply(z2, np.conj(z2x)))
#Ay = 1j * (np.multiply(z1, np.conj(z1y)) + np.multiply(z2, np.conj(z2y)))
#Az = 1j * (np.multiply(z1, np.conj(z1z)) + np.multiply(z2, np.conj(z2z)))
#
#Axy = Ax[0:Nx-1, 1:Ny, 0:Nz-1] - Ax[0:Nx-1, 0:Ny-1, 0:Nz-1]
#Axz = Ax[0:Nx-1, 0:Ny-1, 1:Nz] - Ax[0:Nx-1, 0:Ny-1, 0:Nz-1]
#
#Ayx = Ay[1:Nx, 0:Ny-1, 0:Nz-1] - Ay[0:Nx-1, 0:Ny-1, 0:Nz-1]
#Ayz = Ay[0:Nx-1, 0:Ny-1, 1:Nz] - Ay[0:Nx-1, 0:Ny-1, 0:Nz-1]
#
#Azx = Az[1:Nx, 0:Ny-1, 0:Nz-1] - Az[0:Nx-1, 0:Ny-1, 0:Nz-1]
#Azy = Az[0:Nx-1, 1:Ny, 0:Nz-1] - Az[0:Nx-1, 0:Ny-1, 0:Nz-1]
#
#Ax = Ax[0:Nx-1, 0:Ny-1, 0:Nz-1]
#Ay = Ay[0:Nx-1, 0:Ny-1, 0:Nz-1]
#Az = Az[0:Nx-1, 0:Ny-1, 0:Nz-1]
#
#Fxrot = Azy - Ayz
#Fyrot = Axz - Azx
#Fzrot = Ayz - Axy
#
#
#Fx = -2 * np.imag(np.multiply(z2y, np.conj(z2z)) + 
#                  np.multiply(z1y, np.conj(z1z)))
#Fy = -2 * np.imag(np.multiply(z2z, np.conj(z2x)) + 
#                  np.multiply(z1z, np.conj(z1x)))
#Fz = -2 * np.imag(np.multiply(z2x, np.conj(z2y)) + 
#                  np.multiply(z1x, np.conj(z1y)))
#
#print(np.max(np.abs((Fxrot - Fx[0:Nx-1, 0:Ny-1, 0:Nz-1]))))
#
#Hopfintegrand = (- np.multiply(Ax, Fxrot) - np.multiply(Ay, Fyrot) - 
#                 np.multiply(Az, Fzrot))
