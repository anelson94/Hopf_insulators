# -*- coding: utf-8 -*-
"""
Created on Tue Jun 26 10:08:57 2018

@author: aleksandra
"""

# Calculate Wannier functions for each ky, kz separately

from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import numpy as np
from math import pi
import pickle
import math
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

# Make parallel transport along kx, then calculate Hybrid Wannier Centers for 
# each ky, kz separately    
xAveragek = np.empty([Ny, Nz])
for nky in range(0, Ny):
    for nkz in range(0, Nz):
        uOcc = uk[:, nky, nkz, :, 0]
        usmooth = np.empty([Nx, 2], dtype = complex)
        usmooth[0, :] = uOcc[0, :]
        Mprod = 1
        for nkx in range(0, Nx - 1):
            Mold = np.dot(np.conj(usmooth[nkx, :]), uOcc[nkx + 1, :])
            usmooth[nkx + 1, :] = uOcc[nkx + 1, :] * cmath.exp(-1j * np.angle(Mold))
            Mprod = Mprod * abs(Mold)
        Lamb = np.dot(np.conj(usmooth[0, :]), usmooth[-1, :])
        xAveragek[nky, nkz] = -1/Nx*np.angle(Mprod/Lamb)

with open('HybridWannierCenters.pickle', 'rb') as f:
    xAverage= pickle.load(f)
    
print(xAverage[1,:]-xAveragek[1,:])
        
print(xAveragek[0,:]-xAveragek[-1,:])
print(xAveragek[:,0]-xAveragek[:,-1])
        
ky = np.linspace(0, 2*pi, Ny)
kz = np.linspace(0, 2*pi, Nz)

# Cartesian coordinates
[kky, kkz] = np.meshgrid(ky, kz, indexing = 'ij')

figy = plt.figure()
plt.plot(ky,xAveragek[:,10])
plt.show

figz = plt.figure()
plt.plot(kz,xAveragek[10,:])
plt.show
#ax = plt.axes(projection='3d')
#ax.plot_surface(kky, kkz, xAveragek,rstride=1, cstride=1,
#                cmap='viridis', edgecolor='none')
#ax.view_init(0,90) 
   