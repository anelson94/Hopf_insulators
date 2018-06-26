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
with open('Hopfeigeneachk.pickle', 'rb') as f:
    [Ek, uk] = pickle.load(f)
    
xAverage = np.empty([Ny, Nz])
for nky in range(0, Ny):
    for nkz in range(0, Nz):
        uOcc = uk[:, nky, nkz, :, 1]
        usmooth = np.empty([Nx, 2], dtype = complex)
        usmooth[0, :] = uOcc[0, :]
        Mprod = 1
        for nkx in range(0, Nx - 1):
            Mold = np.dot(np.conj(usmooth[nkx, :]), uOcc[nkx + 1, :])
            usmooth[nkx + 1, :] = uOcc[nkx + 1, :] * cmath.exp(-1j * np.angle(Mold))
            Mprod = Mprod * abs(Mold)
        Lamb = np.dot(np.conj(usmooth[0, :]), usmooth[-1, :])
        xAverage[nky, nkz] = -1/Nx*np.angle(Mprod/Lamb)
        
ky = np.linspace(0, 2*pi, Ny)
kz = np.linspace(0, 2*pi, Nz)

# Cartesian coordinates
[kky, kkz] = np.meshgrid(ky, kz)

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot_surface(kky, kkz, xAverage,rstride=1, cstride=1,
                cmap='viridis', edgecolor='none')
#ax.view_init(0,90) 
   