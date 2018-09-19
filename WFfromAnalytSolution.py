# -*- coding: utf-8 -*-
"""
Created on Wed Sep  5 18:36:26 2018

@author: aleksandra
"""

# Construct Wannier Functions from initial eigenstate of Hopf Hamiltonian

import numpy as np
from math import pi
import matplotlib.pyplot as plt
import pickle
from mayavi import mlab

t=1
h=0

kx = np.linspace(0, 2*pi, 51)
ky = np.linspace(0, 2*pi, 51)
kz = np.linspace(0, 2*pi, 51)

# Cartesian coordinates, indexing for correct order x,y,z (not y,x,z)
[kkx, kky, kkz] = np.meshgrid(kx, ky, kz, indexing = 'ij')

# Set the size of WF matrices - number of points (x, y, z)
# The 3d function of (x, y, z)
WF1 = np.empty((21, 21, 21), dtype = complex)
WF2 = np.empty((21, 21, 21), dtype = complex)
# The function of only one variable, e.g. (x)
WFx1 = np.empty(200, dtype = complex)
WFx2 = np.empty(200, dtype = complex)

# For (x, y, z) in 4 BZs calculate Wannier functions 
# as a Fourier transform of analytical Bloch solutions
for nx in range(21):
    x = (nx-10)/5
    print('nx=', nx)
    for ny in range(21):
        y = (ny-10)/5
        print('ny=', ny)
        for nz in range(21):
            z = (nz-10)/5
            # The dependence on (x, y, z) from exp(ikr)
            u1 = np.multiply(np.divide(np.sin(kkx) + 1j * t * np.sin(kky), np.sqrt(
                 np.power(np.sin(kkx), 2) + t**2 * np.power(np.sin(kky), 2) + 
                 np.power(np.sin(kkz), 2) + 
                 np.power(np.cos(kkx) + np.cos(kky) + np.cos(kkz) + h, 2))),
                 np.exp(1j * (kkx * x + kky * y + kkz * z)))

            u2 = np.multiply(np.divide(np.sin(kkz) + 
                           1j * (np.cos(kkx) + np.cos(kky) + np.cos(kkz) + h),
                 np.sqrt(
                 np.power(np.sin(kkx), 2) + t**2 * np.power(np.sin(kky), 2) + 
                 np.power(np.sin(kkz), 2) + 
                 np.power(np.cos(kkx) + np.cos(kky) + np.cos(kkz) + h, 2))),
                 np.exp(1j * (kkx * x + kky * y + kkz * z)))

            uFourier = np.fft.fftn(u1)/51**3
            WF1[nx, ny, nz] = uFourier[0, 0, 0] # Consider only WF(R=0)
            uFourier = np.fft.fftn(u2)/51**3
            WF2[nx, ny, nz] = uFourier[0, 0, 0]

# Write WFs into file        
with open('WannierLoc.pickle', 'wb') as f:
    pickle.dump([WF1, WF2], f)

# For y = z = 0 and dependency on x calculate WFs as a Fourier transform of 
# Bloch solutions    
for nx in range(200):
    x = nx/10
    print('nx=', nx)
    y = 0
    z = 0
    u1 = np.multiply(np.divide(np.sin(kkx) + 1j * t * np.sin(kky), np.sqrt(
                 np.power(np.sin(kkx), 2) + t**2 * np.power(np.sin(kky), 2) + 
                 np.power(np.sin(kkz), 2) + 
                 np.power(np.cos(kkx) + np.cos(kky) + np.cos(kkz) + h, 2))),
                 np.exp(1j * (kkx * x + kky * y + kkz * z)))

    u2 = np.multiply(np.divide(np.sin(kkz) + 
                           1j * (np.cos(kkx) + np.cos(kky) + np.cos(kkz) + h),
                 np.sqrt(
                 np.power(np.sin(kkx), 2) + t**2 * np.power(np.sin(kky), 2) + 
                 np.power(np.sin(kkz), 2) + 
                 np.power(np.cos(kkx) + np.cos(kky) + np.cos(kkz) + h, 2))),
                 np.exp(1j * (kkx * x + kky * y + kkz * z)))

    uFourier = np.fft.fftn(u1)/51**3
    WFx1[nx] = uFourier[0, 0, 0] # Consider only WF(R=0)
    uFourier = np.fft.fftn(u2)/51**3
    WFx2[nx] = uFourier[0, 0, 0]
        
with open('WannierLocx.pickle', 'wb') as f:
    pickle.dump([WFx1, WFx2], f)

