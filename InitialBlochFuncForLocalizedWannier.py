# -*- coding: utf-8 -*-
"""
Created on Tue Jul  3 16:58:00 2018

@author: aleksandra
"""

# Constract initial Bloch basis by projecting of Gaussian functions 
# with centers in (0,0,0) on the obtained Bloch basis

import numpy as np
from math import pi
import pickle
import math
import cmath
import scipy.linalg
import sys
from memory_profiler import profile

# Import parameters for Hopf Hamiltonian from file params.py
import params

t = params.t
h = params.h

Nx = params.Nx
Ny = params.Ny
Nz = params.Nz


# Import eigenstates of Hopf Humiltonian
with open('Hopfeigen.pickle', 'rb') as f:
    [E, u] = pickle.load(f)

  
kx = np.linspace(0, 2*pi, Nx)
ky = np.linspace(0, 2*pi, Ny)
kz = np.linspace(0, 2*pi, Nz)

# Cartesian coordinates, indexing for correct order x,y,z (not y,x,z)
[kkx, kky, kkz] = np.meshgrid(kx, ky, kz, indexing = 'ij')

# Gaussian function on k mesh
Gauss = np.exp((- np.power(kkx, 2) - np.power(kky, 2) - 
                np.power(kkz, 2))) / (2*pi)

Gauss = Gauss[:, :, :, np.newaxis, np.newaxis]

Gaussvect1 = np.concatenate((Gauss, np.zeros((Nx, Ny, Nz, 1, 1))), axis = -2)
Gaussvect2 = np.concatenate((np.zeros((Nx, Ny, Nz, 1, 1)), Gauss), axis = -2)
Gaussmatr = np.concatenate((Gaussvect1, Gaussvect2), axis = -1)
print(Gaussmatr.shape)
print(sys.getsizeof(Gaussmatr))
print(u.shape)
print(sys.getsizeof(u))
Gauss = None
Gaussvect = None
Gaussvect2 = None

Identity = np.sum(np.transpose(u, (0,1,2,4,3)).reshape(Nx,Ny,Nz,2,2,1) * 
                  np.conj(np.transpose(u, (0,1,2,4,3)).reshape(Nx,Ny,Nz,2,1,2)),-3)

print(np.amin(np.linalg.det(Identity)))
print(np.amax(np.absolute(np.linalg.det(Identity))))
# Project Gaussian function on the obtained wavefunctions

# Overlap with u
uGaussOvlp = np.sum(np.conj(u).reshape(Nx, Ny, Nz, 
                    2, 2, 1) * Gaussmatr.reshape(Nx, Ny, Nz, 2, 1, 2), -3)
#Gaussmatr = None
# Use multiplication method from 
# https://jameshensman.wordpress.com/2010/06/14/multiple-matrix-multiplication-in-numpy/
gOvlp = np.sum(np.conj(Gaussmatr).reshape(Nx, Ny, Nz, 2, 2, 1) 
               * Gaussmatr.reshape(Nx, Ny, Nz, 2, 1, 2), -3)

print(np.amin(np.absolute(np.linalg.det(uOvlp))))
print(np.amax(np.absolute(np.linalg.det(uOvlp))))
# Project on u
unew = np.sum(np.transpose(u,(0,1,2,4,3)).reshape(Nx, Ny, Nz, 
              2, 2, 1) * uGaussOvlp.reshape(Nx, Ny, Nz, 2, 1, 2), -3)

uGaussOvlp = None
#u = None
# Ortonormalize new functions
uOvlp = np.sum(np.conj(u).reshape(Nx, Ny, Nz, 2, 2, 1) 
               * u.reshape(Nx, Ny, Nz, 2, 1, 2), -3)

print(np.amin(np.absolute(np.linalg.det(uOvlp))))
print(np.amax(np.absolute(np.linalg.det(uOvlp))))
#uforWan = np.empty((Nx, Ny, Nz, 2, 2), dtype = complex)
#for nkx in range(0, Nx):
#    for nky in range(0, Ny):
#        for nkz in range(0, Nz):
#            sqrtUOvlp = scipy.linalg.fractional_matrix_power(uOvlp[nkx, nky, nkz, :, :], -1/2)
#            for nBand in [0,1]:
#                uforWan[nkx, nky, nkz, :, nBand] = (
#                        sqrtUOvlp[ 0, nBand]*unew[nkx, nky, nkz, 0, :] 
#                        + sqrtUOvlp[ 1, nBand]*unew[nkx, nky, nkz, 1, :])
#            

 

