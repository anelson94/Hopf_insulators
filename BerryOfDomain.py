# -*- coding: utf-8 -*-
"""
Created on Mon Aug  6 12:20:36 2018

@author: aleksandra
"""

# Calculate Berry phase of different domains in BZ of Hopf insulator
# These domaines are bounded by Fermi arcs at E=0

import numpy as np
from math import pi
import pickle
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

# Import smooth eigenstates of Hopf Humiltonian
with open('Hopfsmoothstates.pickle', 'rb') as f:
    usmooth = pickle.load(f)

# For each point (kx, ky) calculate Berry phase along kz as 1d system
Berry = np.empty((Nx, Ny))    
for nx in range(0, Nx):
    for ny in range(0, Ny):
        # numerical expression for Berry phase as a sum of Im(log(A))
        Az = np.sum(np.multiply(np.conj(usmooth[nx, ny, 0:Nz-1, :]), 
                          usmooth[nx, ny, 1:Nz, :]), axis = -1)
        Berry[nx, ny] = -np.sum((np.log(Az)).imag)
        
# We expect quantization of the Berry phase in different domains

plt.figure()
plt.imshow(Berry)
plt.colorbar()

# There is no quantization