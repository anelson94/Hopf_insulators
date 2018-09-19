# -*- coding: utf-8 -*-
"""
Created on Tue Sep  4 16:14:58 2018

@author: aleksandra
"""
# Project occupied states on the localized orbitals

import numpy as np
import pickle
from math import pi
import matplotlib.pyplot as plt

import params

t = params.t
h = params.h

Nx = params.Nx
Ny = params.Ny
Nz = params.Nz

kx = np.linspace(0, 2*pi, Nx)
ky = np.linspace(0, 2*pi, Ny)
kz = np.linspace(0, 2*pi, Nz)

with open('Hopfsmoothstates.pickle', 'rb') as f:
    usmooth = pickle.load(f)

# The projection on a state (a, b) with a**2 + b**2 = 1  
# Check that for different combinations of a, b 
# there are points with zero projection -> 
# conventional localized orbitals calculation method is not valid anymore
plt.figure()    
plt.imshow(np.abs(usmooth[:, :, 0, 0] + usmooth[:, :, 0, 1]), cmap='RdBu')
plt.colorbar()
plt.show()