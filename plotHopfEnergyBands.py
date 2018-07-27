# -*- coding: utf-8 -*-
"""
Created on Wed Jun 27 13:57:24 2018

@author: aleksandra
"""
# Plot energyy bands for Hopf Insulator

import numpy as np
from math import pi
import math
import pickle
import matplotlib.pyplot as plt
#import math

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

# Import eigenstates of Hopf Humiltonian
with open('Hopfeigen.pickle', 'rb') as f:
    [E, u] = pickle.load(f)

#print(E.shape)    
#figy = plt.figure()
#plt.plot(kx, E[:, int((Ny-1)/2), int((Nz-1)/2), 0], kx, E[:, int((Ny-1)/2), int((Nz-1)/2), 1])
#plt.show

figy = plt.figure()
plt.plot(kx, E[:, 50, 0, 0], kx, E[:, 50, 0, 1])
plt.show
print(E[:, 50, 25, 0])