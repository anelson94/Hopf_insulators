# -*- coding: utf-8 -*-
"""
Created on Mon Jun 25 18:20:25 2018

@author: aleksandra
"""

from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt

import numpy as np
import pickle
from math import pi

import params

t = params.t
h = params.h

Nx = params.Nx
Ny = params.Ny
Nz = params.Nz

ky = np.linspace(0, 2*pi, Ny)
kz = np.linspace(0, 2*pi, Nz)

# Cartesian coordinates
[kky, kkz] = np.meshgrid(ky, kz, indexing = 'ij')

with open('HybridWannierCenters.pickle', 'rb') as f:
    xAverage= pickle.load(f)
    
figy = plt.figure()
plt.plot(ky,xAverage[:,9])
plt.show

figz = plt.figure()
plt.plot(kz,xAverage[9,:])
plt.show
    
#fig = plt.figure()
#ax = plt.axes(projection='3d')
#ax.plot_surface(kky, kkz, xAverage,rstride=1, cstride=1,
#                cmap='viridis', edgecolor='none')
#ax.view_init(0,90) 
   
