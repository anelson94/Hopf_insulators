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

# Load parameters of the system
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

# Load calculated HW Centers
with open('HybridWannierCenters.pickle', 'rb') as f:
    xAverage= pickle.load(f)

# Calculate the difference in HW Centers between two points: ky/z=0 and 2pi
# It should be zero
xdiffky = xAverage[0,:] - xAverage[-1,:]
xdiffkz = xAverage[:,0] - xAverage[:,-1]
# Write this difference in file
file = open('xdiffky.txt','w')
file.close()
np.savetxt('xdiffky.txt', xdiffky, delimiter='\n', header='Difference between HW Centers at ky=0 and ky=2pi')
file = open('xdiffkz.txt','w')
file.close()
np.savetxt('xdiffkz.txt', xdiffkz, delimiter='\n', header='Difference between HW Centers at kz=0 and kz=2pi')
# Plot this difference as a function of kz/y
figdiffy = plt.figure()
plt.plot(kz, xdiffky) 
plt.show
figdiffz = plt.figure()
plt.plot(ky, xdiffkz) 
plt.show 

#Plot the cut of HW Centers at exact value of kz/y as a function of ky/z  
figy = plt.figure()
plt.plot(ky,xAverage[:,100])
plt.show

figz = plt.figure()
plt.plot(kz,xAverage[100,:])
plt.show
    
#fig = plt.figure()
#ax = plt.axes(projection='3d')
#ax.plot_surface(kky, kkz, xAverage,rstride=1, cstride=1,
#                cmap='viridis', edgecolor='none')
#ax.view_init(0,90) 
   
