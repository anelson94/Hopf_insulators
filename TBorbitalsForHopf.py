# -*- coding: utf-8 -*-
"""
Created on Wed Jul 11 10:17:05 2018

@author: aleksandra
"""

# Plot orbitals that provide Hopf Hamiltonian

import numpy as np
from numpy import power as pw
from mayavi import mlab

# Parameters of Gauss functions
ax = 1
ay = 1
az = 0.5
axpy = 0.7
axmy = 0.7
axpz = 0.7
axmz = 0.7
aypz = 0.7
aymz = 0.7
b = 0.2
b2 = 0.4

Nx = 61
Ny = 61
Nz = 31

x = np.linspace(-3*ax, 3*ax, Nx)
y = np.linspace(-3*ay, 3*ay, Ny)
z = np.linspace(-3*az, 3*az, Nz)

# Define a grid
[xx, yy, zz] = np.meshgrid(x, y, z, indexing = 'ij')

# Basis orbitals are linear combinations of Gauss functions
Psi1 = (np.exp( - 2*pw((xx - ax)/ax, 2) - 2*pw(yy/b, 2) - 2*pw(zz/b, 2)) +
       np.exp( - 2*pw((xx + yy - axpy)/axpy, 2) - 
              2*pw((yy - xx)/b, 2) - 2*pw(zz/b, 2)) + 
       np.exp( - 2*pw((yy - ay)/ay, 2) - 2*pw(xx/b, 2) - 2*pw(zz/b, 2)) - 
       np.exp( - 2*pw((yy - xx - axmy)/axmy, 2) - 
              2*pw((yy + xx)/b, 2) - 2*pw(zz/b, 2)) - 
       np.exp( - 2*pw((xx + ax)/ax, 2) - 2*pw(yy/b, 2) - 2*pw(zz/b, 2)) - 
       np.exp( - 2*pw((xx + yy + axpy)/axpy, 2) - 
              2*pw((yy - xx)/b, 2) - 2*pw(zz/b, 2)) - 
       np.exp( - 2*pw((yy + ay)/ay, 2) - 2*pw(xx/b, 2) - 2*pw(zz/b, 2)) +
       np.exp( - 2*pw((yy - xx + axmy)/axmy, 2) - 
              2*pw((yy + xx)/b, 2) - 2*pw(zz/b, 2)) +
       np.exp( - 2*pw((xx + zz - axpz)/axpz, 2) - 
              2*pw((xx - zz)/b, 2) - 2*pw(yy/b, 2)) +
       np.exp( - 2*pw((yy + zz - aypz)/aypz, 2) - 
              2*pw((yy - zz)/b2, 2) - 2*pw(xx/b2, 2)) -
       np.exp( - 2*pw((zz - xx - axmz)/axmz, 2) - 
              2*pw((xx + zz)/b, 2) - 2*pw(yy/b, 2)) -
       np.exp( - 2*pw((zz - yy - aymz)/aymz, 2) - 
              2*pw((yy + zz)/b2, 2) - 2*pw(xx/b2, 2)) +
       np.exp( - 2*pw((xx - zz - axmz)/axmz, 2) - 
              2*pw((xx + zz)/b, 2) - 2*pw(yy/b, 2)) -
       np.exp( - 2*pw((yy - zz - aymz)/aymz, 2) - 
              2*pw((yy + zz)/b2, 2) - 2*pw(xx/b2, 2)) -
       np.exp( - 2*pw((xx + zz + axpz)/axpz, 2) - 
              2*pw((xx - zz)/b, 2) - 2*pw(yy/b, 2)) +
       np.exp( - 2*pw((yy + zz + aypz)/aypz, 2) - 
              2*pw((yy - zz)/b2, 2) - 2*pw(xx/b2, 2)))

# Normalize the Orbital wavefunction        
sqPsi1 = pw(Psi1, 2) 

sqPsi1[0, :, :] = sqPsi1[0, :, :] / 2
sqPsi1[Nx-1, :, :]  = sqPsi1[Nx-1, :, :] / 2 
sqPsi1[:, 0, :] = sqPsi1[:, 0, :] / 2
sqPsi1[:, Ny-1, :]  = sqPsi1[:, Ny-1, :] / 2    
sqPsi1[:, :, 0] = sqPsi1[:, :, 0] / 2
sqPsi1[:, :, Nz-1]  = sqPsi1[:, :, Nz-1] / 2 

# Calculate the norm using trapeze method
modPsi1 = np.sum(sqPsi1) * 6*ax / (Nx-1) * 6*ay / (Ny-1) * 6*az / (Nz-1)
Psi1 = Psi1 / modPsi1
sqPsi1 = []
modPsi1 = []

# The second orbital      
Psi2 = ( - np.exp( - 2*pw((xx - ax)/ax, 2) - 2*pw(yy/b, 2) - 2*pw(zz/b, 2)) -
       np.exp( - 2*pw((xx + yy - axpy)/axpy, 2) - 
              2*pw((yy - xx)/b, 2) - 2*pw(zz/b, 2)) + 
       np.exp( - 2*pw((yy - ay)/ay, 2) - 2*pw(xx/b, 2) - 2*pw(zz/b, 2)) + 
       np.exp( - 2*pw((yy - xx - axmy)/axmy, 2) - 
              2*pw((yy + xx)/b, 2) - 2*pw(zz/b, 2)) - 
       np.exp( - 2*pw((xx + ax)/ax, 2) - 2*pw(yy/b, 2) - 2*pw(zz/b, 2)) - 
       np.exp( - 2*pw((xx + yy + axpy)/axpy, 2) - 
              2*pw((yy - xx)/b, 2) - 2*pw(zz/b, 2)) + 
       np.exp( - 2*pw((yy + ay)/ay, 2) - 2*pw(xx/b, 2) - 2*pw(zz/b, 2)) +
       np.exp( - 2*pw((yy - xx + axmy)/axmy, 2) - 
              2*pw((yy + xx)/b, 2) - 2*pw(zz/b, 2)) -
       np.exp( - 2*pw((xx + zz - axpz)/axpz, 2) - 
              2*pw((xx - zz)/b, 2) - 2*pw(yy/b, 2)) -
       np.exp( - 2*pw((yy + zz - aypz)/aypz, 2) - 
              2*pw((yy - zz)/b, 2) - 2*pw(xx/b, 2)) -
       np.exp( - 2*pw((zz - xx - axmz)/axmz, 2) - 
              2*pw((xx + zz)/b, 2) - 2*pw(yy/b, 2)) -
       np.exp( - 2*pw((zz - yy - aymz)/aymz, 2) - 
              2*pw((yy + zz)/b, 2) - 2*pw(xx/b, 2)) +
       np.exp( - 2*pw((xx - zz - axmz)/axmz, 2) - 
              2*pw((xx + zz)/b, 2) - 2*pw(yy/b, 2)) +
       np.exp( - 2*pw((yy - zz - aymz)/aymz, 2) - 
              2*pw((yy + zz)/b, 2) - 2*pw(xx/b, 2)) +
       np.exp( - 2*pw((xx + zz + axpz)/axpz, 2) - 
              2*pw((xx - zz)/b, 2) - 2*pw(yy/b, 2)) +
       np.exp( - 2*pw((yy + zz + aypz)/aypz, 2) - 
              2*pw((yy - zz)/b, 2) - 2*pw(xx/b, 2)))

# Normalization
sqPsi2 = pw(Psi2, 2) 

sqPsi2[0, :, :] = sqPsi2[0, :, :] / 2
sqPsi2[Nx-1, :, :]  = sqPsi2[Nx-1, :, :] / 2 
sqPsi2[:, 0, :] = sqPsi2[:, 0, :] / 2
sqPsi2[:, Ny-1, :]  = sqPsi2[:, Ny-1, :] / 2    
sqPsi2[:, :, 0] = sqPsi2[:, :, 0] / 2
sqPsi2[:, :, Nz-1]  = sqPsi2[:, :, Nz-1] / 2 

modPsi2 = np.sum(sqPsi2) * 6*ax / (Nx-1) * 6*ay / (Ny-1) * 6*az / (Nz-1)
Psi2 = Psi2 / modPsi2
sqPsi2 = []
modPsi2 = []

# Plot isosurfaces of basis orbitals
mlab.contour3d(Psi1, contours=4, transparent=False)
mlab.show()

mlab.contour3d(Psi2, contours=4, transparent=False)
mlab.show()

