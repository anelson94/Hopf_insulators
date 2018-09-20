# -*- coding: utf-8 -*-
"""
Created on Mon Sep 17 10:03:13 2018

@author: aleksandra
"""

# Construct the images of different points at S2 (H1, H2, H3) in T3 (kx, ky, kz)
# They will be the linked circles

import numpy as np
from mayavi import mlab
import matplotlib.pyplot as plt
from math import pi

# Three points in S2 (3d vectors with unitary norms)
n1 = np.array([1, 0, 0])
n2 = np.array([0, 1/np.sqrt(2), -1/np.sqrt(2)])
n3 = np.array([0, -1/np.sqrt(2), -1/np.sqrt(2)])

# Parameters of Hopf Hamiltonian
t = -1
h = 4

# Set the space of (kx, ky, kz) on the 3d torus T3
Nx = 201
Ny = 201
Nz = 201 

kx = np.linspace(0, 2*pi, Nx)
ky = np.linspace(0, 2*pi, Ny)
kz = np.linspace(0, 2*pi, Nz)

# Cartesian coordinates, indexing for correct order x,y,z (not y,x,z)
[kkx, kky, kkz] = np.meshgrid(kx, ky, kz, indexing = 'ij')


# Hopf Hamiltonian is a mapping function from T^3 to S^2.
lamb = np.divide(1, np.power(np.sin(kkx), 2) + t**2 * np.power(np.sin(kky), 2) + 
                 np.power(np.sin(kkz), 2) + 
                 np.power(np.cos(kkx) + np.cos(kky) + np.cos(kkz) + h, 2))

Hx = np.multiply(2 * lamb, np.multiply(np.sin(kkx), np.sin(kkz)) + 
                 t*np.multiply(np.sin(kky), (np.cos(kkx) + np.cos(kky) + 
                                             np.cos(kkz) + h)))
Hy = np.multiply(2 * lamb, t*np.multiply(np.sin(kky), np.sin(kkz)) -
                 np.multiply(np.sin(kkx), (np.cos(kkx) + np.cos(kky) + 
                                           np.cos(kkz) + h)))
Hz = np.multiply(lamb, (np.power(np.sin(kkx), 2) + 
                        t**2 * np.power(np.sin(kky), 2) - 
                        np.power(np.sin(kkz), 2) - 
                        np.power((np.cos(kkx) + np.cos(kky) + 
                                  np.cos(kkz) + h), 2)))

# Check if Hx is close to nx, Hy to ny and Hz to nz 
# then the point (kx, ky, kz) is an image of point n

# The accuracy of equality is eps1 or eps2
eps1 = 0.02
eps2 = 0.1

# The result is boolean, multiply it by 1 to get 'True' = 1, 'False' = 0
Image1 = np.all(np.stack((np.isclose(Hx, n1[0] * np.ones((Nx, Ny, Nz)), 
                                     eps2, eps2),
                          np.isclose(Hy, n1[1] * np.ones((Nx, Ny, Nz)), 
                                     eps2, eps2),
                          np.isclose(Hz, n1[2] * np.ones((Nx, Ny, Nz)), 
                                     eps2, eps2)), axis = -1), axis = -1)*1
                                     
Image2 = np.all(np.stack((np.isclose(Hx, n2[0] * np.ones((Nx, Ny, Nz)), 
                                     eps1, eps1),
                          np.isclose(Hy, n2[1] * np.ones((Nx, Ny, Nz)), 
                                     eps1, eps1),
                          np.isclose(Hz, n2[2] * np.ones((Nx, Ny, Nz)), 
                                     eps1, eps1)), axis = -1), axis = -1)*1

Image3 = np.all(np.stack((np.isclose(Hx, n3[0] * np.ones((Nx, Ny, Nz)), 
                                     eps1, eps1),
                          np.isclose(Hy, n3[1] * np.ones((Nx, Ny, Nz)), 
                                     eps1, eps1),
                          np.isclose(Hz, n3[2] * np.ones((Nx, Ny, Nz)), 
                                     eps1, eps1)), axis = -1), axis = -1)*1


# Draw the images of points n1, n2, n3 in T3
print(Image1.shape)
mlab.figure()
mlab.contour3d(Image1, contours=2, color = (1, 153/255, 153/255))
mlab.contour3d(Image2, contours=2, color = (153/255, 1, 153/255))
mlab.contour3d(Image3, contours=2, color = (153/255, 153/255, 1))
mlab.show()