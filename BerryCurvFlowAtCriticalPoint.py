# -*- coding: utf-8 -*-
"""
Created on Tue Oct  2 15:28:20 2018

@author: aleksandra
"""

# At critical points (h = 1) calculate the Chern number flow 
# from the points where energy bands close

import numpy as np
from math import pi

def Berrycurvyz(kxeps):
    # Define yz component of Berry curvature (which points in x direction)
    # at point kx on a grid (kyz)
    
    Fyz = (2 * (np.cos(kyz_z) * np.sin(kyz_y) * (np.power(np.sin(kxeps), 2) + 
                                            t**2 * np.power(np.sin(kyz_y), 2)) + 
                 t * np.cos(kyz_y) * (np.sin(kyz_y) * t * (1 + (h + np.cos(kxeps) + 
                                                     np.cos(kyz_y)) * np.cos(kyz_z)) - 
                 (h + np.cos(kxeps) + np.cos(kyz_y)) * np.sin(kxeps) * np.sin(kyz_z))) /
           (np.power(np.power(np.sin(kxeps), 2) + t**2 * np.power(np.sin(kyz_y), 2) +
                     np.power(np.sin(kyz_z), 2) + np.power(np.cos(kxeps) + np.cos(kyz_y) + 
                                                     np.cos(kyz_z) + h, 2), 2)))
    return Fyz

def Berrycurvxz(kyeps):
    # Define xz component of Berry curvature (which points in y direction)
    # at point ky on a grid (kxz)
    
    Fxz = ( - 2 * (np.cos(kxz_z) * np.sin(kxz_x) * (np.power(np.sin(kxz_x), 2) + 
                                            t**2 * np.power(np.sin(kyeps), 2)) + 
                   np.cos(kxz_x) * (np.sin(kxz_x) * (1 + (h + np.cos(kxz_x) + 
                                                     np.cos(kyeps)) * np.cos(kxz_z)) + 
                 t * (h + np.cos(kxz_x) + np.cos(kyeps)) * np.sin(kyeps) * np.sin(kxz_z))) /
           (np.power(np.power(np.sin(kxz_x), 2) + t**2 * np.power(np.sin(kyeps), 2) +
                     np.power(np.sin(kxz_z), 2) + np.power(np.cos(kxz_x) + np.cos(kyeps) + 
                                                     np.cos(kxz_z) + h, 2), 2)))
    return Fxz

def Berrycurvxy(kzeps):
    # Define xy component of Berry curvature (which points in z direction)
    # at point kz on a grid (kxy)
    
    Fxy =  ((2 * t * np.power(np.cos(kxy_y), 2) + t * (5 + np.cos(2*kxy_x)) * 
             np.cos(kxy_y) * (h + np.cos(kzeps)) + 2 * np.power(np.cos(kxy_x), 2) * 
             (t + t * np.cos(kxy_y) * (h + np.cos(kzeps))) + t * np.cos(kxy_x) * 
             (2 * np.power(np.cos(kxy_y), 2) * (h + np.cos(kzeps)) + (5 + np.cos(2*kxy_y)) 
             * (h + np.cos(kzeps)) + 4 * np.cos(kxy_y) * (3 + h**2 + 2 * h * np.cos(kzeps))) -
             2 * (-2 * t + t * np.power(np.sin(kxy_x), 2) + t * np.power(np.sin(kxy_y), 2) - 
             np.sin(2*kxy_x) * np.sin(kxy_y) * np.sin(kzeps) + t**2 * np.sin(kxy_x) * 
             np.sin(2*kxy_y) * np.sin(kzeps))) / 2 / 
             (np.power(np.power(np.sin(kxy_x), 2) + t**2 * np.power(np.sin(kxy_y), 2) +
                     np.power(np.sin(kzeps), 2) + np.power(np.cos(kxy_x) + np.cos(kxy_y) + 
                                                     np.cos(kzeps) + h, 2), 2)))
    return Fxy
    
def trapWeight(Nx):
    # Define the weight function for the integration by trapeze
    weight = np.ones((Nx - 2, Nx - 2))
    weightcolumn = np.ones(Nx - 2)
    weight = np.concatenate((weightcolumn[np.newaxis, :] / 2, weight, 
                             weightcolumn[np.newaxis, :] / 2), axis = 0)

    weightcolumn = np.append(weightcolumn / 2, np.array([1 / 4]), axis = 0)
    weightcolumn = np.append(np.flip(weightcolumn, axis = 0), np.array([1 / 4]), axis = 0)
    weight = np.concatenate((weightcolumn[:, np.newaxis], weight, 
                             weightcolumn[:, np.newaxis]), axis = 1)
    
    return weight

# Set parameters
h = 1 # critical value
t = 1

# Crossing point
kcx = pi
kcy = 0
kcz = pi

# Size of crossing point vicinity
eps = 0.01

# k grid
Nx = 101
Ny = Nx
Nz = Nx

kx = np.linspace(kcx - eps, kcx + eps, Nx)
ky = np.linspace(kcy - eps, kcy + eps, Ny)
kz = np.linspace(kcz - eps, kcz + eps, Nz)

[kxy_x, kxy_y] = np.meshgrid(kx, ky, indexing = 'ij')
[kxz_x, kxz_z] = np.meshgrid(kx, kz, indexing = 'ij')
[kyz_y, kyz_z] = np.meshgrid(ky, kz, indexing = 'ij')

# Berry flux
Chern = (np.sum(Berrycurvxy(eps) * trapWeight(Nx)) +
         np.sum(Berrycurvxy(-eps) * trapWeight(Nx)) + 
         np.sum(Berrycurvxz(eps) * trapWeight(Nx)) + 
         np.sum(Berrycurvxz(-eps) * trapWeight(Nx)) +
         np.sum(Berrycurvyz(eps) * trapWeight(Nx)) +
         np.sum(Berrycurvyz(-eps) * trapWeight(Nx))) * (2 * eps)**2 / (Nx - 1)**2


print(Chern)
#print((np.sum(Berrycurvyz(eps) * trapWeight(Nx)) / (Nx - 1)**2 -
#         np.sum(Berrycurvyz(-eps) * trapWeight(Nx)) / (Nx - 1)**2))
