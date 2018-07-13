# -*- coding: utf-8 -*-
"""
Created on Thu Jul 12 09:39:27 2018

@author: aleksandra
"""

# Calculate Hopf invariant directly from Hamiltonian
# For this use formulas for F, A in terms of Hx, Hy, Hz

import numpy as np
from numpy import multiply as mlt
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

# Cartesian coordinates, indexing for correct order x,y,z (not y,x,z)
[kkx, kky, kkz] = np.meshgrid(kx, ky, kz, indexing = 'ij')

# Coefficients in front of sigma_x,y,z in Hopf Hamiltonian
lamb = np.divide(1, np.power(np.sin(kkx), 2) + np.power(np.sin(kky), 2) + 
                 np.power(np.sin(kkz), 2) + 
                 np.power(np.cos(kkx) + np.cos(kky) + np.cos(kkz) + h, 2))

Hx = mlt(2 * lamb, mlt(np.sin(kkx), np.sin(kkz)) + 
                 t*mlt(np.sin(kky), (np.cos(kkx) + np.cos(kky) + 
                                             np.cos(kkz) + h)))
Hy = mlt(2 * lamb, t*mlt(np.sin(kky), np.sin(kkz)) -
                 mlt(np.sin(kkx), (np.cos(kkx) + np.cos(kky) + 
                                           np.cos(kkz) + h)))
Hz = mlt(lamb, (np.power(np.sin(kkx), 2) + 
                        t**2 * np.power(np.sin(kky), 2) - 
                        np.power(np.sin(kkz), 2) - 
                        np.power((np.cos(kkx) + np.cos(kky) + 
                                  np.cos(kkz) + h), 2)))

# Constract partial derivatives, Hxy = d_yH_x
Hxx = Hx[1:Nx, 0:Ny-1, 0:Nz-1] - Hx[0:Nx-1, 0:Ny-1, 0:Nz-1]
Hxy = Hx[0:Nx-1, 1:Ny, 0:Nz-1] - Hx[0:Nx-1, 0:Ny-1, 0:Nz-1]
Hxz = Hx[0:Nx-1, 0:Ny-1, 1:Nz] - Hx[0:Nx-1, 0:Ny-1, 0:Nz-1]

Hyx = Hy[1:Nx, 0:Ny-1, 0:Nz-1] - Hy[0:Nx-1, 0:Ny-1, 0:Nz-1]
Hyy = Hy[0:Nx-1, 1:Ny, 0:Nz-1] - Hy[0:Nx-1, 0:Ny-1, 0:Nz-1]
Hyz = Hy[0:Nx-1, 0:Ny-1, 1:Nz] - Hy[0:Nx-1, 0:Ny-1, 0:Nz-1]

Hzx = Hz[1:Nx, 0:Ny-1, 0:Nz-1] - Hz[0:Nx-1, 0:Ny-1, 0:Nz-1]
Hzy = Hz[0:Nx-1, 1:Ny, 0:Nz-1] - Hz[0:Nx-1, 0:Ny-1, 0:Nz-1]
Hzz = Hz[0:Nx-1, 0:Ny-1, 1:Nz] - Hz[0:Nx-1, 0:Ny-1, 0:Nz-1]

# Don't take the point k=2pi (equal to k=0)
Hx = Hx[0:Nx-1, 0:Ny-1, 0:Nz-1]
Hy = Hy[0:Nx-1, 0:Ny-1, 0:Nz-1]
Hz = Hz[0:Nx-1, 0:Ny-1, 0:Nz-1]

# Berry curvature vector
Fx = 1/4/pi * (mlt(Hx, (mlt(Hyy, Hzz) - mlt(Hzy, Hyz))) + 
               mlt(Hy, (mlt(Hzy, Hxz) - mlt(Hxy, Hzz))) +
               mlt(Hz, (mlt(Hxy, Hyz) - mlt(Hyy, Hxz))))
Fy = 1/4/pi * (mlt(Hx, (mlt(Hyz, Hzx) - mlt(Hzz, Hyx))) + 
               mlt(Hy, (mlt(Hzz, Hxx) - mlt(Hxz, Hzx))) +
               mlt(Hz, (mlt(Hxz, Hyx) - mlt(Hyz, Hxx))))
Fz = 1/4/pi * (mlt(Hx, (mlt(Hyx, Hzy) - mlt(Hzx, Hyy))) + 
               mlt(Hy, (mlt(Hzx, Hxy) - mlt(Hxx, Hzy))) +
               mlt(Hz, (mlt(Hxx, Hyy) - mlt(Hyx, Hxy))))

# Berry connection depends on the gauge
# We constract it in almost the whole BZ in one gauge
# and use another only in the vicinity of the points with singularity

# Set the size of Berry connection vector
Ax = np.zeros((Nx-1, Ny-1, Nz-1))
Ay = np.zeros((Nx-1, Ny-1, Nz-1))
Az = np.zeros((Nx-1, Ny-1, Nz-1))

# Approximate number of k=pi point
midNx = round(Nx/2)
midNy = round(Ny/2)
midNz = round(Nz/2)

# Set vicinity
nx = 5
ny = 5
nz = 5

# Define numinators and denominators of Berry connection in both gauges
def Berryfrac(xst, xf, yst, yf):
    berrynumtorx = (mlt(Hy[xst:xf, yst:yf, :], Hxx[xst:xf, yst:yf, :]) - 
                    mlt(Hx[xst:xf, yst:yf, :], Hyx[xst:xf, yst:yf, :]))
    
    berrynumtory = (mlt(Hy[xst:xf, yst:yf, :], Hxy[xst:xf, yst:yf, :]) - 
                    mlt(Hx[xst:xf, yst:yf, :], Hyy[xst:xf, yst:yf, :]))
    
    berrynumtorz = (mlt(Hy[xst:xf, yst:yf, :], Hxz[xst:xf, yst:yf, :]) - 
                    mlt(Hx[xst:xf, yst:yf, :], Hyz[xst:xf, yst:yf, :]))
    
    berrydenomtor1 = 1 + Hz[xst:xf, yst:yf, :]
    
    berrydenomtor2 = 1 - Hz[xst:xf, yst:yf, :]
    
    return [berrynumtorx, berrynumtory, berrynumtorz, 
            berrydenomtor1, berrydenomtor2]

# Berry connection in the first gauge    
def Berryconnection1(xst, xf, yst, yf):
    
    [berrynumtorx, berrynumtory, berrynumtorz, 
            berrydenomtor1, berrydenomtor2] = Berryfrac(xst, xf, yst, yf)
    
    berry1x = -1/2 * np.divide(berrynumtorx, berrydenomtor1)
    berry1y = -1/2 * np.divide(berrynumtory, berrydenomtor1)
    berry1z = -1/2 * np.divide(berrynumtorz, berrydenomtor1)
    
    return berry1x, berry1y, berry1z

# Around points ky=kz=0 choose another gauge
def Berryconnection2(xst, xf, yst, yf):
    
    [berrynumtorx, berrynumtory, berrynumtorz, 
            berrydenomtor1, berrydenomtor2] = Berryfrac(xst, xf, yst, yf)
    
    berry2x = 1/2 * np.divide(berrynumtorx, berrydenomtor2)
    berry2y = 1/2 * np.divide(berrynumtory, berrydenomtor2)
    berry2z = 1/2 * np.divide(berrynumtorz, berrydenomtor2)
    
    return berry2x, berry2y, berry2z

# Define Berry connection at all points far enough from k_x,y=0,pi,2pi    
[Ax[nx : midNx-nx, ny : midNy-ny, :],
    Ay[nx : midNx-nx, ny : midNy-ny, :],
    Az[nx : midNx-nx, ny : midNy-ny, :]] = (
    Berryconnection1(nx, midNx-nx, ny, midNy-ny))
        
[Ax[nx : midNx-nx, midNy+ny : Ny-ny, :],
    Ay[nx : midNx-nx, midNy+ny : Ny-ny, :],
    Az[nx : midNx-nx, midNy+ny : Ny-ny, :]] = (
    Berryconnection1(nx, midNx-nx, midNy+ny, Ny-ny))
        
[Ax[midNx+nx : Nx-nx, ny : midNy-ny, :],
    Ay[midNx+nx : Nx-nx, ny : midNy-ny, :],
    Az[midNx+nx : Nx-nx, ny : midNy-ny, :]] = (
    Berryconnection1(midNx+nx, Nx-nx, ny, midNy-ny))
        
[Ax[midNx+nx : Nx-nx, midNy+ny : Ny-ny, :],
    Ay[midNx+nx : Nx-nx, midNy+ny : Ny-ny, :],
    Az[midNx+nx : Nx-nx, midNy+ny : Ny-ny, :]] = (
    Berryconnection1(midNx+nx, Nx-nx, midNy+ny, Ny-ny))

# In the vicinities of k_x,y=0,pi,2pi use the second gauge
[Ax[0 : nx, 0 : ny, :], 
    Ay[0 : nx, 0 : ny, :], 
    Az[0 : nx, 0 : ny, :]] = (
    Berryconnection2(0, nx, 0, nx))
    
[Ax[0 : nx, midNy-ny : midNy+ny, :], 
    Ay[0 : nx, midNy-ny : midNy+ny, :],
    Az[0 : nx, midNy-ny : midNy+ny, :]] = (
    Berryconnection2(0, nx, midNy-ny, midNy+ny))
    
[Ax[0 : nx, Ny-ny : Ny-1, :], 
    Ay[0 : nx, Ny-ny : Ny-1, :], 
    Az[0 : nx, Ny-ny : Ny-1, :]] = (
    Berryconnection2(0, nx, Ny-ny, Ny-1))

[Ax[midNx-nx : midNx+nx, 0 : ny, :], 
    Ay[midNx-nx : midNx+nx, 0 : ny, :], 
    Az[midNx-nx : midNx+nx, 0 : ny, :]] = (
    Berryconnection2(midNx-nx, midNx+nx, 0, nx))

[Ax[midNx-nx : midNx+nx, midNy-ny : midNy+ny, :],
    Ay[midNx-nx : midNx+nx, midNy-ny : midNy+ny, :],
    Az[midNx-nx : midNx+nx, midNy-ny : midNy+ny, :]] = (
    Berryconnection2(midNx-nx, midNx+nx, midNy-ny, midNy+ny))

[Ax[midNx-nx : midNx+nx, Ny-ny : Ny-1, :], 
    Ay[midNx-nx : midNx+nx, Ny-ny : Ny-1, :], 
    Az[midNx-nx : midNx+nx, Ny-ny : Ny-1, :]] = (
    Berryconnection2(midNx-nx, midNx+nx, Ny-ny, Ny-1))
     
[Ax[Nx-nx : Nx-1, 0 : ny, :], 
    Ay[Nx-nx : Nx-1, 0 : ny, :], 
    Az[Nx-nx : Nx-1, 0 : ny, :]] = (
    Berryconnection2(Nx-nx, Nx-1, 0, nx))

[Ax[Nx-nx : Nx-1, midNy-ny : midNy+ny, :], 
    Ay[Nx-nx : Nx-1, midNy-ny : midNy+ny, :], 
    Az[Nx-nx : Nx-1, midNy-ny : midNy+ny, :]] = (
    Berryconnection2(Nx-nx, Nx-1, midNy-ny, midNy+ny)) 

[Ax[Nx-nx : Nx-1, Ny-ny : Ny-1, :], 
    Ay[Nx-nx : Nx-1, Ny-ny : Ny-1, :], 
    Az[Nx-nx : Nx-1, Ny-ny : Ny-1, :]] = (
    Berryconnection2(Nx-nx, Nx-1, Ny-ny, Ny-1))

# Constract F*A to calculate Hopf invariant
FA =  mlt(Fx, Ax) + mlt(Fy, Ay) + mlt(Fz, Az)

# Hopf invariant is a sum over the whole Brillouen zone of F*A
print(sum(sum(sum(FA))))

# Show integrand at exact kz point as a function of kx, ky
plt.imshow(FA[:, :, 10], cmap='RdBu')
plt.colorbar()
plt.show()
