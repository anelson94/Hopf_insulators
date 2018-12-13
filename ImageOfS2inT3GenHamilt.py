"""
 Created by alexandra at 12.12.18 16:41

 Construct the images of different points at S2 (H1, H2, H3) in T3 (kx, ky, kz)
 for generalized Hopf map with p, q \neq 1
"""

import numpy as np
from math import pi
from mayavi import mlab

# Three points in S2 (3d vectors with unitary norms)
n1 = np.array([1, 0, 0])
n2 = np.array([0, 1, 0])
n3 = np.array([0, 0, -1])

# Parameters of Hopf Hamiltonian
t = 1
h = 2
p = 3
q = 1

# Set the space of (kx, ky, kz) on the 3d torus T3
Nx = 201
Ny = 201
Nz = 201

kx = np.linspace(0, 2*pi, Nx)
ky = np.linspace(0, 2*pi, Ny)
kz = np.linspace(0, 2*pi, Nz)

# Cartesian coordinates, indexing for correct order x,y,z (not y,x,z)
[kkx, kky, kkz] = np.meshgrid(kx, ky, kz, indexing='ij')

# Hopf Hamiltonian is a mapping function from T^3 to S^2.
# Construct generalized Hamiltonian with p, q
# Define vector (H_x, H_y, H_z)
lamb = np.divide(
    1, np.power(np.abs(np.sin(kkx) + 1j * t * np.sin(kky)), 2 * p)
    + np.power(np.abs(
        np.sin(kkz)
        + 1j * (np.cos(kkx) + np.cos(kky) + np.cos(kkz) + h)), 2 * q))

Hz = np.multiply(
    lamb, np.power(np.abs(np.sin(kkx) + 1j * t * np.sin(kky)), 2 * p)
    - np.power(np.abs(
        np.sin(kkz)
        + 1j * (np.cos(kkx) + np.cos(kky) + np.cos(kkz) + h)), 2 * q))

H12 = np.multiply(
    2 * lamb, np.multiply(
        np.power(np.sin(kkx) - 1j * t * np.sin(kky), p),
        np.power(np.sin(kkz)
                 + 1j * (np.cos(kkx) + np.cos(kky) + np.cos(kkz) + h), q)))

Hx = np.real(H12)
Hy = - np.imag(H12)

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
                                     eps2, eps2)), axis=-1), axis=-1) * 1

Image2 = np.all(np.stack((np.isclose(Hx, n2[0] * np.ones((Nx, Ny, Nz)),
                                     eps2, eps2),
                          np.isclose(Hy, n2[1] * np.ones((Nx, Ny, Nz)),
                                     eps2, eps2),
                          np.isclose(Hz, n2[2] * np.ones((Nx, Ny, Nz)),
                                     eps2, eps2)), axis=-1), axis=-1) * 1

Image3 = np.all(np.stack((np.isclose(Hx, n3[0] * np.ones((Nx, Ny, Nz)),
                                     eps2, eps2),
                          np.isclose(Hy, n3[1] * np.ones((Nx, Ny, Nz)),
                                     eps2, eps2),
                          np.isclose(Hz, n3[2] * np.ones((Nx, Ny, Nz)),
                                     eps2, eps2)), axis=-1), axis=-1) * 1

print('Images made')

# Plot the resulting preimage
mlab.figure(bgcolor=(1, 1, 1))
con1 = mlab.contour3d(Image1, contours=2, color=(1, 153/255, 153/255),
                      transparent=False)
mlab.axes(con1, xlabel='', ylabel='', zlabel='', color=(0, 0, 0))
mlab.contour3d(Image2, contours=2, color=(153/255, 1, 153/255),
               transparent=False)
# mlab.contour3d(Image3, contours=2, color=(153/255, 153/255, 1),
#                transparent=False)
mlab.show()
