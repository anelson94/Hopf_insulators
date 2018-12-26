"""
 Created by alexandra at 26.12.18 11:03

 Calculate Hybrid Wannier centers from Analytical solutions
"""


import numpy as np
from math import pi
import matplotlib.pyplot as plt


def u1(kx, ky, kz):
    """First component of eigenvector"""
    lamb = np.sqrt(
        np.power(np.sin(kx), 2) + t**2 * np.power(np.sin(ky), 2)
        + np.power(np.sin(kz), 2) + np.power(
            np.cos(kx) + np.cos(ky) + np.cos(kz) + h, 2))
    return np.divide(np.sin(kz) - 1j * (
            np.cos(kx) + np.cos(ky) + np.cos(kz) + h), lamb)


def u2(kx, ky, kz):
    """Second component of eigenvector"""
    lamb = np.sqrt(
        np.power(np.sin(kx), 2) + t**2 * np.power(np.sin(ky), 2)
        + np.power(np.sin(kz), 2) + np.power(
            np.cos(kx) + np.cos(ky) + np.cos(kz) + h, 2))
    return np.divide(-np.sin(kx) + 1j * t * np.sin(ky), lamb)


def overlap(kx, ky, kz1, kz2):
    """Calculate overlap between two neighbor wavevectors <u_k|u_k+dk>"""
    return (np.conj(u1(kx, ky, kz1)) * u1(kx, ky, kz2)
            + np.conj(u2(kx, ky, kz1)) * u2(kx, ky, kz2))


# parameters
t = 1
h = 1  # critical point

Nx = 100
Ny = 100
Nz = 200

# Set Kx, Ky values
Kx = np.linspace(pi, 2 * pi, Nx)
Ky = np.linspace(pi, 2 * pi, Ny)

# Remove values where determinant is zero
Kx = Kx[1:Nx - 1]
Ky = Ky[1:Ny - 1]

# Add second part of BZ
Kx = np.concatenate((Kx - pi, Kx), axis=None)
Ky = np.concatenate((Ky - pi, Ky), axis=None)
print(Kx.shape)

# Make coordinates in BZ
[Kkx, Kky] = np.meshgrid(Kx, Ky, indexing='ij')

# Calculate the product of overlaps for kz from 0 to 2pi
prodM = 1
for nkz in range(Nz):
    Kz1 = nkz * 2 * pi / Nz
    Kz2 = (nkz + 1) * 2 * pi / Nz
    prodM *= overlap(Kkx, Kky, Kz1, Kz2)

# Calculate hybrid wannier center from product of overlaps
zCenter = -np.imag(np.log(prodM)) / 2 / pi

# Plot the wannier centers as a function of Kx, Ky
fig = plt.figure()
plt.imshow(zCenter)
plt.colorbar()
plt.show()
