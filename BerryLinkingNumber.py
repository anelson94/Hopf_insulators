"""
 Created by alexandra at 30.08.19 11:37

 Calculate the linking number of the Berry curvature of Hopf insulator
"""

import numpy as np
from math import pi
import math, cmath
from mayavi import mlab
import pickle


def u1(kx, ky, kz):
    """First component of eigenvector"""
    lamb = math.sqrt(math.sin(kx)**2 + math.sin(ky)**2 + math.sin(kz)**2
                     + (math.cos(kx) + math.cos(ky) + math.cos(kz) + h)**2)
    return (math.sin(kz) - 1j * (
            math.cos(kx) + math.cos(ky) + math.cos(kz) + h)) / lamb


def u2(kx, ky, kz):
    """Second component of eigenvector"""
    lamb = math.sqrt(math.sin(kx) ** 2 + math.sin(ky) ** 2 + math.sin(kz) ** 2
                     + (math.cos(kx) + math.cos(ky) + math.cos(kz) + h) ** 2)
    return (-math.sin(kx) + 1j * math.sin(ky)) / lamb


def berryx(kx, ky, kz):
    """Calculate x component of Berry curvature at kx, ky, kz point"""
    dk = 0.01
    ovlp1 = (np.conj(u1(kx, ky - dk, kz - dk)) * u1(kx, ky + dk, kz - dk)
             + np.conj(u2(kx, ky - dk, kz - dk)) * u2(kx, ky + dk, kz - dk))
    ovlp2 = (np.conj(u1(kx, ky + dk, kz - dk)) * u1(kx, ky + dk, kz + dk)
             + np.conj(u2(kx, ky + dk, kz - dk)) * u2(kx, ky + dk, kz + dk))
    ovlp3 = (np.conj(u1(kx, ky + dk, kz + dk)) * u1(kx, ky - dk, kz + dk)
             + np.conj(u2(kx, ky + dk, kz + dk)) * u2(kx, ky - dk, kz + dk))
    ovlp4 = (np.conj(u1(kx, ky - dk, kz + dk)) * u1(kx, ky - dk, kz - dk)
             + np.conj(u2(kx, ky - dk, kz + dk)) * u2(kx, ky - dk, kz - dk))

    return -cmath.log(ovlp1 * ovlp2 * ovlp3 * ovlp4).imag / 4 / dk**2


def berryy(kx, ky, kz):
    """Calculate y component of Berry curvature at kx, ky, kz point"""
    dk = 0.01
    ovlp1 = (np.conj(u1(kx - dk, ky, kz - dk)) * u1(kx - dk, ky, kz + dk)
             + np.conj(u2(kx - dk, ky, kz - dk)) * u2(kx - dk, ky, kz + dk))
    ovlp2 = (np.conj(u1(kx - dk, ky, kz + dk)) * u1(kx + dk, ky, kz + dk)
             + np.conj(u2(kx - dk, ky, kz + dk)) * u2(kx + dk, ky, kz + dk))
    ovlp3 = (np.conj(u1(kx + dk, ky, kz + dk)) * u1(kx + dk, ky, kz - dk)
             + np.conj(u2(kx + dk, ky, kz + dk)) * u2(kx + dk, ky, kz - dk))
    ovlp4 = (np.conj(u1(kx + dk, ky, kz - dk)) * u1(kx - dk, ky, kz - dk)
             + np.conj(u2(kx + dk, ky, kz - dk)) * u2(kx - dk, ky, kz - dk))

    return -cmath.log(ovlp1 * ovlp2 * ovlp3 * ovlp4).imag / 4 / dk**2


def berryz(kx, ky, kz):
    """Calculate z component of Berry curvature at kx, ky, kz point"""
    dk = 0.01
    ovlp1 = (np.conj(u1(kx - dk, ky - dk, kz)) * u1(kx + dk, ky - dk, kz)
             + np.conj(u2(kx - dk, ky - dk, kz)) * u2(kx + dk, ky - dk, kz))
    ovlp2 = (np.conj(u1(kx + dk, ky - dk, kz)) * u1(kx + dk, ky + dk, kz)
             + np.conj(u2(kx + dk, ky - dk, kz)) * u2(kx + dk, ky + dk, kz))
    ovlp3 = (np.conj(u1(kx + dk, ky + dk, kz)) * u1(kx - dk, ky + dk, kz)
             + np.conj(u2(kx + dk, ky + dk, kz)) * u2(kx - dk, ky + dk, kz))
    ovlp4 = (np.conj(u1(kx - dk, ky + dk, kz)) * u1(kx - dk, ky - dk, kz)
             + np.conj(u2(kx - dk, ky + dk, kz)) * u2(kx - dk, ky - dk, kz))

    return -cmath.log(ovlp1 * ovlp2 * ovlp3 * ovlp4).imag / 4 / dk ** 2


def normalize(bx, by, bz):
    norm = 1  # math.sqrt(bx**2 + by**2 + bz**2)
    return bx / norm, by / norm, bz / norm


h = 1.5

# Three points in S2 (3d vectors with unitary norms)
n1 = np.array([1, 0, 0])
n2 = np.array([0, 1, 0])
n3 = np.array([0, 0, -1])

Nk = 80

eps1 = 0.02
eps2 = 0.1

# Pauli matrices
sigmax = np.array([[0, 1], [1, 0]])
sigmay = np.array([[0, -1j], [1j, 0]])
sigmaz = np.array([[1, 0], [0, -1]])

Image1 = np.empty((Nk, Nk, Nk), dtype=int)
Image2 = np.empty((Nk, Nk, Nk), dtype=int)
f = np.empty((Nk, Nk, Nk, 2, 2), dtype=complex)
Bmin = 1
Bmax = 0
for idx in range(Nk):
    for idy in range(Nk):
        for idz in range(Nk):
            kx = 2 * pi * idx / (Nk - 1)
            ky = 2 * pi * idy / (Nk - 1)
            kz = 2 * pi * idz / (Nk - 1)
            Bx, By, Bz = normalize(
                berryx(kx, ky, kz), berryy(kx, ky, kz), berryz(kx, ky, kz))
            f[idx, idy, idz, :, :] = Bx * sigmax + By * sigmay + Bz * sigmaz
            Bnorm = Bx**2 + By**2 + Bz**2
            if Bnorm < Bmin:
                Bmin = Bnorm
            if Bnorm > Bmax:
                Bmax = Bnorm

            # Image1[idx, idy, idz] = np.all(np.stack(
            #     (math.isclose(Bx, n1[0], rel_tol=eps2, abs_tol=eps2),
            #      math.isclose(By, n1[1], rel_tol=eps2, abs_tol=eps2),
            #      math.isclose(Bz, n1[2], rel_tol=eps2, abs_tol=eps2)))) * 1
            #
            # Image2[idx, idy, idz] = np.all(np.stack(
            #     (math.isclose(Bx, n2[0], rel_tol=eps2, abs_tol=eps2),
            #      math.isclose(By, n2[1], rel_tol=eps2, abs_tol=eps2),
            #      math.isclose(Bz, n2[2], rel_tol=eps2, abs_tol=eps2)))) * 1

print(Bmin, Bmax)
# Calculate eigenvalues and eigenvectors of f hamiltonian
[En, eigfunc] = np.linalg.eigh(f)

with open('Berryeigen.pickle', 'wb') as f:
    pickle.dump([En, eigfunc], f)
# print('Images made')
#
# mlab.figure(bgcolor=(1, 1, 1))
# con1 = mlab.contour3d(Image1, contours=2, color=(1, 153/255, 153/255),
#                       transparent=False)
# mlab.axes(con1, xlabel='', ylabel='', zlabel='', color=(0, 0, 0))
# mlab.contour3d(Image2, contours=2, color=(153/255, 1, 153/255),
#                transparent=False)
# # mlab.contour3d(Image3, contours=2, color=(153/255, 153/255, 1),
# #                transparent=False)
# mlab.show()

