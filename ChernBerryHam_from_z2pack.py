"""
 Created by alexandra at 30.08.19 15:33

 Calculate Chern number of hamiltonian made from Berry curvature of
 Hopf insualtor using Z2pack
"""

import z2pack
import numpy as np
from math import sin, cos, pi
import matplotlib.pyplot as plt
import cmath, math

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


def hamiltonian(k):
    kx, ky, kz = k
    hz = berryz(kx, ky, kz)
    hxy = berryx(kx, ky, kz) - 1j * berryy(kx, ky, kz)

    return np.array([
        [hz, hxy],
        [hxy.conjugate(), -hz]
    ])


h = 1.5
A = 0
system = z2pack.hm.System(hamiltonian)
# Nz = 31
# chern = np.empty(Nz)
# for nkz in range(Nz):
#     Kz = 2 * pi * nkz / (Nz - 1)
#     result = z2pack.surface.run(
#         system=system,
#         surface=lambda t1, t2: [t1, t2, Kz]
#     )
#
#     chern[nkz] = z2pack.invariant.chern(result)
# print(chern)

result = z2pack.surface.run(
        system=system,
        surface=lambda t1, t2: [0, t1, t2]
        # surface=z2pack.shape.Sphere([0.5, pi-0.5, pi], 0.1)
    )
z2pack.plot.chern(result)
plt.show()

print(z2pack.invariant.chern(result))
