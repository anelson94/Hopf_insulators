"""
 Created by alexandra at 18.02.19 10:24

 Calculate Chern numbers using Z2Pack
"""

import z2pack
import numpy as np
from math import sin, cos, pi
import matplotlib.pyplot as plt


def hamiltonian(k):
    kx, ky, kz = k
    hz = (sin(kx)**2 + sin(ky)**2 - sin(kz)**2
          - (cos(kx) + cos(ky) + cos(kz) + h)**2 + A)
    hxy = 2 * (sin(kx) * sin(kz) + sin(ky) * (cos(kx) + cos(ky) + cos(kz) + h)
               - 1j * sin(ky) * sin(kz)
               + 1j * sin(kx) * (cos(kx) + cos(ky) + cos(kz) + h))

    return np.array([
        [hz, hxy],
        [hxy.conjugate(), -hz]
    ])


h = -1.5
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
        surface=lambda t1, t2: [1, t1, t2]
        # surface=z2pack.shape.Sphere([0.5, pi-0.5, pi], 0.1)
    )
z2pack.plot.chern(result)
plt.show()

print(z2pack.invariant.chern(result))
