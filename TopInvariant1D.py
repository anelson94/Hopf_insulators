"""
 Created by alexandra at 04.09.19 18:15
 Calculate topological invariant of a 1d insulator in 010 direction 
 at high symmetry point Gamma, K or M 
"""

from math import sin, cos, pi
import cmath


def q(k, d):
    """Off diagonal element of flat hamiltonian in 010 direction"""
    lamb = sin(k)**2 + (cos(k) + h + d)**2
    return (
        2 * sin(k) * (cos(k) + h + d)
        - 1j * (sin(k)**2 - (cos(k) + h + d)**2)
    ) / lamb


def invariant(kx, ky):
    delta = cos(kx) + cos(ky)
    q0 = q(0, delta)
    p = 0
    for idy in range(Nk - 1):
        ky = 2 * pi * (idy + 1) / (Nk - 1)
        q1 = q(ky, delta)
        p += -cmath.phase(q0.conjugate() * q1) / 2 / pi
        q0 = q1
    return p


h = 1.5
Nk = 100
print(invariant(0, 0))
