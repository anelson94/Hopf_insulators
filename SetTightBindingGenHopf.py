"""
 Created by alexandra at 13.12.18 17:06

 For generalized Hopf hamiltonian create tight binding model
"""

from math import sin, cos, pi
import cmath
from scipy import integrate
import scipy
import numpy as np


def hamilt11(kx, ky, kz):
    """11 component of Hopf Hamiltonian"""
    return ((sin(kx)**2 + t**2 * sin(ky)**2)**p
            - (sin(kz)**2 + (cos(kx) + cos(ky)
                             + cos(kz) + h)**2)**q)


def hamilt12(kx, ky, kz):
    """12 component of Hopf Hamiltonian"""
    return 2 * ((sin(kx) - 1j * t * sin(ky))**p
                * (sin(kz) - 1j * (cos(kx) + cos(ky)
                                   + cos(kz) + h))**q)


def hamilt21(kx, ky, kz):
    """21 component of Hopf Hamiltonian"""
    return 2 * ((sin(kx) + 1j * t * sin(ky))**p
                * (sin(kz) + 1j * (cos(kx) + cos(ky)
                                   + cos(kz) + h))**q)


def hamilt22(kx, ky, kz):
    """22 component of Hopf Hamiltonian"""
    return -((sin(kx)**2 + t**2 * sin(ky)**2)**p
             - (sin(kz)**2 + (cos(kx) + cos(ky)
                              + cos(kz) + h)**2)**q)


def write_gvec(x, y, z):
    """Write into file the unit cell where the hopping is"""
    ftemp.write(repr(x).rjust(5) + repr(y).rjust(5) + repr(z).rjust(5))


def write_ij(i, j):
    """Write into file the orbital where the hopping is"""
    ftemp.write(repr(i).rjust(5) + repr(j).rjust(5))


def calc_tb(i, j, x, y, z):
    """Calculate hopping to (x, y, z) unit cell and (i, j) orbital"""
    # Define real and imaginary part of the integrand (hamiltonian * exp)
    def funintre(kx, ky, kz):
        return scipy.real(
            hamilt_dict[(i, j)](kx, ky, kz)
            * cmath.exp(-1j * (x * kx + y * ky + z * kz)))

    def funintim(kx, ky, kz):
        return scipy.imag(
            hamilt_dict[(i, j)](kx, ky, kz)
            * cmath.exp(-1j * (x * kx + y * ky + z * kz)))

    # Integrate real and imaginary part separately
    tbre, tberrorre = integrate.tplquad(
        funintre, 0, 2 * pi,
        lambda kx: 0, lambda kx: 2 * pi,
        lambda kx, ky: 0, lambda kx, ky: 2 * pi)
    tbim, tberrorim = integrate.tplquad(
        funintim, 0, 2 * pi,
        lambda kx: 0, lambda kx: 2 * pi,
        lambda kx, ky: 0, lambda kx, ky: 2 * pi)

    tbre = tbre / (2 * pi) ** 3
    tbim = tbim / (2 * pi) ** 3

    # return real and imaginary part of the hopping
    return tbre, tbim


# Parameters of Hopf Hamiltonian
p = 3
q = 1
h = 2
t = 1

# Dictionary of Hamiltonian functions
hamilt_dict = {(0, 0): hamilt11, (0, 1): hamilt12,
               (1, 0): hamilt21, (1, 1): hamilt22}


pqmax = max(p, q)

f = open('model_hopf_gen.dat', 'w')
f.write('Tight-binding model fot generalized Hopf hamiltonian \n')

ftemp = open('model_temporary.dat', 'w')
# Count, how many hoppings in tight-binding model
Nbinding = 0
for a in range(2 * pqmax, -2 * pqmax - 1, -1):
    for b in range(2 * pqmax - abs(a), -2 * pqmax + abs(a) - 1, -1):
        for c in range(min(2 * pqmax - abs(a) - abs(b), 2 * q),
                       -min(2 * pqmax - abs(a) - abs(b), 2 * q) - 1, -1):
            # Construct 2*2 matrix of hoppings to (a, b, c) unit cell
            TBre = np.empty((2, 2))
            TBim = np.empty((2, 2))
            n_zeros = 0
            for i_orb in range(2):
                for j_orb in range(2):
                    # Calculate hoppings and analize
                    TBre[i_orb, j_orb], TBim[i_orb, j_orb] = (
                        calc_tb(i_orb, j_orb, a, b, c))
                    if (abs(TBre[i_orb, j_orb]) < 0.00000001
                            and abs(TBim[i_orb, j_orb]) < 0.00000001):
                        n_zeros += 1
            # If the matrix has nonzero elements, write them
            # in a temporary file
            if n_zeros < 4:
                Nbinding += 1
                for i_orb in range(2):
                    for j_orb in range(2):
                        # Write hoppings in temporary file
                        write_gvec(a, b, c)
                        write_ij(i_orb + 1, j_orb + 1)
                        ftemp.write(
                            '{0:16.8f}{1:16.8f}'.format(TBre[i_orb, j_orb],
                                                        TBim[i_orb, j_orb])
                            + '\n')

ftemp.close()

# Write number of bands
f.write('2 \n')
# Write number of unit cells with hoppings
f.write(repr(round(Nbinding)) + '\n')
# Write degeneracy (always 1)
for idx in range(Nbinding):
    f.write(repr(1).rjust(5))
    if (idx + 1) % 15 == 0:
        f.write('\n')
if Nbinding % 15 != 0:
    f.write('\n')
ftemp = open('model_temporary.dat', 'r')
# Write the hoppings
f.write(ftemp.read())
ftemp.close()

