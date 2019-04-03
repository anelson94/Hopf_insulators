"""
 Created by alexandra at 12.02.19 14:38

 Calculate polarization of a 1d chain at (kx, ky) from smoothed numerical
 solutions
"""

import numpy as np
from math import pi
import matplotlib.pyplot as plt
import pickle


def scalarprod(a, b):
    # Scalar product of two stackes of wavefunctions of the same size
    # Returns a stack of <A[i,j,...,:]| B[i,j,...,:]>
    prod = np.sum(np.multiply(np.conj(a), b), axis=-1)
    return prod


# Import eigenstates of Hopf Humiltonian
with open('Hopfsmoothstates.pickle', 'rb') as f:
    usmooth = pickle.load(f)

Nz = 1000
Nkgrid = 50

P001 = np.real(
    1j * np.sum(
        1
        - scalarprod(usmooth[:, :, :Nz - 1, :], usmooth[:, :, 1:Nz, :]),
        axis=-1) / 2 / pi)

print(np.sum(P001) / Nkgrid**2)
plt.figure()
plt.imshow(P001)
# plt.plot(Kz[0: Nk - 1], np.real((psi1[:Nk - 1] - psi1[1:]))
#          * np.imag(psi1[:Nk - 1]))
plt.colorbar()
plt.show()
