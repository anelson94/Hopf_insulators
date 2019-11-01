"""
 Created by alexandra at 13.10.19 23:07
"""

import numpy as np


class HopfInsulator(object):

    def __init__(self, h, t, m, N_slab, model):
        # store parameters in object
        self.h = h
        self.t = t
        self.m = m
        self.N_slab = N_slab

        if model not in ['initial', 'PRL']:
            raise Exception('unknown model {}'.format(model))

        self.model = model


    def slab_hamiltonian(self, kx, ky):
        """Construct a z-slab hamiltonian for kx,ky wavevectors
        (always written in a 1d array)"""

        ntotal = len(kx)

        # shape: ((kx, ky), ...)
        hh = np.zeros((ntotal, 2*self.N_slab, 2*self.N_slab), dtype=complex)

        # Construct blockes for Hopf Hamiltonian
        # a = (np.power(np.sin(kx), 2) + t ** 2 * np.power(np.sin(ky), 2) -
        #      np.power(np.cos(kx) + np.cos(ky) + h, 2) - 1)
        # b = - np.cos(kx) - np.cos(ky) - h
        # c = 2 * np.multiply(t * np.sin(ky) - 1j * np.sin(kx),
        #                     np.cos(kx) + np.cos(ky) + h)
        # d = 2 * (t * np.sin(ky) - 1j * np.sin(kx))

        # PRL model


        a = (-np.sin(kx)**2 - np.sin(ky)**2)
             + 1 + (np.cos(kx) + np.cos(ky) + m - 3)**2)
        b = np.cos(kx) + np.cos(ky) + m - 3
        c = -2 * (np.sin(ky) - 1j * np.sin(kx)) * (np.cos(kx) + np.cos(ky) + m - 3)
        d = -2 * (np.sin(ky) - 1j * np.sin(kx))

        # Compose onsite and hopping matrices
        e = np.stack((np.stack((a, np.conj(c)), axis=-1),
                      np.stack((c, -a), axis=-1)), axis=-1)

        delta = np.stack((np.stack((b, np.zeros(ntotal)), axis=-1),
                          np.stack((d, -b), axis=-1)), axis=-1)  # this gives
                                                                 # [b d
                                                                 #  0 -b]

        diagonal = np.eye(2*self.N_slab)
        np.kron(ey[np.newaxis, :, :], e)

        up_diag = np.diag(np.ones(2*N_slab-1), 1)
        down_diag = up_diag.T



        # Construct Hamiltonian for all Nz sites from these blockes
        hh[:, 0:2, 0:2] = e
        for nz in range(0, N_slab - 1):
            hh[:, 2 * nz + 2: 2 * nz + 4, 2 * nz + 2: 2 * nz + 4] = e
            hh[:, 2 * nz: 2 * nz + 2, 2 * nz + 2: 2 * nz + 4] = (
                np.transpose(np.conj(delta), (0, 2, 1)))
            hh[:, 2 * nz + 2: 2 * nz + 4, 2 * nz: 2 * nz + 2] = delta
        return hh

    def diag(self):
        h = HopfInsulator.slab_hamiltonian(self, kx, ky)
        h = self.slab_hamiltonian(kx, ky)

# h = HopfInsulator(1, 2, 3, 16, 'inital')




#hh =  h.slab_hamiltonian()


ev, evec =

a = np.arange(30).reshape((2, 3, 5))
print(a**2)