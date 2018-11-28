# -*- coding: utf-8 -*-
"""
Created on Thu Sep  6 18:51:09 2018

@author: aleksandra

Draw WFs obtained as a Fourier transform of analytical Bloch solutions.
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import math
from scipy.optimize import curve_fit
# from mayavi import mlab


def maxarray(wf_massive):
    Nwf = wf_massive.size
    wf_maxarray = np.zeros(Nwf)
    x_max = np.zeros(Nwf)
    n_max = 0
    wf_max = 0
    for idx in range(0, Nwf):
        if wf_massive[idx] > wf_max:
            wf_max = wf_massive[idx]
        elif wf_massive[idx - 2] < wf_max:
            wf_maxarray[n_max] = wf_max
            x_max[n_max] = idx
            n_max += 1
            wf_max = 0
        else:
            wf_max = wf_massive[idx]
    wf_maxarray = np.trim_zeros(wf_maxarray)
    x_max = np.trim_zeros(x_max)
    return wf_maxarray, x_max


def fitfunc(x, alpha, beta, gamma, delta): # , epsilon
    """Function for fitting Wannier function"""
    # return np.log(np.power(x, alpha) * np.exp(-beta * x) + gamma)
    # In logarithmic scale
    return np.log(np.exp(-beta * x) * np.power(x, alpha) * gamma) #
    # Power law with logarithmic corrections in log scale
    # return np.log(np.power(x, alpha) + beta * np.log(x) + gamma)

# with open('WannierLoc.pickle', 'rb') as f:
#     [WFocc3d, WFval3d, Nx3d] = pickle.load(f)
#
# with open('WannierLoc1d.pickle', 'rb') as f:
#     [WFoccx, WFvalx, WFoccy, WFvaly, WFoccz, WFvalz,
#      WFoccxy, WFvalxy, WFoccxyz, WFvalxyz,
#      Nx1d, Nunitcell] = pickle.load(f)

# with open('HybridWannierLoc.pickle', 'rb') as f:
#     [wf_hybr, n_xhybr, n_hybr_uc] = pickle.load(f)


with open('WannieFromBloch.pickle', 'rb') as f:
    [Wannier, Nx, x_cell, NG] = pickle.load(f)

Wannier = np.squeeze(Wannier)
print(Wannier.shape)

# WFoccx_abs = np.sqrt(np.sum(WFoccx * np.conj(WFoccx), axis=-1))
# WFoccz_abs = np.sqrt(np.sum(WFoccz * np.conj(WFoccz), axis=-1))
# WFoccxy_abs = np.sqrt(np.sum(WFoccxy * np.conj(WFoccxy), axis=-1))
# WFoccxyz_abs = np.sqrt(np.sum(WFoccxyz * np.conj(WFoccxyz), axis=-1))
# WFvalx_abs = np.sqrt(np.sum(WFvalx * np.conj(WFvalx), axis=-1))

# WFhybrid_abs = np.sqrt(np.sum(wf_hybr * np.conj(wf_hybr), axis=-1))
# Nx1d = n_xhybr
# Nunitcell = n_hybr_uc
# WF_forplot = np.real(WFhybrid_abs)

Nx1d = Nx
Nunitcell = x_cell
WF_forplot = np.abs(Wannier)
# WFocc_max, coord_max = maxarray(WF_forplot)
# coord_max = coord_max * Nunitcell / Nx1d
coord = np.linspace(0, Nx1d, Nx1d) * Nunitcell / Nx1d
# WF_max_len = WFocc_max.size

# Fitting
# xcoo = coord_max # [10:]
# ycoo = np.log(WFocc_max) # [10:])
# popt, pcov = curve_fit(fitfunc, xcoo, ycoo)#,
#                        # bounds=([-np.inf, -np.inf, -np.inf],
#                        #         [np.inf, np.inf, np.inf]))

# print(pcov)

matplotlib.rcParams.update({'font.size': 13})

# Check that WFs exponentially decay
plt.figure(figsize=(8, 6))
plt.xlabel('$(100)$ direction', size=15)
plt.ylabel('$|W_{Hybrid}|$', size=15)
# plt.text(14, -2, '$a=1$ - lattice period')
plt.yscale('log')
plt.xscale('log')
# plt.plot(coord_max, WFocc_max, 'k.')
plt.plot(coord, WF_forplot)
# plt.plot(coord, np.exp(fitfunc(coord, *popt)), 'r-',
#          label=r'fit: $\gamma x^{\alpha} e^{-\beta x} + \delta$, '
#                r'with ''\n'
#                r'$\alpha=$%5.3f,''\n'
#                r'$\beta=$%5.3f,''\n'
#                r'$\gamma=$%5.3f,''\n'
#                # r'$\delta=$%5.3f,''\n'
#                r'$\delta=$%5.3f' % tuple(popt))
# plt.legend()
plt.show()  


# Draw the contours of Wannier orbitals
# mlab.figure()
# mlab.contour3d(np.abs(WF13d), contours=4, transparent=False)
# mlab.show()
#
# mlab.figure()
# mlab.contour3d(np.abs(WF23d), contours=6, transparent=False)
# mlab.show()