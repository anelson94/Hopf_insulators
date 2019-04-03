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


def fitfunc(x, alpha, beta, gamma):
    """Function for fitting Wannier function"""
    # In logarithmic scale
    return np.log(np.exp(-beta * x) * np.power(x, alpha) * gamma)  #


# with open('WannierLoc.pickle', 'rb') as f:
#     [WFocc3d, WFval3d, Nx3d] = pickle.load(f)
#
with open('WannierLoc1d.pickle', 'rb') as f:
    [WFoccx, Nx1d] = pickle.load(f)

# with open('HybridWannierLoc.pickle', 'rb') as f:
#     [wf_hybr, n_xhybr] = pickle.load(f)


# with open('WannieFromBloch.pickle', 'rb') as f:
#     [Wannier, Nx, x_cell, NG] = pickle.load(f)
#
# Wannier = np.squeeze(Wannier)
# print(Wannier.shape)

WFoccx_abs = np.sqrt(np.sum(WFoccx * np.conj(WFoccx), axis=-1))
# WFoccz_abs = np.sqrt(np.sum(WFoccz * np.conj(WFoccz), axis=-1))
# WFoccxy_abs = np.sqrt(np.sum(WFoccxy * np.conj(WFoccxy), axis=-1))
# WFoccxyz_abs = np.sqrt(np.sum(WFoccxyz * np.conj(WFoccxyz), axis=-1))
# WFvalx_abs = np.sqrt(np.sum(WFvalx * np.conj(WFvalx), axis=-1))

# WFhybrid_abs = np.sqrt(np.sum(wf_hybr * np.conj(wf_hybr), axis=-1))


# Nx1d = n_xhybr
WF_forplot = np.real(WFoccx_abs)
# WFocc_max, coord_max = maxarray(WF_forplot)
# coord_max = coord_max * Nunitcell / Nx1d
coord = np.linspace(0, Nx1d - 1, Nx1d)
# WF_max_len = WFocc_max.size
print(WF_forplot)

# Fitting
xcoo = coord[1:15:2]  # _max
ycoo = np.log(WF_forplot[1:15:2])  # _max
popt, pcov = curve_fit(fitfunc, xcoo, ycoo)#,
                       # bounds=([-np.inf, -np.inf, -np.inf],
                       #         [np.inf, np.inf, np.inf]))

print(pcov)

matplotlib.rcParams.update({'font.size': 25})
print('Plot start')
# Check that WFs exponentially decay
fig = plt.figure(figsize=(10, 8))
ax = fig.add_axes([0.15, 0.15, 0.8, 0.8])
ax.set_xlabel('Unit cell number', size=25)  # $(100)$ direction
ax.set_ylabel('$|W|$', size=25)  # _{Hybrid}
ax.set_xticks([0, 5, 10, 15])
ax.set_xticklabels(('0', '5', '10', '15'), size=20)
ax.xaxis.set_minor_locator(plt.MultipleLocator(1))
ax.tick_params(which='both', labelsize=20, width=2)
ax.tick_params(axis='x', which='both', length=4)
plt.setp(ax.spines.values(), linewidth=2)
# plt.text(14, -2, '$a=1$ - lattice period')
plt.yscale('log')
# plt.xscale('log')
# plt.plot(coord_max, WFocc_max, 'k.')
plt.plot(coord[:16], WF_forplot[:16], 'k.', markersize=15)
plt.plot(coord[1:16], np.exp(fitfunc(coord[1:16], *popt)), 'r-', linewidth=4.0,
         label=r'fit: $\gamma x^{\alpha} e^{-\beta x}$, '
               r'with ''\n'
               r'$\alpha=$%5.3f,''\n'
               r'$\beta=$%5.3f,''\n'
               r'$\gamma=$%5.3f,''\n' % tuple(popt))
plt.legend()
plt.show()


# Draw the contours of Wannier orbitals
# mlab.figure()
# mlab.contour3d(np.abs(WF13d), contours=4, transparent=False)
# mlab.show()
#
# mlab.figure()
# mlab.contour3d(np.abs(WF23d), contours=6, transparent=False)
# mlab.show()
