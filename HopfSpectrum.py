"""
 Created by alexandra at 14.01.19 17:31

 Calculate the energy spectrum and plot it crossings with Fermi level
 at critical points
"""

import numpy as np
from math import pi
import matplotlib.pyplot as plt

h = -3
t = 1
Asquare = 1

k = np.linspace(-pi, pi, 100)
kx = 0
ky = 0
kz = k
# ky, kz = np.meshgrid(k, k, indexing='ij')

Hx = 2 * (np.sin(kx) * np.sin(kz)
          + t * np.sin(ky) * (np.cos(kx) + np.cos(ky) + np.cos(kz) + h))
Hy = 2 * (t * np.sin(ky) * np.sin(kz)
          - np.sin(kx) * (np.cos(kx) + np.cos(ky) + np.cos(kz) + h))
Hz = (np.power(np.sin(kx), 2) + np.power(np.sin(ky), 2)
      - np.power(np.sin(kz), 2)
      - np.power(np.cos(kx) + np.cos(ky) + np.cos(kz) + h, 2)
      + Asquare)
# + Asquare * np.sin(kz) + Asquare * np.sin(ky) + Asquare * np.sin(kx))

E1 = np.sqrt(np.power(Hx, 2) + np.power(Hy, 2) + np.power(Hz, 2))
fig = plt.figure(figsize=(8, 6))
ax = fig.add_axes([0.13, 0.11, 0.83, 0.85])
fs = 33
fss = 25
ls = 3
lss=2.5
ax.tick_params(labelsize=fss, width=lss)
ax.tick_params(axis='x', labelsize=fss)
for axis in ['top', 'bottom', 'left', 'right']:
    ax.spines[axis].set_linewidth(lss)
ax.xaxis.set_label_coords(0.8, -0.02)
ax.yaxis.set_label_coords(-0.08, 0.5)
ax.set_xlabel(r'$k_z$', fontsize=fs)
ax.set_ylabel(r'Energy', fontsize=fs)
ax.set_xlim(-pi, pi)
ax.set_xticks([-pi, 0, pi])
# ax.set_yticks([-2, 0, 2])
ax.set_xticklabels([r'$-\pi$', r'0', r'$\pi$'])
# ax.set_ylim(max(E1), -max(E1))
ax.plot(k, E1, 'r', k, -E1, 'b', linewidth=ls)
# plt.show()
plt.savefig('Images/CritSpectra/Spectra_crit_h-3t1A1.png',
            bbox_inches=None)

#
# deltaE = 0.5
#
# Ecrossings = (E1 < deltaE)
#
# fs = 15
# fig, ax = plt.subplots()
# # plt.plot(k, E1, 'k', k, -E1, 'k')
# ax.imshow(Ecrossings)
# ax.set_xlabel('$k_x$', fontsize=fs)
# ax.set_ylabel('$k_z$', fontsize=fs)
# plt.show()
