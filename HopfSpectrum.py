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

k = np.linspace(0, 2 * pi, 100)
kx = 2 * pi
ky, kz = np.meshgrid(k, k, indexing='ij')

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
deltaE = 0.5

Ecrossings = (E1 < deltaE)

fs = 15
fig, ax = plt.subplots()
# plt.plot(k, E1, 'k', k, -E1, 'k')
ax.imshow(Ecrossings)
ax.set_xlabel('$k_x$', fontsize=fs)
ax.set_ylabel('$k_z$', fontsize=fs)
plt.show()
