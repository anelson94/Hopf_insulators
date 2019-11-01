"""
 Created by alexandra at 02.04.19 16:10

 Read from the file arc.dat_l created by Wannier tools and plot the picture
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from math import pi

# with open('arc_h05.dat_l','rb') as f:
data = np.loadtxt('arc_h05.dat_l')

kx = np.reshape(data[:, 0], (50, 50))
ky = np.reshape(data[:, 1], (50, 50))
DOS1 = np.reshape(data[:, 2], (50, 50))

# Change to [-pi,pi]
DOS = np.concatenate(
    (np.concatenate((DOS1[25:49, 25:49], DOS1[0:24, 25:49]), axis=0),
     np.concatenate((DOS1[25:49, 0:24], DOS1[0:24, 0:24]), axis=0)), axis=1)

cmapDOS = colors.LinearSegmentedColormap.from_list(
    "", [(25/255, 78/255, 255/255), 'white', 'red'])

# Sizes for paper
fs = 35
fss = 30
ls = 3
lss = 2.5
fig = plt.figure(figsize=(8.4, 7.2))
ax = fig.add_axes([0.1, 0.08, 0.8, 0.9])
ax.yaxis.set_label_coords(-0.05, 0.7)
ax.xaxis.set_label_coords(0.85, -0.02)

# Sizes for poster
# fs = 35
# fss = 30
# fig = plt.figure(figsize=(9.5, 8))
# ax = fig.add_axes([0.07, 0.1, 0.8, 0.85])
# ax.yaxis.set_label_coords(-0.05, 0.85)
# ax.xaxis.set_label_coords(0.9, -0.02)
ax.set_xlabel('$k_x$', size=fs)
ax.set_ylabel('$k_y$', size=fs, rotation=0)

ax.set_xticks([-pi, 0, pi])
ax.set_xticklabels(('$-\pi$', '$0$', '$\pi$'), size=fss)
ax.set_yticks([-pi, 0, pi])
ax.set_yticklabels(('$-\pi$', '$0$', '$\pi$'), size=fss)
plt.xlim(-pi, pi)
plt.ylim(-pi, pi)
ax.tick_params(width=lss)
DOSplot = plt.imshow(DOS, extent=[-pi, pi, -pi, pi], cmap=cmapDOS)
cbar = fig.colorbar(DOSplot, ticks=[-8, -4, 0, 4], fraction=0.046, pad=0.04)
cbar.ax.tick_params(labelsize=fss, width=lss, rotation=0)
cbar.outline.set_linewidth(lss)
for axis in ['top', 'bottom', 'left', 'right']:
    ax.spines[axis].set_linewidth(lss)



plt.savefig('Images/FermiArc_fromWT/FA_h05t1.png', bbox_inches=None)
# plt.show()
