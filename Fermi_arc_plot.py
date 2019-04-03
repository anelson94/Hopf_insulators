"""
 Created by alexandra at 02.04.19 16:10

 Read from the file arc.dat_l created by Wannier tools and plot the picture
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from math import pi

# with open('arc_h05.dat_l','rb') as f:
data = np.loadtxt('arc_h0.dat_l')

kx = np.reshape(data[:, 0], (50, 50))
ky = np.reshape(data[:, 1], (50, 50))
DOS1 = np.reshape(data[:, 2], (50, 50))

# Change to [-pi,pi]
DOS = np.concatenate(
    (np.concatenate((DOS1[25:49, 25:49], DOS1[0:24, 25:49]), axis=0),
     np.concatenate((DOS1[25:49, 0:24], DOS1[0:24, 0:24]), axis=0)), axis=1)

cmapDOS = colors.LinearSegmentedColormap.from_list(
    "", [(25/255, 78/255, 255/255), 'white', 'red'])

fs = 10
fss = 8
fig = plt.figure(figsize=(1.6, 1.35))
ax = fig.add_axes([0.23, 0.22, 0.64, 0.79])
ax.set_xlabel('$k_x$', size=fs)
ax.yaxis.set_label_coords(-0.17, 0.5)
ax.set_ylabel('$k_y$', size=fs)
ax.xaxis.set_label_coords(0.5, -0.23)
ax.set_xticks([-pi, 0, pi])
ax.set_xticklabels(('$-\pi$', '$0$', '$\pi$'), size=fss)
ax.set_yticks([-pi, 0, pi])
ax.set_yticklabels(('$-\pi$', '$0$', '$\pi$'), size=fss)
plt.xlim(-pi, pi)
plt.ylim(-pi, pi)
ax.tick_params(width=1.5)
for axis in ['top','bottom','left','right']:
    ax.spines[axis].set_linewidth(1.5)
DOSplot = plt.imshow(DOS, extent=[-pi, pi, -pi, pi], cmap=cmapDOS)
cbar = fig.colorbar(DOSplot, fraction=0.046, pad=0.04)
cbar.ax.tick_params(labelsize=fss, width=1.5, rotation=90)

plt.savefig('FA_h0t1_actsize.png', bbox_inches=None)
