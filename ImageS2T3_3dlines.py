# -*- coding: utf-8 -*-
"""
Created on Thu Sep 27 14:41:18 2018

@author: aleksandra
"""

# Draw preimages of S2 on T3 in a convenient way

import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as plt3d

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Draw cube
line = plt3d.art3d.Line3D([0, 0], [0, 0], [0, 1])
ax.add_line(line)
line = plt3d.art3d.Line3D([0, 0], [0, 1], [0, 0])
ax.add_line(line)
line = plt3d.art3d.Line3D([0, 1], [0, 0], [0, 0])
ax.add_line(line)
line = plt3d.art3d.Line3D([1, 1], [0, 0], [0, 1])
ax.add_line(line)
line = plt3d.art3d.Line3D([1, 1], [0, 1], [0, 0])
ax.add_line(line)
line = plt3d.art3d.Line3D([0, 0], [1, 1], [0, 1])
ax.add_line(line)
line = plt3d.art3d.Line3D([0, 1], [1, 1], [0, 0])
ax.add_line(line)
line = plt3d.art3d.Line3D([0, 0], [0, 1], [1, 1])
ax.add_line(line)
line = plt3d.art3d.Line3D([0, 1], [0, 0], [1, 1])
ax.add_line(line)
line = plt3d.art3d.Line3D([1, 1], [1, 1], [0, 1])
ax.add_line(line)
line = plt3d.art3d.Line3D([1, 1], [0, 1], [1, 1])
ax.add_line(line)
line = plt3d.art3d.Line3D([0, 1], [1, 1], [1, 1])
ax.add_line(line)

#Draw Images
line = plt3d.art3d.Line3D([1, 0.5], [0.15, 0.15], [0.5, 1], color = 'red')
ax.add_line(line)
line = plt3d.art3d.Line3D([1, 0.5], [0.65, 0.65], [0.5, 1], color = 'red')
ax.add_line(line)
line = plt3d.art3d.Line3D([0.8, 0.8], [0, 1], [0.7, 0.7], color = 'red')
ax.add_line(line)
line = plt3d.art3d.Line3D([0.5, 0], [0.15, 0.15], [0, 0.5], color = 'red')
ax.add_line(line)
line = plt3d.art3d.Line3D([0.5, 0], [0.65, 0.65], [0, 0.5], color = 'red')
ax.add_line(line)
line = plt3d.art3d.Line3D([0.3, 0.3], [0, 1], [0.2, 0.2], color = 'red')
ax.add_line(line)

line = plt3d.art3d.Line3D([0.35, 0.35], [1, 0.5], [0.5, 1], color = 'green')
ax.add_line(line)
line = plt3d.art3d.Line3D([0.85, 0.85], [1, 0.5], [0.5, 1], color = 'green')
ax.add_line(line)
line = plt3d.art3d.Line3D([0, 1], [0.7, 0.7], [0.8, 0.8], color = 'green')
ax.add_line(line)
line = plt3d.art3d.Line3D([0.35, 0.35], [0.5, 0], [0, 0.5], color = 'green')
ax.add_line(line)
line = plt3d.art3d.Line3D([0.85, 0.85], [0.5, 0], [0, 0.5], color = 'green')
ax.add_line(line)
line = plt3d.art3d.Line3D([0, 1], [0.2, 0.2], [0.3, 0.3], color = 'green')
ax.add_line(line)

plt.show()


######## Next step #############


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Draw cube
line = plt3d.art3d.Line3D([0, 0], [0, 0], [0, 1])
ax.add_line(line)
line = plt3d.art3d.Line3D([0, 0], [0, 1], [0, 0])
ax.add_line(line)
line = plt3d.art3d.Line3D([0, 1], [0, 0], [0, 0])
ax.add_line(line)
line = plt3d.art3d.Line3D([1, 1], [0, 0], [0, 1])
ax.add_line(line)
line = plt3d.art3d.Line3D([1, 1], [0, 1], [0, 0])
ax.add_line(line)
line = plt3d.art3d.Line3D([0, 0], [1, 1], [0, 1])
ax.add_line(line)
line = plt3d.art3d.Line3D([0, 1], [1, 1], [0, 0])
ax.add_line(line)
line = plt3d.art3d.Line3D([0, 0], [0, 1], [1, 1])
ax.add_line(line)
line = plt3d.art3d.Line3D([0, 1], [0, 0], [1, 1])
ax.add_line(line)
line = plt3d.art3d.Line3D([1, 1], [1, 1], [0, 1])
ax.add_line(line)
line = plt3d.art3d.Line3D([1, 1], [0, 1], [1, 1])
ax.add_line(line)
line = plt3d.art3d.Line3D([0, 1], [1, 1], [1, 1])
ax.add_line(line)

#Draw Images
line = plt3d.art3d.Line3D([0.8, 0.3], [0.15, 0.15], [0.7, 1.2], color = 'red')
ax.add_line(line)
line = plt3d.art3d.Line3D([0.8, 0.3], [0.65, 0.65], [0.7, 1.2], color = 'red')
ax.add_line(line)
line = plt3d.art3d.Line3D([0.8, 0.8], [0, 1], [0.7, 0.7], color = 'red')
ax.add_line(line)
line = plt3d.art3d.Line3D([0.3, 0.3], [0.15, 0.65], [1.2, 1.2], color = 'red')
ax.add_line(line)
line = plt3d.art3d.Line3D([0.3, -0.2], [0.15, 0.15], [0.2, 0.7], color = 'red')
ax.add_line(line)
line = plt3d.art3d.Line3D([0.3, -0.2], [0.65, 0.65], [0.2, 0.7], color = 'red')
ax.add_line(line)
line = plt3d.art3d.Line3D([0.3, 0.3], [0, 1], [0.2, 0.2], color = 'red')
ax.add_line(line)
line = plt3d.art3d.Line3D([-0.2, -0.2], [0.15, 0.65], [0.7, 0.7], color = 'red')
ax.add_line(line)

line = plt3d.art3d.Line3D([0.35, 0.35], [1.2, 0.2], [0.3, 1.3], color = 'green')
ax.add_line(line)
line = plt3d.art3d.Line3D([0.85, 0.85], [1.2, 0.2], [0.3, 1.3], color = 'green')
ax.add_line(line)
line = plt3d.art3d.Line3D([0, 1], [0.7, 0.7], [0.8, 0.8], color = 'green')
ax.add_line(line)
line = plt3d.art3d.Line3D([0, 1], [0.2, 0.2], [0.3, 0.3], color = 'green')
ax.add_line(line)

ax.scatter([0.8, 0.8], [0, 1], [0.7, 0.7], color = 'red', marker = 's', alpha = 1)
ax.scatter([0.3, 0.3], [0, 1], [0.2, 0.2], color = 'red', marker = 'o', alpha = 1)
ax.scatter([0.8, -0.2], [0.15, 0.15], [0.7, 0.7], color = 'red', marker = '^', s = 30, alpha = 1)
ax.scatter([0.8, -0.2], [0.65, 0.65], [0.7, 0.7], color = 'red', marker = '^', s = 30, alpha = 1)
ax.scatter([0.3, 0.3], [0.15, 0.15], [0.2, 1.2], color = 'red', marker = 'X', s = 30, alpha = 1)
ax.scatter([0.3, 0.3], [0.65, 0.65], [0.2, 1.2], color = 'red', marker = 'X', s = 30, alpha = 1)

ax.scatter([0, 1], [0.7, 0.7], [0.8, 0.8], color = 'green', marker = 's', alpha = 1)
ax.scatter([0, 1], [0.2, 0.2], [0.3, 0.3], color = 'green', marker = 'o', alpha = 1)
ax.scatter([0.35, 0.35, 0.35], [1.2, 0.2, 0.2], [0.3, 1.3, 0.3], color = 'green', marker = '^', s = 30, alpha = 1)
ax.scatter([0.85, 0.85, 0.85], [1.2, 0.2, 0.2], [0.3, 1.3, 0.3], color = 'green', marker = 'X', s = 30, alpha = 1)

plt.show()
