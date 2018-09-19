# -*- coding: utf-8 -*-
"""
Created on Thu Sep  6 18:51:09 2018

@author: aleksandra
"""

# Draw WFs obtained as a Fourier transform of analytical Bloch solutions.

import pickle
import numpy as np
import matplotlib.pyplot as plt
from mayavi import mlab

with open('WannierLoc.pickle', 'rb') as f:
    [WF1, WF2] = pickle.load(f)
    
with open('WannierLocx.pickle', 'rb') as f:
    [WFx1, WFx2] = pickle.load(f)    

# Check that WFs exponentially decay
plt.figure()
plt.plot(np.linspace(0,200, 200), np.abs(WFx1))
plt.show()  

plt.figure()
plt.plot(np.linspace(0,200, 200), np.abs(WFx2))
plt.show()   

  
#plt.figure
#plt.imshow(np.abs(WF1[:, :, 0]))
#plt.colorbar()
#plt.show()

# Draw the contours of Wannier orbitals
mlab.figure()
mlab.contour3d(np.abs(WF1), contours=4, transparent=False)
mlab.show()

mlab.figure()
mlab.contour3d(np.abs(WF2), contours=6, transparent=False)
mlab.show()
