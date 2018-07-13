# -*- coding: utf-8 -*-
"""
Created on Wed Jul 11 18:59:34 2018

@author: aleksandra
"""

import numpy as np
from math import pi
import pickle
import matplotlib.pyplot as plt

with open('Hopfinvariant.pickle', 'rb') as f:
    underHopf = pickle.load(f)
    
plt.imshow(underHopf[:, :, 50], cmap='RdBu')
plt.colorbar()
plt.show()