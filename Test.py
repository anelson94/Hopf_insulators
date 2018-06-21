# -*- coding: utf-8 -*-
"""
Created on Thu Jun 21 17:32:38 2018

@author: aleksandra
"""

# Test program

import numpy as np
A = np.array([[1,2],[3,4]])

print(A + 1)
print(np.power(A + 1, 2))

x = [1,2]
y = [1,2]
z = [3,4]

[xx, yy, zz] = np.meshgrid(x, y, z)
print('xx=',xx, 'yy=', yy, 'zz=', zz, sep='\n')