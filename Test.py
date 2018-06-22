# -*- coding: utf-8 -*-
"""
Created on Thu Jun 21 17:32:38 2018

@author: aleksandra
"""

# Test program

import numpy as np
#A = np.array([[1,2],[3,4]])
#
#print(A + 1)
#print(np.power(A + 1, 2))
#
#x = [1,2]
#y = [1,2]
#z = [3,4]
#
#[xx, yy, zz] = np.meshgrid(x, y, z)
#print('xx=',xx, 'yy=', yy, 'zz=', zz, sep='\n')

sigmax = np.array([[0, 1], [1, 0]])
print(sigmax.shape)
sigmax = sigmax[np.newaxis,:,:]
print(sigmax.shape)
sigmastack = np.tile(sigmax, (3,1,1))
print('sigma=', sigmastack)
print(sigmastack.shape)

E, u = np.linalg.eigh(sigmastack)

print(E)
print('Edim=', E.shape, ' udim=', u.shape)


x = np.array([1,3])
z = np.concatenate((x[:, np.newaxis], x[:, np.newaxis]), axis = 1)
print(z)
print(z.shape)
y = np.array([3,4])
x = x[:, np.newaxis]
y = y[np.newaxis, :]
print(x + y)
print(x.shape)
print(y.shape)
print((x+y).shape)