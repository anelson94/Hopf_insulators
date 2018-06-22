# -*- coding: utf-8 -*-
"""
Created on Fri Jun 22 14:05:25 2018

@author: Aleksandra
"""

import numpy as np
from math import pi
import pickle

# Import parameters for Hopf Hamiltonian from file params.py
import params

t = params.t
h = params.h

Nx = params.Nx
Ny = params.Ny
Nz = params.Nz

with open('Hopfeigen.pickle', 'rb') as f:
    [E, u] = pickle.load(f)
    