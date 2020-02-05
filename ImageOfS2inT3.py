# -*- coding: utf-8 -*-
"""
Created on Mon Sep 17 10:03:13 2018

@author: aleksandra

Construct the images of different points at S2 (H1, H2, H3) in T3 (kx, ky, kz)
They will be the linked circles
"""


import numpy as np
from mayavi import mlab
import hopfham


def image(n, model, eps1=0.1, eps2=0.1):
    """Preimage at T^3 of a point n at S^2 for a map model"""
    hx = model['hx']
    hy = model['hy']
    hz = model['hz']
    nx = n[0]

    image = (np.isclose(model['hx'], n[0], atol=eps1, rtol=eps2)
             & np.isclose(model['hy'], n[1], atol=eps1, rtol=eps2)
             & np.isclose(model['hz'], n[2], atol=eps1, rtol=eps2)) * 1

    return image


def main():
    # Three points in S2 (3d vectors with unitary norms)
    n1 = np.array([1, 0, 0])
    n2 = np.array([0, 1, 0])
    n3 = np.array([0, 0, 1])

    nx = 101
    ny = 101
    nz = 101

    kx, ky, kz = hopfham.mesh_make(nx, ny, nz)

    model = hopfham.model_mrw_norm(kx, ky, kz)

    image1 = image(n1, model)
    image2 = image(n2, model)
    image3 = image(n3, model)

    # Plot the resulting preimage
    mlab.figure(bgcolor=(1, 1, 1))
    con1 = mlab.contour3d(image1, contours=2, color=(1, 153 / 255, 153 / 255),
                          transparent=False)
    mlab.axes(con1, xlabel='', ylabel='', zlabel='', color=(0, 0, 0))
    mlab.contour3d(image2, contours=2, color=(153 / 255, 1, 153 / 255),
                   transparent=False)
    mlab.contour3d(image3, contours=2, color=(153 / 255, 153 / 255, 1),
                   transparent=False)
    mlab.show()

    return model['hx'], model['hy'], model['hz']


if __name__ == '__main__':
    hx, hy, hz = main()

