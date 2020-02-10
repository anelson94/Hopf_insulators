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


def color_list():
    # Define a list with colors for preimage plotting
    red = (1, 153 / 255, 153 / 255)
    blue = (153 / 255, 153 / 255, 1)
    green = (153 / 255, 1, 153 / 255)
    brown = (174 / 255, 129 / 255, 58 / 255)
    lightblue = (58 / 255, 138 / 255, 200 / 255)
    return (red, green, blue, brown, lightblue)


def preimage(n, model, eps1=0.1, eps2=0.1):
    """Preimage at T^3 of a point n at S^2 for a map model"""

    im = (np.isclose(model['hx'], n[0], atol=eps1, rtol=eps2)
          & np.isclose(model['hy'], n[1], atol=eps1, rtol=eps2)
          & np.isclose(model['hz'], n[2], atol=eps1, rtol=eps2)) * 1

    return im


def preimage_plot(model, *n_list, nx=101, ny=101, nz=101, **kwargs):
    """Plot preimages of n_list points"""
    kx, ky, kz = hopfham.mesh_make(nx, ny, nz)
    if callable(model):
        model = model(kx, ky, kz)

    mlab.figure(bgcolor=(1, 1, 1))
    i_color = 0
    colors = color_list()
    for n in n_list:
        im = preimage(n, model, **kwargs)
        con = mlab.contour3d(im, contours=2,
                             color=colors[i_color % len(colors)],
                             transparent=False)
        i_color += 1
    mlab.axes(con, xlabel='', ylabel='', zlabel='', color=(0, 0, 0))
    mlab.show()


def image(k, model):
    """Find """


def main():
    # Three points in S2 (3d vectors with unitary norms)
    n1 = np.array([1, 0, 0])
    n2 = np.array([0, 1, 0])
    n3 = np.array([0, 0, 1])
    # n4 = np.array([-1, 0, 0])
    # n5 = np.array([0, -1, 0])

    preimage_plot(hopfham.model_mrw_norm, n1, n2, n3)



if __name__ == '__main__':
    main()

