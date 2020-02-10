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
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from math import pi


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
    mlab.axes(con, xlabel='x', ylabel='y', zlabel='z', color=(0, 0, 0))
    mlab.show()


def image_plot(kx, ky, kz, model):
    """Find """
    if callable(model):
        model = model(kx, ky, kz)

    # Plot images of (kx, ky, kz)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(model['hx'], model['hy'], model['hz'])

    plt.show()

    return model['hx'], model['hy'], model['hz']


def main():
    preimageplot = 1
    imageplot = 0

    if preimageplot == 1:
        # Three points in S2 (3d vectors with unitary norms)
        n = np.array([-0.3696628, 0, -0.92916596])
        # n = np.array([-0.459850, 0.136869, -0.8773850])
        n1 = np.array([1, 0, 0])
        # n2 = np.array([0, 1, 0])
        # n3 = np.array([0, 0, 1])
        # n4 = np.array([-1, 0, 0])
        # n5 = np.array([0, -1, 0])

        preimage_plot(hopfham.model_mrw_norm, n, n1, nx=401, ny=401, nz=401,
                      eps1=0.05, eps2=0.05)

    if imageplot == 1:
        # kx = np.ones(100) * pi
        # ky = np.ones(100) * 2.1
        # kz = np.linspace(-pi, pi, 100)
        kx = pi
        ky = 2.1
        kz = -1.3
        hx, hy, hz = image_plot(kx, ky, kz, hopfham.model_mrw_norm)
        print(hx, hy, hz)


if __name__ == '__main__':
    main()

