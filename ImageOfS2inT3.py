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


def preimage(n, model, eps1, eps2):
    """Preimage at T^3 of a point n at S^2 for a map model"""

    im = (np.isclose(model['hx'], n[0], atol=eps1, rtol=eps2)
          & np.isclose(model['hy'], n[1], atol=eps1, rtol=eps2)
          & np.isclose(model['hz'], n[2], atol=eps1, rtol=eps2)) * 1

    return im


def preimage_plot(model, *n_list, nx=101, ny=101, nz=101, eps1=0.1, eps2=0.1,
                  **kwargs):
    """Plot preimages of n_list points"""
    kx, ky, kz = hopfham.mesh_make(nx, ny, nz)
    if (model == hopfham.model_edgeconst
        or model == hopfham.model_edgeconst_maps
            or model == hopfham.model_edgeconst_maps_rotated):

        kx = kx[2:-2, 2:-2, 2:-2]
        ky = ky[2:-2, 2:-2, 2:-2]
        kz = kz[2:-2, 2:-2, 2:-2]
    if callable(model):
        model = model(kx, ky, kz, **kwargs)

    mlab.figure(bgcolor=(1, 1, 1), fgcolor=(0., 0., 0.))
    i_color = 0
    colors = color_list()
    for n in n_list:
        im = preimage(n, model, eps1, eps2)
        coords = np.where(im == 1)
        print(len(coords))
        # con = mlab.contour3d(kx, ky, kz, im, contours=2,
        #                      color=colors[i_color % len(colors)],
        #                      transparent=False)
        x_coord = coords[0]*2*pi/(nx-1) - pi
        y_coord = coords[1]*2*pi/(ny-1) - pi
        z_coord = coords[2]*2*pi/(nz-1) - pi

        con = mlab.points3d(x_coord, y_coord, z_coord, scale_factor=.05,
                            color=colors[i_color % len(colors)])
        i_color += 1
    points = np.array([-pi, pi])
    pointsx, pointsy, pointsz = np.meshgrid(points, points, points,
                                            indexing='ij')
    pointsx = np.ravel(pointsx)
    pointsy = np.ravel(pointsy)
    pointsz = np.ravel(pointsz)
    mlab.points3d(pointsx, pointsy, pointsz, scale_factor=.25)
    # mlab.axes(con, xlabel='x', ylabel='y', zlabel='z', color=(0, 0, 0),
    #           extend=[-pi, pi, -pi, pi, -pi, pi])
    mlab.show()


def image_plot(kx, ky, kz, model, **kwargs):
    """Find """
    if callable(model):
        model = model(kx, ky, kz, **kwargs)

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
        # n = np.array([-0.3696628, 0, -0.92916596])
        # n = np.array([-0.459850, 0.136869, -0.8773850])
        a = 3*pi/4
        n = np.array([np.sin(a), 0, -np.cos(a)])
        n1 = np.array([0, 1, 0])
        # n1 = np.array([np.sqrt(3)/2, 1/2, 0])
        # n = np.array([-0.45985, 0.136869, -0.877385])
        # n2 = np.array([0, 0, -1])
        # n3 = np.array([0, 0, 1])
        # n4 = np.array([1, 0, 0])
        # n5 = np.array([np.sqrt(2)/2, 0, np.sqrt(2)/2])
        # n4 = np.array([-1, 0, 0])
        # n5 = np.array([0, -1, 0])

        # List of models
        # hopfham.model_mrw
        # hopfham.model_mrw_norm
        # hopfham.model_edgeconst
        # hopfham.model_mrw_maps
        # hopfham.model_mrw_maps_rotated
        # hopfham.model_edgeconst_maps
        # hopfham.model_edgeconst_maps_rotated

        # MRW model args
        # model_args = {'m': 1}
        # No args needed for model
        # model_args = {}
        # Rotated MRW model from maps args
        # model_args = {'m': 1, 'alpha': 0}
        # Rotation angle in args
        model_args = {'alpha': -a/2}

        preimage_plot(hopfham.model_mrw_maps_rotated, n, n1,
                      nx=201, ny=201, nz=201,
                      eps1=0.05, eps2=0.05, **model_args)

    if imageplot == 1:
        # kx = np.ones(100) * pi
        # ky = np.ones(100) * 2.1
        # kz = np.linspace(-pi, pi, 100)
        model_args = {'alpha': pi/8}
        # model_args = {}
        kx = pi
        ky = 0
        kz = 0
        hx, hy, hz = image_plot(kx, ky, kz,
                                hopfham.model_edgeconst_maps_rotated,
                                **model_args)
        print(hx, hy, hz)


if __name__ == '__main__':
    main()

