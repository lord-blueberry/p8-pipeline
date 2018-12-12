#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 12 20:30:46 2018

@author: jon
"""

data = load_debug()
nuft = nfft(data)

import numpy.fft as fftnumpy

import matplotlib.pyplot as plt
from algorithms.CoordinateDescent2 import CoordinateDescent1 as CD
from algorithms.CoordinateDescent2 import calc_cache
from algorithms.CoordinateDescent2 import fourier_starlets
from algorithms.CoordinateDescent2 import equi_starlets
from algorithms.CoordinateDescent2 import _magnitude
from algorithms.CoordinateDescent2 import _shrink
from algorithms.CoordinateDescent2 import calc_residual
from algorithms.CoordinateDescent2 import _nfft_approximation






tmp = np.asarray([1.0 / 16.0, 1.0 / 4.0, 3.0 / 8.0, 1.0 / 4.0, 1.0 / 16.0])
row = np.asmatrix([tmp])
bspline =(np.dot(np.transpose(row), row))

tmp = np.zeros(data.imsize, dtype=np.complex128)
def insert_spaced(mat, kernel, J):
    disp = 2**J
    for xi in range(0, kernel.shape[0]):
        for yi in range(0, kernel.shape[1]):
            mat[xi * disp, yi * disp] = kernel[xi, yi]
    roll = -2**(J+1)
    mat = np.roll(mat, roll, axis=0)
    mat = np.roll(mat, roll, axis=1)
    return mat
new_base = insert_spaced(tmp.copy(), bspline, 0)
bfft = fftnumpy.fft2(new_base)
tmp[32,32] = 5
tmp = np.real(tmp)


tmp_f = fftnumpy.fft2(tmp)
tmp_conf = np.real(fftnumpy.ifft2(tmp_f*bfft))
plt.imshow(tmp_conf)
