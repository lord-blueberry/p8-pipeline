#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 09:32:26 2018

@author: jon
"""


import cmath
import math
import numpy as np
import matplotlib.pyplot as plt

from pynufft import NUFFT_cpu

def expectedIFT(img, X, u, v):
    N = img.shape[0]//2
    for x in range (-N, N):
        for y in range(-N, N):
            img[x-N, y-N] = img[x-N,y-N] + X * cmath.exp(-2j*math.pi * (u*x + v*y))
           
u0 = 0.05
v0 = 0.013
u1 = 0.0018
v1 = 0.046

img = np.zeros([32,32], dtype=np.complex128)
expectedIFT(img, 3, u0, v0)
expectedIFT(img, 2.5, u1, v1)

plt.imshow(np.real(img))
print(np.max(np.real(img)))


NufftObj = NUFFT_cpu()
Nd = (32, 32)  # image size
Kd = (64, 64)  # k-space size
Jd = (2, 2)  # interpolation size
om = [
      [u0, v0],
      [u1,v1]]

NufftObj.plan(np.asarray(om), Nd, Kd, Jd)
img2 = NufftObj.adjoint([3, 2.5])
plt.imshow(np.real(img2))
print(np.max(np.real(img2)))

