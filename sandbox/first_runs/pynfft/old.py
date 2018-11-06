#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 28 17:27:45 2018

@author: jon
"""

# -*- coding: utf-8 -*-


import cmath
import math
import numpy as np
from pynfft import NFFT
import matplotlib.pyplot as plt

def inverseFT(img, X, u, v):
    N = img.shape[0]/2
    for x in range (-N, N):
        for y in range(-N, N):
            img[x-N, y-N] = img[x-N,y-N] + X * cmath.exp(-2j*math.pi * (u*x + v*y))
           


dim = [32,32]
img = np.zeros(dim, dtype=np.complex128)

#inverseFT(img, 3, 0.05,0.013)
inverseFT(img, 2.5, 0.038,0.046)

plt.imshow(np.real(img))
print(img[0,0])





plan = NFFT(dim, 6)

plan.precompute()
infft = Solver(plan)
plan.x = np.asarray([0.5, 0.9,0.8,0.6, 0.1,0.1, 0.012,0.1, 0.245,0.102, 0.4,0.2])
#infft.w = w
infft.y = np.asarray([3,1, 0.5, 1.4, 0.6, 0.8])
infft.f_hat_iter = np.zeros(dim)
infft.before_loop()

niter = 2000 # set number of iterations to 10
for iiter in range(niter):
    infft.loop_one_step()

res = infft.f_hat_iter

plt.imshow(np.real(res))
print(res[0,0])