#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 29 08:39:01 2018

@author: jon
"""

import cmath
import math
import numpy as np
import matplotlib.pyplot as plt

from pynufft import NUFFT_cpu
import scipy.io as sio

y = sio.loadmat("/home/jon/Desktop/export_mat/y.mat")['y'][0,0].reshape(307780)
p = np.asarray(sio.loadmat("/home/jon/Desktop/export_mat/p.mat")["p"])
dirty = np.asarray(sio.loadmat("/home/jon/Desktop/export_mat/dirty.mat")["dirty"])


NufftObj = NUFFT_cpu()
Nd = (512, 512)  # image size
Kd = (2048, 2048)  # k-space size
Jd = (8, 8)  # interpolation size

NufftObj.plan(p, Nd, Kd, Jd)
res = NufftObj.adjoint(y)
#res = NufftObj.solve(y, solver='cg',maxiter=1000)


plt.imshow(np.real(res[256:640,256:640]))