#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 29 09:32:02 2018

@author: jon
"""

import cmath
import math
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio

from pynfft import NFFT, Solver

#y= sio.loadmat("/home/jon/Desktop/export_mat/y_raw.mat")["y_I"].reshape(307780)
y = sio.loadmat("/home/jon/Desktop/export_mat/y.mat")['y'][0,0].reshape(307780)
p = np.asarray(sio.loadmat("/home/jon/Desktop/export_mat/p.mat")["p"])
dirty = np.asarray(sio.loadmat("/home/jon/Desktop/export_mat/dirty.mat")["dirty"])

#p_shaped = np.reshape( (1/6.0)*p, (p.shape[0]*p.shape[1]))
#dim= (1024,1024)
p_shaped = np.reshape( (1/4.0)*p, (p.shape[0]*p.shape[1]))
dim= (256,256)
plan = NFFT(dim, y.size)

plan.x = p_shaped
plan.precompute()

plan.f = y
res = plan.adjoint()
plt.imshow(np.real(res))
print(np.max(np.real(res)))


'''
infft = Solver(plan)
infft.y = y
infft.f_hat_iter = np.zeros(dim, dtype=np.complex128)
infft.before_loop()

niter = 100 # set number of iterations to 10
for iiter in range(niter):
    print("alive"+str(iiter))
    infft.loop_one_step()
    if(np.all(infft.r_iter < 0.001)):
        break
    
res = infft.f_hat_iter
plt.imshow(np.real(res))
'''