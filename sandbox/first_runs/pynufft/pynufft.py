# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import cmath
import math
import numpy as np
import matplotlib.pyplot as plt

from pynufft.nufft import NUFFT_cpu

def inverseFT(img, vis, u, v):
    for x in range (0,img.shape[0]):
        for y in range(0, img.shape[1]):
            img[x,y] = img[x,y] + vis * cmath.exp(2j*math.pi * (u*((x-27)/1.0) + v*((y-27)/1.0)))

def toComplex(amp, phase):
    x = amp * math.cos(phase)
    y = amp * math.sin(phase)
    return complex(x, y) 

def MHZToWavelength(x):
    return 299792458.0 / x / 1000000

cell = math.radians(8/3600.0)
wave_length =  MHZToWavelength(1122.0)
u0 = (-cell)*(-112.15 / wave_length)
v0 = (cell)*(576.66 / wave_length)
vis0 = toComplex(47.753, math.radians(173.2))


u1 = (-cell)*(71.50 / wave_length)
v1 = (cell)*(-368.67 / wave_length)
vis1 = toComplex(53.456, math.radians(173.2))
dim = [54, 54]
img = np.zeros(dim, dtype=np.complex128)

inverseFT(img, vis0, u0, v0)
inverseFT(img, vis1, u1, v1)
#inverseFT(img, 1, 0.2,1)

plt.imshow(np.real(img))
print(img[0,0])



NufftObj = NUFFT_cpu()
cell = math.radians(48/3600.0)
u0 = (-cell)*(-112.15 / wave_length)
v0 = (cell)*(576.66 / wave_length)
u1 = (-cell)*(71.50 / wave_length)
v1 = (cell)*(-368.67 / wave_length)


Nd = (54, 54)  # image size
Kd = (2*Nd[0], 2*Nd[1])  # k-space size
Jd = (8, 8)  # interpolation size

om = [
      [u0, v0],
      [u1,v1]]

om = np.asarray(om)
NufftObj.plan(om, Nd, Kd, Jd)
res = NufftObj.adjoint(np.asarray([vis0, vis1]))
#res = NufftObj.solve([vis0, vis1], solver='cg',maxiter=50)

plt.imshow(np.real(res))
print(np.max(np.real(res)))


