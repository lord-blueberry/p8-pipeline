# -*- coding: utf-8 -*-


import cmath
import math
import numpy as np
from pynfft import NFFT, Solver
from pynfft.nfft import NFFT
import matplotlib.pyplot as plt

def inverseFT(img, vis, u, v):
    for x in range (0,img.shape[0]):
        for y in range(0, img.shape[1]):
            img[x,y] = img[x,y] + vis * cmath.exp(2j*math.pi * (u*(x-27) + v*(y-27)))

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
vis0 = toComplex(47.753, math.radians(173.2)+2*math.pi*(u0*27+v0*27))
vis0 = toComplex(47.753, math.radians(173.2))


u1 = (-cell)*(71.50 / wave_length)
v1 = (cell)*(-368.67 / wave_length)
vis1 = toComplex(53.456, math.radians(162.4)+2*math.pi*(u1*27+v1*27))
vis1 = toComplex(53.456, math.radians(173.2))
dim = [54,54]
img = np.zeros(dim, dtype=np.complex128)

inverseFT(img, vis0, u0, v0)
inverseFT(img, vis1, u1, v1)
#inverseFT(img, 1, 0.2,1)

plt.imshow(np.real(img))
print(img[0,0])
.


plan = NFFT(dim, 2)
plan.x = np.asarray([u0, v0, u1, v1])
plan.precompute()

plan.f = np.asarray([vis0, vis1])
res = plan.adjoint(use_dft=True)
print(res[0,0])
plt.imshow(np.real(res))





