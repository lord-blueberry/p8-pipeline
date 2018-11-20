#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  7 13:32:28 2018

@author: jon
"""
#import sys
#from pyuvdata import UVData
from pynfft import NFFT
import numpy as np
import matplotlib.pyplot as plt
from scipy import constants

from mslib import MS_jon

imsize = (256, 256)
cell = np.asarray([0.5, -0.5]) / 3600.0 # #arcseconds. revert v axis because it is empirically right. the axes in the ms are not really standardized
cell = np.radians(cell)
ms = MS_jon()
ms.read_ms("simkat64-default.ms")
# 4 polarizations are XX, XY, YX and YY
#Intensity image should be XX + YY

wavelengths = ms.freq_array[0, 0] / constants.c
uvw_wavelengths = np.dot(ms.uvw_array, np.diag(np.repeat(wavelengths, 3)))
uv = np.multiply(uvw_wavelengths[:,0:2], cell)

plan = NFFT(imsize, uv.shape[0])
plan.x = uv.flatten()
plan.precompute()

plan.f = ms.data_array[:,:,0,0]
dirty = plan.adjoint() / uv.shape[0]
plt.imshow(np.flipud(np.transpose(np.real(dirty))))  

