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

def singleFrequency():
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

def allFrequencies():
    imsize = (256, 256)
    cell = np.asarray([0.5, -0.5]) / 3600.0 # #arcseconds. revert v axis because it is empirically right. the axes in the ms are not really standardized
    cell = np.radians(cell)
    ms = MS_jon()
    ms.read_ms("simkat64-default.ms")
    
    wavelengths = ms.freq_array[0] / constants.c
    
    offset = ms.uvw_array.shape[0]
    start = 0
    end = offset
    uv = np.zeros((ms.uvw_array.shape[0] * wavelengths.size, 2))
    vis = np.zeros(ms.uvw_array.shape[0] * wavelengths.size, dtype=np.complex128)
    for i in range(0, wavelengths.size):
        uvw_wavelengths = np.dot(ms.uvw_array, np.diag(np.repeat(wavelengths[i], 3)))
        #skip w component
        uv[start:end] = uvw_wavelengths[:, 0:2]
        #add the XX and YY Polarization to get an intensity
        vis[start:end] = ms.data_array[:, 0, i, 0] + ms.data_array[:, 0, i, 3]
        start += offset
        end += offset
    uv = np.multiply(uv, cell)
    
    plan = NFFT(imsize, uv.shape[0])
    plan.x = uv.flatten()
    plan.precompute()
    
    plan.f = vis
    dirty = plan.adjoint() / uv.shape[0] / 2
    plt.imshow(np.real(dirty))
    print(np.max(np.real(dirty)))
    
    
    return 0

allFrequencies()