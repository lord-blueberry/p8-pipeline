#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  5 15:29:03 2018

@author: jon
"""

from msinput import MS_jon
from scipy import constants
import matplotlib.pyplot as plt
import numpy as np
import os
import math

from debug_data.load_debug import load_debug

from InputData import InputData
from nufftwrapper import nfft_wrapper as nfft


def read_all_freq(msfile, data_column, imsize, cell):
    ms = MS_jon()
    ms.read_ms(msfile, data_column=data_column)
    
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
    
    return InputData(vis,uv, np.ones(vis.size), imsize, cell)


def write_img(image, name):
    imprefix="./img_output/"
    width=image.shape[0] # pixels
    height=image.shape[1]
    margin=50 # pixels
    dpi=100. # dots per inch
    
    figsize=((width+2*margin)/dpi, (height+2*margin)/dpi) # inches
    left = margin/dpi/figsize[0] #axes ratio
    bottom = margin/dpi/figsize[1]
    
    fig = plt.figure(figsize=figsize, dpi=dpi)
    fig.subplots_adjust(left=left, bottom=bottom, right=1.-left, top=1.-bottom)
    
    plt.imshow(np.flipud(np.transpose(np.real(image))), interpolation="bilinear")
    plt.savefig(imprefix + name + '.png')
    
prefix= "./benchmark/simulated_data/"
bmark = ["sim00_mixed_sources", "sim01_point", "sim02_point"]
column = ["CORRECTED_DATA","CORRECTED_DATA", "DATA" ]
size = [(1080,1080), (256, 256), (256, 256)]
cell = np.radians(np.asarray([[0.5, -0.5], [0.5, -0.5], [0.5, -0.5]]) / 3600.0)



def run_debug():
    data = load_debug()

    nuft = nfft(data)
    from algorithms.CoordinateDescent import CoordinateDescent as CD
    cd_alg = CD(data, nuft, 2)
    _, starlets = cd_alg.optimize(data.vis.copy(), 0.02)
    plt.imshow(np.reshape(starlets.sum(axis=0), (64,64)))

def run_CD(idx):
    data = read_all_freq(prefix+bmark[idx]+"/simulation.ms", column[idx], size[idx], cell[idx])
    nuft = nfft(data)
    write_img(nuft.ifft_normalized(data.vis), bmark[idx]+"_dirty")
    
    from algorithms.CoordinateDescent import CoordinateDescent as CD
    cd_alg = CD(data,nuft , 4)
    
    _, starlets = cd_alg._nfft_approximation(data.vis, 4.0)
    img = starlets.sum(axis=0).reshape(data.imsize)
    write_img(img, bmark[idx]+"_approx")
    
    img = starlets[4].reshape(data.imsize)
    write_img(img, bmark[idx]+"_approx")
    
    residuals, starlets = cd_alg.optimize(data.vis.copy(), 4.0)
    img = starlets.sum(axis=0).reshape(data.imsize)
    write_img(img, bmark[idx]+"_run1")
    
    img = cd_alg.rerun_inner_cd_cached(residuals, 4.0)
    img = starlets.sum(axis=0).reshape(data.imsize)
    write_img(img, bmark[idx]+"_run2")

    
    

