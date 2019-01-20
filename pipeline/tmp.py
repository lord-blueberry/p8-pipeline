#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 20 14:28:06 2019

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
    
import numpy.fft as fftnumpy
def to_image_debug1(starlets, equi_base):
    starlets_convolved = starlets.copy()
    for J in range(0, len(starlets)):
        x_fft = fftnumpy.fft2(starlets_convolved[J])
        res_starlet = np.real(fftnumpy.ifft2(x_fft * equi_base[J]))
        res_starlet[res_starlet < 0] = 0
        starlets_convolved[J] = res_starlet
    return starlets_convolved.sum(axis=0)

def to_image_debug2(starlets, equi_base):
    starlets_convolved = starlets.copy()
    for J in range(0, len(starlets)):
        x_fft = fftnumpy.fft2(starlets_convolved[J])
        res_starlet = np.real(fftnumpy.ifft2(x_fft * equi_base[J]))
        starlets_convolved[J] = res_starlet
    out = starlets_convolved.sum(axis=0)
    out[out< 0] = 0
    return out

prefix= "./benchmark/simulated_data/"
bmark = ["sim00_mixed_sources", "sim01_point", "sim02_point"]
column = ["CORRECTED_DATA","CORRECTED_DATA", "DATA" ]
size = [(1080,1080), (256, 256), (256, 256)]
cell = np.radians(np.asarray([[0.5, -0.5], [0.5, -0.5], [0.5, -0.5]]) / 3600.0)

idx=1
data = read_all_freq(prefix+bmark[idx]+"/simulation.ms", column[idx], size[idx], cell[idx])
#data = load_debug()
nuft = nfft(data)
write_img(nuft.ifft_normalized(data.vis), bmark[idx]+"_dirty")

from algorithms.CoordinateDescent2 import _magnitude
from algorithms.CoordinateDescent2 import full_algorithm2 as full_algorithm
from algorithms.CoordinateDescent2 import to_image
from algorithms.CoordinateDescent2 import fourier_starlets
from algorithms.CoordinateDescent2 import equi_starlets
from algorithms.CoordinateDescent2 import _nfft_approximation
from algorithms.CoordinateDescent2 import positive_starlets
from algorithms.CoordinateDescent2 import *
prefix_csv="./img_output/"

starlet_levels = 3
lambda_cs = 0.1
starlet_base_orig = fourier_starlets(nuft, data, starlet_levels)
#equi_base = equi_starlets(data, starlet_levels)
starlet_base, equi_base =  positive_starlets(nuft, data.vis.size, data.imsize, starlet_levels)
starlets = _nfft_approximation(nuft, data.imsize,starlet_base, 0.0, data.vis)

orig = nuft.ifft(starlet_base_orig[3] * data.vis)
new = nuft.ifft(starlet_base[3] * data.vis)
np.max(orig - new)



write_img(starlets[0], "starlets0")
np.savetxt(prefix_csv+"starlets0", starlets[0], delimiter=",")


x_starlets = np.zeros((starlet_levels+1, data.imsize[0], data.imsize[1]))
residuals = data.vis

debug = np.zeros(data.imsize)
i = 0
residuals, x_starlets, full_cache_debug = full_algorithm(data, nuft, 1000, starlet_base, lambda_cs, residuals, x_starlets)


J = 0
starlets = _nfft_approximation(nuft, data.imsize, starlet_base, 0.0, residuals)
active_set, active_lambda = calc_active(starlets[J], max_full, lambda_cs)
print("found active set with ", np.count_nonzero(active_set))

active_set[active_set > 0.0] = 1
full_cache_debug = full_cache_debug + active_set

print("calculated cache")
cache = calc_cache(data.uv, data.imsize, active_set, data.vis)
res_tmp = residuals * starlet_base[J]
x = x_starlets[J].copy()

for i in range(0, 10):
    res_tmp, x = CoordinateDescent1(lambda_cs, active_set, cache, res_tmp, x)
x_diff = x - x_starlets[J]
x_starlets[J] = x
res_diff = np.zeros(data.vis.shape)
res_diff = calc_residual(active_set, cache, res_diff, x_diff)
residuals = residuals + (res_diff * starlet_base[J])

print(_magnitude(residuals))




debug += full_cache_debug
reconstruction = to_image(x_starlets, equi_base)
write_img(nuft.ifft_normalized(residuals), "res"+str(i))
write_img(reconstruction, "image"+str(i))
np.savetxt(prefix_csv+"image"+str(i), reconstruction, delimiter=",")
np.savetxt(prefix_csv+"image_1dbg"+str(i), to_image_debug1(x_starlets, equi_base), delimiter=",")
np.savetxt(prefix_csv+"image_2dbg"+str(i), to_image_debug2(x_starlets, equi_base), delimiter=",")
    print("nonzero ", np.count_nonzero(x_starlets))
write_img(debug, "full_cache_debug")
np.savetxt(prefix_csv+"full_cache_debug", debug, delimiter=",")