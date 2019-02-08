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

import numpy.fft as fftnumpy
def to_image_pos(starlets, equi_base):
    starlets_convolved = starlets.copy()
    for J in range(0, len(starlets)):
        x_fft = fftnumpy.fft2(starlets_convolved[J])
        res_starlet = np.real(fftnumpy.ifft2(x_fft * equi_base[J]))
        starlets_convolved[J] = res_starlet
    summed = starlets_convolved.sum(axis=0)
    summed[summed < 0] = 0 
    return summed


def run_CD_starlet(idx):
    data = read_all_freq(prefix+bmark[idx]+"/simulation.ms", column[idx], size[idx], cell[idx])
    prefix_csv="./img_output/"
    #data = load_debug()
    nuft = nfft(data)
    dirty = nuft.ifft_normalized(data.vis)
    write_img(dirty, bmark[idx]+"_dirty")
    np.savetxt(prefix_csv+"dirty", dirty, delimiter=",")
    
    from algorithms.CoordinateDescent3 import full_algorithm
    from algorithms.CoordinateDescent3 import to_image
    from algorithms.CoordinateDescent3 import fourier_starlets
    from algorithms.CoordinateDescent3 import equi_starlets
    from algorithms.CoordinateDescent3 import _nfft_approximation
    from algorithms.CoordinateDescent3 import positive_starlets
    
    
    starlet_levels = 4
    lambda_cs = 0.5
    equi_base = equi_starlets(data, starlet_levels)
    starlet_pos_base, equi_pos_base =  positive_starlets(nuft, data.vis.size, data.imsize, starlet_levels)
    starlet_base = fourier_starlets(nuft, data, starlet_levels)
    
    #starlet_pos_base = starlet_base
    #equi_pos_base = equi_base
    
    starlets = _nfft_approximation(nuft, data.imsize,starlet_base, 0.0, data.vis)
    write_img(starlets[0], "starlets0")
    np.savetxt(prefix_csv+"starlets0", starlets[0], delimiter=",")
    
    x_starlets = np.zeros((starlet_levels+1, data.imsize[0], data.imsize[1]))
    residuals = data.vis
    
    debug = np.zeros(data.imsize)
    for i in range(0,1):
        residuals, x_starlets, full_cache_debug = full_algorithm(data, nuft, 1000, starlet_base, starlet_pos_base, lambda_cs, residuals, x_starlets)
        debug += full_cache_debug
        reconstruction = to_image(x_starlets, equi_pos_base)

        write_img(nuft.ifft_normalized(residuals), "res"+str(i))
        write_img(reconstruction, "image"+str(i))
        np.savetxt(prefix_csv+"image"+str(i), reconstruction, delimiter=",")
        np.savetxt(prefix_csv+"image_pos"+str(i), to_image_pos(x_starlets, equi_pos_base), delimiter=",")
        print("nonzero ", np.count_nonzero(x_starlets))
    write_img(debug, "full_cache_debug")
    np.savetxt(prefix_csv+"full_cache_debug", debug, delimiter=",")

    
run_CD_starlet(1)