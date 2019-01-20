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
    
def run_CD2():
    idx=1
    data = read_all_freq(prefix+bmark[idx]+"/simulation.ms", column[idx], size[idx], cell[idx])
    #data = load_debug()
    nuft = nfft(data)
    write_img(nuft.ifft_normalized(data.vis), bmark[idx]+"_dirty")
    
    from algorithms.CoordinateDescent2 import CoordinateDescent1 as CD
    from algorithms.CoordinateDescent2 import calc_cache 
    from algorithms.CoordinateDescent2 import _magnitude
    
    
    def starlet_img(starlets):
        return starlets.sum(axis=0).reshape(data.imsize)
    
    def dump_starlets(starlets, name):
        for i in range(0, starlets.shape[0]):
            write_img(starlets[i].reshape(data.imsize), name+str(i))
            
    def dump_starlets_weird(starlets, name):
        reg = 0.01
        for i in range(0, starlets.shape[0]):
            write_img(shrink(starlets[i], reg).reshape(data.imsize), name+str(i))
            reg = reg * 10
    
    lambda_cs = 0.5
    active_set = np.zeros(data.imsize)
    #active_set[22:42, 22:42] = 1
    active_set[255:258, 218:221] = 1
    res = data.vis
    
    x = np.zeros(data.imsize)
    cache = calc_cache(data.uv, data.imsize, active_set, res, x)
    
    print(_magnitude(res))
    res, x = CD(lambda_cs, active_set, cache, res, x)
    print(_magnitude(res))
    
    for i in range(0, 10):
        res, x = CD(lambda_cs/100.0, active_set, cache, res, x)
        print(_magnitude(res))
    write_img(x, "bla")
    print(np.max(x))
    print(x[256,219])
    print(x[255,218])
    
    
    write_img(nuft.ifft_normalized(res), bmark[idx]+"_residual")
    write_img(active_set, "bla")
    
    
    import numpy.fft as fftnumpy
    def printstuff(cache, active_set, res, x):
        cache_idx = 0
        for xi in range(0, x.shape[0]):
            for yi in range(0, x.shape[1]):
                if(active_set[xi, yi] > 0.0):
                    if(xi <= 257 and xi >= 255) and (yi <= 220 and yi >= 218):
                        x_old = x[xi, yi]
                        
                        f_col = cache[cache_idx]
                        cache_idx += 1
                        f_r = np.real(f_col)
                        f_i = np.imag(f_col)
                        res_r = np.real(res)
                        res_i = np.imag(res)
        
                        #calc -b/(2a)
                        a = np.sum(np.square(f_r) + 2*f_r*f_i + np.square(f_i))
                        b = np.sum(f_r*res_r + f_r*res_i + f_i*res_r + f_i*res_i) 
                        x_new = b / a # this times -2, it cancels out the -1/2 of the original equation
                        
                        print(x_old, xi, yi)
                        #print(x_new, xi, yi)
                    else:
                        print(xi, yi)
                        cache_idx += 1
                        
    printstuff(cache, active_set, res, x)

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

def run_CD_starlet(idx):
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
    
    prefix_csv="./img_output/"
    
    starlet_levels = 7
    lambda_cs = 0.01
    starlet_base = fourier_starlets(nuft, data, starlet_levels)
    starlets = _nfft_approximation(nuft, data.imsize,starlet_base, 0.0, data.vis)
    write_img(starlets[0], "starlets0")
    np.savetxt(prefix_csv+"starlets0", starlets[0], delimiter=",")
    
    equi_base = equi_starlets(data, starlet_levels)
    x_starlets = np.zeros((starlet_levels+1, data.imsize[0], data.imsize[1]))
    residuals = data.vis
    
    debug = np.zeros(data.imsize)
    for i in range(0,9):
        residuals, x_starlets, full_cache_debug = full_algorithm(data, nuft, 1000, starlet_base, lambda_cs, residuals, x_starlets)
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

    
run_CD_starlet(0)