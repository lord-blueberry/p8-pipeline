#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 12 20:30:46 2018

@author: jon
"""

idx=0
data = read_all_freq(prefix+bmark[idx]+"/simulation.ms", column[idx], size[idx], cell[idx])
#data = load_debug()
nuft = nfft(data)
write_img(nuft.ifft_normalized(data.vis), bmark[idx]+"_dirty")
import numpy.fft as fftnumpy

def toImage(starlets):
    starlets_convolved = starlets.copy()
    for J in range(0, starlet_levels +1):
        x_fft = fftnumpy.fft2(starlets_convolved[J])
        res_starlet = np.real(fftnumpy.ifft2(x_fft * equi_base[J]))
        starlets_convolved[J] = res_starlet
    return starlets_convolved.sum(axis=0)

from algorithms.CoordinateDescent2 import CoordinateDescent1 as CD
from algorithms.CoordinateDescent2 import calc_cache
from algorithms.CoordinateDescent2 import fourier_starlets
from algorithms.CoordinateDescent2 import equi_starlets
from algorithms.CoordinateDescent2 import _magnitude
from algorithms.CoordinateDescent2 import _shrink
from algorithms.CoordinateDescent2 import calc_residual
from algorithms.CoordinateDescent2 import _nfft_approximation
from algorithms.CoordinateDescent2 import to_image



def ago_separate(data, nuft, max_full, starlet_base, equi_base, starlet_levels, lambda_cs, residuals, x_starlets):
    full_cache_debug = np.zeros(data.imsize)
    print(_magnitude(residuals))
    for J in range(0, starlet_levels+1):
        #approx
        starlets = _nfft_approximation(nuft, data.imsize, starlet_base, 0.0, residuals)
        active_set, active_lambda = calc_active(starlets[J], max_full, lambda_cs)
        print("found active set with ", np.count_nonzero(active_set))
        active_set[active_set > 0.0] = 1
        full_cache_debug = full_cache_debug + active_set
        
        cache = calc_cache(data.uv, data.imsize, active_set, data.vis)
        print("calculated cache")
        for J2 in range(0, starlet_levels+1):
            res_tmp = residuals * starlet_base[J2]
            x = x_starlets[J2].copy()
            
            for i in range(0, 10):
                res_tmp, x = CD(lambda_cs, active_set, cache, res_tmp, x)
            x_diff = x - x_starlets[J2]
            x_starlets[J2] = x
            res_diff = np.zeros(data.vis.shape)
            res_diff = calc_residual(active_set, cache, res_diff, x_diff)
            residuals = residuals - (res_diff * starlet_base[J2])
        print(_magnitude(residuals))
    return residuals, x_starlets, active_set, full_cache_debug


def calc_active(img, max_full, start_lambda):
    current_l = start_lambda
    tmp = _shrink(img, current_l)
    while np.count_nonzero(tmp) > max_full:
        current_l = current_l * 10.0
        tmp = _shrink(img, current_l)
    
    nonzeros = np.count_nonzero(tmp)
    if nonzeros < max_full and nonzeros >= 10:
        return tmp, current_l
    
    diff = current_l - current_l / 10.0
    while (nonzeros > max_full or nonzeros <= 10) and diff > 0.00001:
        diff = diff/2
        if nonzeros > max_full:
            current_l = current_l + diff
        else:
            current_l = current_l - diff
        tmp = _shrink(img, current_l)
        nonzeros = np.count_nonzero(tmp)
    
    return tmp, current_l



starlet_levels = 4
lambda_cs = 0.01
starlet_base = fourier_starlets(nuft, data, starlet_levels)
equi_base = equi_starlets(data, starlet_levels)
x_starlets = np.zeros((starlet_levels+1, data.imsize[0], data.imsize[1]))
residuals = data.vis
res2, x_new ,active, full_cache_debug = ago_separate(data, nuft, 1000, starlet_base, equi_base, starlet_levels, lambda_cs, residuals, x_starlets)
write_img(toImage(x_new), "result")
bla = full_cache_debug
bla[bla > 0] = 1
write_img(bla, "active")


res2, x_new ,active, full_cache_debug = ago_separate(data, nuft, 1000, starlet_base, equi_base, starlet_levels, lambda_cs, res2, x_new)
write_img(toImage(x_new), "result")
bla = full_cache_debug
bla[bla > 0] = 1
write_img(bla, "active")
write_img(nuft.ifft_normalized(res2), bmark[idx]+"res")

write_img(x_new[3], "bla")










write_img(nuft.ifft_normalized(residuals), bmark[idx]+"res")
res2 = data.vis - nuft.fft(toImage(x_starlets))
write_img(nuft.ifft_normalized(res2), bmark[idx]+"res")






full_cache_debug = np.zeros(data.imsize)
print(_magnitude(residuals))


max_full = 1000
#approx
J = 0

starlets = _nfft_approximation(nuft, data.imsize, starlet_base, 0.0, residuals)
active_set, active_lambda = calc_active(starlets[J], max_full, lambda_cs)
print("found active set with ", np.count_nonzero(active_set))
active_set[active_set > 0.0] = 1
full_cache_debug = full_cache_debug + active_set


for J in range(0, starlet_levels+1):
    #approx
    starlets = _nfft_approximation(nuft, data.imsize, starlet_base, 0.0, residuals)
    active_set, active_lambda = calc_active(starlets[J], max_full, lambda_cs)
    print("found active set with ", np.count_nonzero(active_set))
    active_set[active_set > 0.0] = 1
    full_cache_debug = full_cache_debug + active_set
    
    cache = calc_cache(data.uv, data.imsize, active_set, data.vis)
    print("calculated cache")
    for J2 in range(0, starlet_levels+1):
        res_tmp = residuals * starlet_base[J2]
        x = x_starlets[J2].copy()
        
        for i in range(0, 10):
            res_tmp, x = CD(lambda_cs, active_set, cache, res_tmp, x)
        x_diff = x - x_starlets[J2]
        x_starlets[J2] = x
        res_diff = np.zeros(data.vis.shape)
        res_diff = calc_residual(active_set, cache, res_diff, x_diff)
        residuals = residuals - (res_diff * starlet_base[J2])
    print(_magnitude(residuals))


write_img(toImage(x_starlets), "result")

write_img(full_cache_debug, "cache")