#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 12 16:29:59 2018

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


starlet_levels=2
lambda_cs = 0.01
active_set = np.zeros(data.imsize)
starlet_levels = 4
starlet_base = fourier_starlets(nuft, data, starlet_levels)
equi_base = equi_starlets(data, starlet_levels)
#active_set[22:42, 22:42] = 1
#active_set[250:262, 250:262] = 1
res = data.vis

starlets = _nfft_approximation(nuft, data.imsize, starlet_base, 3000, data.vis)
active_set = starlets.sum(axis=0)
active_set[active_set > 0] = 1
starlets = np.zeros((starlet_levels+1, data.imsize[0], data.imsize[1]))
cache = calc_cache(data.uv, data.imsize, active_set, res)

for J in range(0, starlet_levels+1):
    x = starlets[J]
    x[active_set <= 0.0] = 0
    starlets[J] = x
    
res = data.vis
print(_magnitude(res))
starlets_old = starlets.copy()
for J in range(0, starlet_levels+1):
    print(J)
    res_tmp = res * starlet_base[J]
    x = starlets[J]
    for i in range(0, 10):
        res_tmp, x = CD(lambda_cs*(10-i), active_set, cache, res_tmp, x)
    starlets[J] = x
    res = res_tmp / starlet_base[J]
#diff = starlets - starlets_old
res2 = data.vis - nuft.fft(toImage(starlets))
print(_magnitude(res2))
print(_magnitude(res))
write_img(toImage(starlets), "bla")

dirty = nuft.ifft_normalized(data.vis)
dirty[active_set <= 0] = 0

starlets_old = starlets.copy()
for J in range(0, starlet_levels+1):
    print(J)
    res_tmp = res * starlet_base[J]
    x = starlets[J]
    for i in range(0, 5):
        res_tmp, x = CD(lambda_cs, active_set, cache, res_tmp, x)
    starlets[J] = x

write_img(toImage(starlets), "bla")

starlets = _nfft_approximation(nuft, data.imsize, starlet_base, lambda_cs, data.vis)
starlets_cd = starlets.copy()
x = starlets.sum(axis=0)
x_fft = fftnumpy.fft2(x)
for J in range(0, starlet_levels +1):
    res_starlet = np.real(fftnumpy.ifft2(x_fft * equi_base[J]))
    res_starlet = _shrink(res_starlet, lambda_cs)
    res_starlet[active_set <= 0.0] = 0
    starlets[J] = res_starlet

np.abs(starlets_cs - starlets).sum()
bla = 4
write_img(np.abs(starlets_cd[bla] - starlets[bla]), "bla")




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
            current_lambda = lambda_cs * (J2+1)
            res_tmp = residuals * starlet_base[J]
            x = x_starlets[J].copy()
            
            for i in range(0, 10):
                res_tmp, x = CD(lambda_cs*(10-i), active_set, cache, res_tmp, x)
            x_diff = x - x_starlets[J]
            x_starlets[J] = x
            res_diff = np.zeros(data.vis.shape)
            res_diff = calc_residual(active_set, cache, res_diff, x_diff)
            residuals = residuals - (res_diff * starlet_base[J])
        print(_magnitude(residuals))
    return residuals, x_starlets, active_set, full_cache_debug


def calc_active(img, max_full, start_lambda):
    out = img.copy()
    current_l = start_lambda
    tmp = _shrink(img, current_l)
    while np.count_nonzero(tmp) > max_full:
        current_l = current_l * 10.0
        tmp = _shrink(img, current_l)

    nonzeros = np.count_nonzero(tmp)
    if nonzeros < max_full and nonzeros >= 10:
        return tmp, current_l
    
    lower_l = current_l / 10.0
    diff = current_l - lower_l
    while (nonzeros > max_full or nonzeros <= 10) and diff > 0.0001:
        diff = diff/2
        if np.count_nonzero(tmp) > max_full:
            lower_l = lower_l - diff
        else:
            lower_l = lower_l + diff
        tmp = _shrink(img, lower_l)
        nonzeros = np.count_nonzero(tmp)
    return tmp, lower_l

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

def printstuff(cache, active_set, res, x):
    cache_idx = 0
    for xi in range(0, x.shape[0]):
        for yi in range(0, x.shape[1]):
            if(active_set[xi, yi] > 0.0):
                x_old = x[xi, yi]
                uv = -2j * np.pi * data.uv
                center_pixel = math.floor(data.imsize[0] / 2.0)
                pix = np.dot(uv, np.asarray([xi-center_pixel, yi-center_pixel]))
                
                res_tmp = res
                f_col =  (np.exp(pix))
                #f_col = cache[cache_idx]
                #cache_idx += 1
                f_r = np.real(f_col)
                f_i = np.imag(f_col)
                res_r = np.real(res_tmp)
                res_i = np.imag(res_tmp)

                #calc -b/(2a)
                a = np.sum(np.square(f_r) + 2*f_r*f_i + np.square(f_i))
                b = np.sum(f_r*res_r + f_r*res_i + f_i*res_r + f_i*res_i) 
                x_new = b / a # this times -2, it cancels out the -1/2 of the original equation
                
                x_plus = x_new
                x_new = _shrink(x_new + x_old, lambda_cs)
                print(x_new, xi, yi, x_old, x_plus)
                
printstuff(cache, active_set, res, starlets.sum(axis=0))
