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


starlet_levels=2
lambda_cs = 0.01
active_set = np.zeros(data.imsize)
starlet_levels = 4
starlet_base = fourier_starlets(nuft, data, starlet_levels)
equi_base = equi_starlets(data, starlet_levels)
#active_set[22:42, 22:42] = 1
active_set[250:262, 250:262] = 1
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





tmp = np.asarray([1.0 / 16.0, 1.0 / 4.0, 3.0 / 8.0, 1.0 / 4.0, 1.0 / 16.0])
row = np.asmatrix([tmp])
bspline =(np.dot(np.transpose(row), row))

tmp = np.zeros(data.imsize, dtype=np.complex128)
def insert_spaced(mat, kernel, J):
    disp = 2**J
    for xi in range(0, kernel.shape[0]):
        for yi in range(0, kernel.shape[1]):
            mat[xi * disp, yi * disp] = kernel[xi, yi]
    roll = -2**(J+1)
    mat = np.roll(mat, roll, axis=0)
    mat = np.roll(mat, roll, axis=1)
    return mat
new_base = insert_spaced(tmp, bspline, 0)
bfft = fftnumpy.fft2(new_base)
write_img(new_base, "bla")







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
