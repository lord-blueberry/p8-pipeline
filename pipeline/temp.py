#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  7 02:56:26 2018

@author: jon
"""

idx=1
data = read_all_freq(prefix+bmark[idx]+"/simulation.ms", column[idx], size[idx], cell[idx])
#data = load_debug()
nuft = nfft(data)
write_img(nuft.ifft_normalized(data.vis), bmark[idx]+"_dirty")

from algorithms.CoordinateDescent2 import CoordinateDescent1 as CD
from algorithms.CoordinateDescent2 import calc_cache
from algorithms.CoordinateDescent2 import calc_cache_conv
from algorithms.CoordinateDescent2 import fourier_starlets
from algorithms.CoordinateDescent2 import _magnitude
from algorithms.CoordinateDescent2 import _shrink
from algorithms.CoordinateDescent2 import calc_residual


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

lambda_cs = 0.02
active_set = np.zeros(data.imsize)
starlet_base = fourier_starlets(nuft, data, 2)
#active_set[22:42, 22:42] = 1
active_set[250:262, 211:227] = 1
res = data.vis

x = np.zeros(data.imsize)
x0 = np.zeros(data.imsize)
x1 = np.zeros(data.imsize)
x2 = np.zeros(data.imsize)
starlet_base = fourier_starlets(nuft, data, 2)

cache = calc_cache(data.uv, data.imsize, active_set, res, x)
cache0 = calc_cache_conv(data.uv, data.imsize, starlet_base[0], active_set, res, x)
cache1 = calc_cache_conv(data.uv, data.imsize, starlet_base[1], active_set, res, x)
cache2 = calc_cache_conv(data.uv, data.imsize, starlet_base[2], active_set, res, x)

for i in range(0, 40):
    res, x = CD(lambda_cs, active_set, cache, res, x)
    
for i in range(0, 10):
    res, x0 = CD(lambda_cs, active_set, cache0, res, x0)
print(_magnitude(res))
for i in range(0, 10):
    res, x1 = CD(lambda_cs, active_set, cache1, res, x1)
for i in range(0, 10):
    res, x2 = CD(lambda_cs, active_set, cache2, res, x2)

print(_magnitude(res))
x_star = x0+x1+x2
write_img(x_star, "bla")
print(np.max(x_star))
print(x_star[256,219])
print(x_star[255,218])


write_img(nuft.ifft_normalized(res*cache), bmark[idx]+"_residual")
write_img(active_set, "bla")



res, x0 = CD(lambda_cs, active_set, cache0, res, x0)
write_img(x0, "bla")
write_img(x1, "bla")
write_img(x2, "bla")

def printstuff(cache, active_set, res, x):
    cache_idx = 0
    for xi in range(0, x.shape[0]):
        for yi in range(0, x.shape[1]):
            if(active_set[xi, yi] > 0.0):
                x_old = x[xi, yi]
                uv = -2j * np.pi * data.uv
                center_pixel = math.floor(data.imsize[0] / 2.0)
                pix = np.dot(uv, np.asarray([xi-center_pixel, yi-center_pixel]))
                
                res_tmp = res * starlet_base[0]
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

dirty = nuft.ifft(data.vis)
starlet_0_img = _shrink(nuft.ifft_normalized(res* starlet_base[0]), lambda_cs=0.02)
res2 = res - nuft.fft(starlet_0_img)
printstuff(cache0, active_set, res2, starlet_0_img)
write_img(dirty - starlet_0_img, "starlet 0")


tmp = np.asarray([1.0 / 16.0, 1.0 / 4.0, 3.0 / 8.0, 1.0 / 4.0, 1.0 / 16.0])
row = np.asmatrix([tmp])
bspline =(np.dot(np.transpose(row), row))
tmp = np.zeros(data.imsize)
tmp[0:5, 0:5] = bspline
tmp = np.roll(tmp, -2, axis= 0)
tmp = np.roll(tmp, -2, axis= 1)

import numpy.fft as ff

convF = ff.fft2(tmp)
dirty = nuft.ifft_normalized(res)
dirtyF = ff.fft2(dirty)
dirtyConf = np.real(ff.ifft2((dirtyF * convF)))
printstuff(cache0, active_set, res, dirtyConf)