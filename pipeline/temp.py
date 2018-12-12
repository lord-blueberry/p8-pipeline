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
starlet_base = fourier_starlets(nuft, data, 2)
#active_set[22:42, 22:42] = 1
active_set[255:258, 218:221] = 1
res = data.vis

x = np.zeros(data.imsize)
cache = calc_cache(data.uv, data.imsize, active_set, res, x)
cache_0 = calc_cache_conv(data.uv, data.imsize, starlet_base[0], active_set, res, x)
cache_1 = calc_cache_conv(data.uv, data.imsize, starlet_base[1], active_set, res, x)
cache_2 = calc_cache_conv(data.uv, data.imsize, starlet_base[1], active_set, res, x)

print(_magnitude(res))
res, x = CD(lambda_cs, active_set, cache_0, res, x)
print(_magnitude(res))

for i in range(0, 10):
    res, x = CD(lambda_cs/100.0, active_set, cache_0, res, x)
    print(_magnitude(res))
write_img(x, "bla")
print(np.max(x))
print(x[256,219])
print(x[255,218])


write_img(nuft.ifft_normalized(res), bmark[idx]+"_residual")
write_img(active_set, "bla")



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






from algorithms.CoordinateDescent2 import CoordinateDescent_slow as CD_s
lambda_cs = 0.5
active_set = np.zeros(data.imsize)
#active_set[22:42, 22:42] = 1
active_set[255:258, 218:221] = 1
res = data.vis
x = np.zeros(data.imsize)
res, x = CD_s(lambda_cs, data.imsize, active_set, data.uv, res, x)
res, x = CD_s(lambda_cs, data.imsize, active_set, data.uv, res, x)
res, x = CD_s(lambda_cs, data.imsize, active_set, data.uv, res, x)
res, x = CD_s(lambda_cs, data.imsize, active_set, data.uv, res, x)
write_img(x, "bla2")

def printstuff(cache, dimensions, active_set, uv, res, x):
    cache_idx = 0
    uv = -2j * np.pi * uv
    center_pixel = math.floor(dimensions[0] / 2.0)
    
    for xi in range(0, x.shape[0]):
        for yi in range(0, x.shape[1]):
            if(active_set[xi, yi] > 0.0):
                if(xi <= 257 and xi >= 255) and (yi <= 220 and yi >= 218):
                    x_old = x[xi, yi]
                        
                    pix = np.dot(uv, np.asarray([xi-center_pixel, yi-center_pixel]))
                    f_col = np.exp(pix)
                    f_r = np.real(f_col)
                    f_i = np.imag(f_col)
                    res_r = np.real(res)
                    res_i = np.imag(res)

                    #calc -b/(2a)
                    a = np.sum(np.square(f_r) + 2*f_r*f_i + np.square(f_i))
                    b = np.sum(f_r*res_r + f_r*res_i + f_i*res_r + f_i*res_i) 
                    x_new = b / a # this times -2, it cancels out the -1/2 of the original equation
                    
                    print(x_old, xi, yi, xi-center_pixel, yi-center_pixel)
                    #print(x_new, xi, yi)
                    cache_idx += 1
                else:
                    cache_idx += 1
                    
printstuff(cache, data.imsize, active_set, data.uv, res, x)






"""

def reverse(cache, dimensions, active_set, uv, res, x):
    cache_idx = 0
    uv = -2j * np.pi * uv
    center_pixel = math.floor(dimensions[0] / 2.0)
    
    for xi in range(0, x.shape[0]):
        for yi in range(0, x.shape[1]):
            if(active_set[xi, yi] > 0.0):
                x_old = x[xi, yi]
                pix = np.asarray([xi-center_pixel, yi-center_pixel])

                f_col = np.exp((-1)*np.dot(uv, pix)) / res.size
                x[xi, yi] = np.real(np.sum(f_col*res))
                print(x[xi, yi] , xi, yi, xi-center_pixel, yi-center_pixel)
                #print(x_new, xi, yi)
                cache_idx += 1
            else:
                cache_idx += 1
                    
reverse(cache, data.imsize, active_set, data.uv, res, x)
"""
    