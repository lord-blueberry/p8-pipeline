#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  7 13:08:38 2018

@author: jon
"""

import numpy as np
import math
import matplotlib.pyplot as plt

def _shrink(x, lambda_cs):
    out = np.sign(x) * np.maximum(np.abs(x) - lambda_cs, 0.)
    return np.maximum(out, 0.)

def _magnitude(vis):
    return np.sum(np.square(np.real(vis))+np.square(np.imag(vis)))


def CoordinateDescent_slow(lambda_cs, dimensions, active_set, uv, res, x):
    res_out = res
    uv = -2j * np.pi * uv
    center_pixel = math.floor(dimensions[0] / 2.0)
    for xi in range(0, x.shape[0]):
        for yi in range(0, x.shape[1]):
            if(active_set[xi, yi] > 0.0):
                x_old = x[xi, yi]
                
                pix = np.dot(uv, np.asarray([xi-center_pixel, yi-center_pixel]))
                f_col = np.exp(pix)
                f_r = np.real(f_col)
                f_i = np.imag(f_col)
                res_r = np.real(res_out)
                res_i = np.imag(res_out)

                #calc -b/(2a)
                a = np.sum(np.square(f_r) + 2*f_r*f_i + np.square(f_i))
                b = np.sum(f_r*res_r + f_r*res_i + f_i*res_r + f_i*res_i) 
                x_new = b / a # this times -2, it cancels out the -1/2 of the original equation
  
                x_new = _shrink(x_new + x_old, lambda_cs)
                x[xi, yi] = x_new
                diff = x_new - x_old
                diff_res = f_col*diff
                res_out = res_out - diff_res
    return res_out, x


def CoordinateDescent1(lambda_cs, active_set, cache, res, x):
    res_out = res
    cache_idx = 0
    x_out = x
    for xi in range(0, x.shape[0]):
        for yi in range(0, x.shape[1]):
            if(active_set[xi, yi] > 0.0):
                x_old = x_out[xi, yi]
                
                f_col = cache[cache_idx]
                cache_idx += 1
                f_r = np.real(f_col)
                f_i = np.imag(f_col)
                res_r = np.real(res_out)
                res_i = np.imag(res_out)

                #calc -b/(2a)
                a = np.sum(np.square(f_r) + 2*f_r*f_i + np.square(f_i))
                b = np.sum(f_r*res_r + f_r*res_i + f_i*res_r + f_i*res_i) 
                x_new = b / a # this times -2, it cancels out the -1/2 of the original equation
  
                x_new = _shrink(x_new + x_old, lambda_cs)
                x_out[xi, yi] = x_new
                diff = x_new - x_old
                diff_res = f_col*diff
                res_out = res_out - diff_res
    return res_out, x_out


def CoordinateDescent2(lambda_cs, active_set, cache, res, x):
    res_out = res
    cache_idx = 0
    for xi in range(0, x.shape[0]):
        for yi in range(0, x.shape[1]):
            if(active_set[xi, yi] > 0.0):
                x_old = x[xi, yi]
                
                f_col = cache[cache_idx]
                cache_idx += 1
                f_r = np.real(f_col)
                f_i = np.imag(f_col)
                res_r = np.real(res_out)
                res_i = np.imag(res_out)

                #calc -b/(2a)
                a = np.sum(np.square(f_r) + 2*f_r*f_i + np.square(f_i))
                b = np.sum(f_r*res_r + f_r*res_i + f_i*res_r + f_i*res_i) 
                x_new = (b+lambda_cs) / a + x_old# this times -2, it cancels out the -1/2 of the original equation
                
                #soft threshold
                #if x_new > lambda_cs:
  
                if 0.0 > x_new :
                    x_new = 0
                x[xi, yi] = x_new
                res_out = res_out - f_col*(x_new - x_old) 
    return res_out, x


def fourier_starlets(nfft, data, starlet_levels):
        def insert_spaced(mat, kernel, J):
            mat_center = int(math.floor(mat.shape[0] / 2.0))
            kernel_center = int(math.floor(kernel.shape[0] / 2.0))   #kernel is always square
            disp = 2**J
            for xi in range(-kernel_center,kernel_center+1):
                for yi in range(-kernel_center,kernel_center+1):
                    val = kernel[xi + kernel_center, yi + kernel_center]
                    x_disp = xi * disp
                    y_disp = yi * disp
                    mat[mat_center +x_disp, mat_center + y_disp] = val
            return mat
        
        tmp = np.asarray([1.0 / 16.0, 1.0 / 4.0, 3.0 / 8.0, 1.0 / 4.0, 1.0 / 16.0])
        row = np.asmatrix([tmp])
        bspline =(np.dot(np.transpose(row), row))
        
        four_base = np.zeros((starlet_levels+1, data.vis.size), dtype=np.complex128)
        tmp = np.zeros(data.imsize)
        last_conv_mat = None
        for J in range(0, starlet_levels):
            new_mat = nfft.fft(insert_spaced(tmp.copy(), bspline, J)) / data.vis.size
            if last_conv_mat is None:
                four_base[J] = (1 - new_mat)
                last_conv_mat = new_mat
            else:
                #(conv_mat[J-1]- conv_mat[J]) arises from the multi-scale wavelet analysis
                #w_J = c_(J-1) - conv_mat_J *c_(J-1)
                new_mat = last_conv_mat * new_mat
                four_base[J] = (last_conv_mat - new_mat)
                last_conv_mat = new_mat
            
        #add cJ as last
        four_base[starlet_levels] = last_conv_mat
        return four_base



def calc_cache(uv, dimensions, active_set, res, x):
    uv = -2j * np.pi * uv
    center_pixel = math.floor(dimensions[0] / 2.0)
    cache = np.zeros((np.count_nonzero(active_set), res.size), dtype=np.complex128)
    cache_idx = 0
    for xi in range(0, x.shape[0]):
        for yi in range(0, x.shape[1]):
            if(active_set[xi, yi] > 0.0):
                pix = np.dot(uv, np.asarray([xi-center_pixel, yi-center_pixel]))
                f_col = np.exp(pix)
                cache[cache_idx] = f_col
                cache_idx += 1
    return cache    

def calc_cache_conv(uv, dimensions, conv, active_set, res, x):
    uv = -2j * np.pi * uv
    center_pixel = math.floor(dimensions[0] / 2.0)
    cache = np.zeros((np.count_nonzero(active_set), res.size), dtype=np.complex128)
    cache_idx = 0

    for xi in range(0, x.shape[0]):
        for yi in range(0, x.shape[1]):
            if(active_set[xi, yi] > 0.0):
                pix = conv * np.dot(uv, np.asarray([xi-center_pixel, yi-center_pixel]))
                f_col = np.exp(pix)
                cache[cache_idx] = f_col
                cache_idx += 1
    return cache    