#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  7 13:08:38 2018

@author: jon
"""

import numpy as np
import numpy.fft as fftnumpy
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
            new_mat = nfft.fft(insert_spaced(tmp.copy(), bspline, J))
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

def equi_starlets(data, starlet_levels):
        def insert_spaced(mat, kernel, J):
            disp = 2**J
            for xi in range(0, kernel.shape[0]):
                for yi in range(0, kernel.shape[1]):
                    mat[xi * disp, yi * disp] = kernel[xi, yi]
            roll = -2**(J+1)
            mat = np.roll(mat, roll, axis=0)
            mat = np.roll(mat, roll, axis=1)
            return mat
        
        tmp = np.asarray([1.0 / 16.0, 1.0 / 4.0, 3.0 / 8.0, 1.0 / 4.0, 1.0 / 16.0])
        row = np.asmatrix([tmp])
        bspline =(np.dot(np.transpose(row), row))
        
        four_base = np.zeros((starlet_levels+1, data.imsize[0], data.imsize[1]), dtype=np.complex128)
        tmp = np.zeros(data.imsize)
        last_conv_mat = None
        for J in range(0, starlet_levels):
            new_mat = fftnumpy.fft2(insert_spaced(tmp.copy(), bspline, J))
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


def _nfft_approximation(nuft, dimensions, starlet_base, lambda_cs, vis):
    starlets = np.zeros((len(starlet_base), dimensions[0], dimensions[1]))
    for J in range(0, len(starlet_base)-1):
        tmp = nuft.ifft_normalized(vis * starlet_base[J])
        starlets[J] = _shrink(tmp, lambda_cs)
    
    tmp = nuft.ifft_normalized(vis * starlet_base[len(starlet_base)-1])
    starlets[len(starlet_base)-1] = _shrink(tmp, lambda_cs)
    
    return starlets

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


def calc_residual(active_set, cache, res_diff, x_diff):
    cache_idx = 0
    for xi in range(0, x_diff.shape[0]):
        for yi in range(0, x_diff.shape[1]):
            if(active_set[xi, yi] > 0.0):
                f_col = cache[cache_idx]
                cache_idx += 1
                res_diff = res_diff - f_col*(x_diff[xi, yi])
    return res_diff
                
    
def calc_cache(uv, dimensions, active_set, res):
    uv = -2j * np.pi * uv
    center_pixel = math.floor(dimensions[0] / 2.0)
    cache = np.zeros((np.count_nonzero(active_set), res.size), dtype=np.complex128)
    cache_idx = 0
    for xi in range(0, dimensions[0]):
        for yi in range(0, dimensions[1]):
            if(active_set[xi, yi] > 0.0):
                pix = np.dot(uv, np.asarray([xi-center_pixel, yi-center_pixel]))
                f_col = np.exp(pix)
                cache[cache_idx] = f_col
                cache_idx += 1
    return cache    


def to_image(starlets, equi_base):
    starlets_convolved = starlets.copy()
    for J in range(0, len(starlets)):
        x_fft = fftnumpy.fft2(starlets_convolved[J])
        res_starlet = np.real(fftnumpy.ifft2(x_fft * equi_base[J]))
        starlets_convolved[J] = res_starlet
    return starlets_convolved.sum(axis=0)


def full_algorithmdata(data, nuft, max_full, starlet_base, lambda_cs, residuals, x_starlets):
    full_cache_debug = np.zeros(data.imsize)
    print(_magnitude(residuals))
    
    for J_main in range(0, len(starlet_base)):
        #approx
        starlets = _nfft_approximation(nuft, data.imsize, starlet_base, 0.0, residuals)
        active_set, active_lambda = calc_active(starlets[J_main], max_full, lambda_cs)
        print("found active set with ", np.count_nonzero(active_set))
        active_set[active_set > 0.0] = 1
        full_cache_debug = full_cache_debug + active_set
        
        print("calculated cache")
        cache = calc_cache(data.uv, data.imsize, active_set, data.vis)
        for J in range(0, len(starlet_base)):
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
    return residuals, x_starlets, full_cache_debug