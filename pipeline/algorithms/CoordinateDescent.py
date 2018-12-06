#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  5 14:01:58 2018

@author: jon
"""

from nufftwrapper import nfft_wrapper as nfft
import numpy as np
import math

def _shrink(x, lambda_cs):
    out = np.sign(x) * np.max(np.abs(x) - lambda_cs, 0.)
    return np.maximum(out, 0.)

class CoordinateDescent:
    
    def __init__(self, data, starlet_levels, lambda_cs):
        self.data = data
        self.lambda_cs = lambda_cs
        self.starlet_levels = starlet_levels
        self.nfft = nfft(data)
        
        pix_star, four_star = CoordinateDescent._precalc_operations(self.nfft, data, starlet_levels)
        self.pix_starlet_base = pix_star
        self.fourier_starlet_base = four_star
        
    
    @staticmethod
    def _precalc_operations(nfft, data, starlet_levels):
        tmp = np.asarray([1.0 / 16.0, 1.0 / 4.0, 3.0 / 8.0, 1.0 / 4.0, 1.0 / 16.0])
        row = np.asmatrix([tmp])
        bspline =(np.dot(np.transpose(row), row))
        
        starlet_base = CoordinateDescent._precalc_pixel_starlets(data.imsize, bspline, starlet_levels)
        starlet_ft = CoordinateDescent._precalc_fourier_starlets(nfft, data, bspline, starlet_levels)
        return starlet_base, starlet_ft
        
    #this method can be refactored once the data shapes are more clear. In General there will be Terabytes of Fourier components, and gigabytes of image data.
    #so every method sucks.
    @staticmethod
    def _precalc_pixel_starlets(dimensions, bspline, starlet_levels):
        def circular_convolution(size, kernel, J):
            print(size[0]*size[1], size[0]*size[1])
            output = np.zeros((size[0]*size[1], size[0]*size[1]))
            kernel = np.fliplr(np.flipud(kernel))
            
            disp = 2**J
            temp = np.zeros(size)
            for i in range(0, kernel.shape[0]):
                for j in range(0, kernel.shape[1]):
                    x = (i * disp) % temp.shape[0]
                    y = (j * disp) % temp.shape[1]
                    temp[x,y] += kernel[i,j]
                    
            mid = (kernel.shape[0]-1) * disp + 1
            temp = np.roll(temp, -(mid//2), axis= 0)
            temp = np.roll(temp, -(mid//2), axis=1)
            
            for x in range(0, size[0]):
                offset = x * size[0]
                for y in range(0, size[1]):
                    output[offset+y] = temp.flatten()
                    temp = np.roll(temp, 1, axis=1)
                temp = np.roll(temp, 1, axis=0)
            return(output)
            
        starlet_base = []
        for i in range(0, starlet_levels):
            starlet_base.append(circular_convolution(dimensions, bspline, i))
        return starlet_base

    
    #NOTE: Assumes square image dimensions
    @staticmethod
    def _precalc_fourier_starlets(nfft, data, bspline, starlet_levels):
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
        
        four_base = np.zeros((starlet_levels+1, data.vis.size), dtype=np.complex128)
        tmp = np.zeros(data.imsize)
        sum_total = np.zeros(data.vis.size, dtype=np.complex128)
        last_conv_mat = None
        for J in range(0, starlet_levels):
            new_mat = nfft.fft(insert_spaced(tmp.copy(), bspline, J))
            if last_conv_mat:
                last_conv_mat = last_conv_mat * new_mat
            else:
                last_conv_mat = new_mat
                
            #(1- conv_mat) arises from the multi-scale wavelet analysis
            #w_J = c_(J-1) - conv_mat_J *c_(J-1)
            four_base[J] = (1 - last_conv_mat)
            sum_total += (1 - last_conv_mat)
            
        #add cJ as last
        four_base[starlet_levels] = sum_total + last_conv_mat
        return four_base
     
    
    def _forward_approx(self, vis, lambda_cs):
        image = self.nfft.ifft(vis)
        last_c = image
    
        starlets = np.zeros((image.size, len(self.pix_starlet_base)+1))
        for i in range(len(self.pix_starlet_base)):
            c = np.dot(self.pix_starlet_base[i], last_c)
            w = last_c - c
            starlets[:, i] = w
            last_c = c
        starlets[:, len(self.pix_starlet_base)] = last_c
        return starlets
    
    def _CD():
        return 0