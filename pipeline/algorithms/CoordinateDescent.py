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
    out = np.sign(x) * np.maximum(np.abs(x) - lambda_cs, 0.)
    return np.maximum(out, 0.)

class CoordinateDescent:
    
    def __init__(self, data, starlet_levels, lambda_cs):
        self.data = data
        self.lambda_cs = lambda_cs
        self.starlet_levels = starlet_levels
        self.nfft = nfft(data)
        self.pixel_count = data.imsize[0]*data.imsize[1]

        self.fourier_starlet_base = CoordinateDescent._precalc_fourier_starlets(self.nfft, data, starlet_levels)
        

    #NOTE: Assumes square image dimensions
    @staticmethod
    def _precalc_fourier_starlets(nfft, data, starlet_levels):
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
        sum_total = np.zeros(data.vis.size, dtype=np.complex128)
        last_conv_mat = None
        for J in range(0, starlet_levels):
            new_mat = nfft.fft(insert_spaced(tmp.copy(), bspline, J))
            if last_conv_mat is None:
                last_conv_mat = new_mat
            else:
                last_conv_mat = last_conv_mat * new_mat
                
                
            #(1- conv_mat) arises from the multi-scale wavelet analysis
            #w_J = c_(J-1) - conv_mat_J *c_(J-1)
            four_base[J] = (1 - last_conv_mat)
            sum_total += (1 - last_conv_mat)
            
        #add cJ as last
        four_base[starlet_levels] = sum_total + last_conv_mat
        return four_base
     
    
    def _forward_approx(self, vis, lambda_cs):
        starlets = np.zeros((self.starlet_levels + 1, self.pixel_count))
        for J in range(0, self.starlet_levels):
            w_J = self.nfft.ifft_normalized(vis * self.fourier_starlet_base[J])
            starlets[J] = w_J.flatten()
        
        c_J = self.nfft.ifft_normalized(vis * self.fourier_starlet_base[self.starlet_levels])
        starlets[self.starlet_levels] = c_J.flatten()
        return starlets
    
    def _CD():
        return 0