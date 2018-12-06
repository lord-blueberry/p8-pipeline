#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  5 14:01:58 2018

@author: jon
"""

import numpy as np
import math

def _shrink(x, lambda_cs):
    out = np.sign(x) * np.maximum(np.abs(x) - lambda_cs, 0.)
    return np.maximum(out, 0.)

def _magnitude(vis):
    return np.sum(np.square(np.real(vis))+np.square(np.imag(vis)))

class CoordinateDescent:
    
    def __init__(self, data, nfft, starlet_levels):
        self.data = data
        self.starlet_levels = starlet_levels
        self.nfft = nfft
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
     
    
    def _nfft_approximation(self, vis, lambda_cs):
        starlets = np.zeros((self.starlet_levels + 1, self.pixel_count))
        image = np.zeros(self.data.imsize)
        for J in range(0, self.starlet_levels):
            w_J = self.nfft.ifft_normalized(vis * self.fourier_starlet_base[J])
            w_J = _shrink(w_J, lambda_cs)
            image += w_J
            starlets[J] = w_J.flatten()
        
        c_J = self.nfft.ifft_normalized(vis * self.fourier_starlet_base[self.starlet_levels])
        c_J = _shrink(c_J, lambda_cs)
        image += c_J
        starlets[self.starlet_levels] = c_J.flatten()
        
        residuals = vis - self.nfft.fft(image)
        return residuals, starlets
    
    #assumes square image
    @staticmethod
    def _CD(dimensions, uv, lambda_cs, active_set, starlet, vis_residual, x):
        x_img = np.reshape(x, dimensions)
        active_set_img = np.reshape(active_set, dimensions)
        center_pixel = math.floor(dimensions[0] / 2.0)  #
        for xi in range(0, x_img.shape[0]):
            for yi in range(0, x_img.shape[1]):
                if active_set_img[xi, yi] > 0.0:
                    x_old = x_img[xi, yi]
                    uv_prod = np.dot(uv, np.asarray([xi - center_pixel, yi - center_pixel]))
                    
                    f_col = starlet * np.exp(uv_prod)
                    f_r = np.real(f_col)
                    f_i = np.imag(f_col)
                    res_r = np.real(vis_residual)
                    res_i = np.imag(vis_residual)
                    
                    #calc -b/(2a)
                    a = np.sum(np.square(f_r) + 2*f_r*f_i + np.square(f_i))
                    b = np.sum(f_r*res_r + f_r*res_i + f_i*res_r + f_i*res_i) 
                    x_new = b / a # this times -2 actually, but it cancels out the -1/2 of -b/(2a)
                    
                    x_new = _shrink(x_new + x_old, lambda_cs)
                    diff = x_new - x_old
                    if diff != 0.0:
                        x_img[xi, yi] = x_new
                        vis_residual = vis_residual - f_col*diff
        
        return vis_residual, x_img.flatten()
    

    def optimize(self, vis, lambda_cs):
        vis_residual, x_starlets = self._nfft_approximation(vis, lambda_cs)
        print("after approx ", _magnitude(vis_residual))
        active_sets = x_starlets.copy()
        
        uv = -2j * np.pi * self.data.uv
        for J in range(0, self.starlet_levels + 1):
            print("descent starlet ", J)
            active_set = active_sets[J]
            starlet = self.fourier_starlet_base[J]
            x = x_starlets[J]
            vis_residual, x = CoordinateDescent._CD(self.data.imsize, uv, lambda_cs, active_set, starlet, vis_residual, x)
            x_starlets[J] = x
            print("descent starlet ", J, " ", _magnitude(vis_residual))
        
        return vis_residual, x_starlets




#debugging function
        
    def descent_starlets(self, uv, lambda_cs, active_sets, vis_residual, x_starlets):
        for J in range(0, self.starlet_levels + 1):
            print("descent starlet ", J)
            active_set = active_sets[J]
            starlet = self.fourier_starlet_base[J]
            x = x_starlets[J]
            vis_residual, x = CoordinateDescent._CD(self.data.imsize, uv, lambda_cs, active_set, starlet, vis_residual, x)
            x_starlets[J] = x
            print("descent starlet ", J, " ", _magnitude(vis_residual))
        return vis_residual, x_starlets
    
    def transform(self,res, starlet, lambda_cs):
        x = np.zeros(self.pixel_count).reshape(self.data.imsize)
        p = -2j * np.pi * self.data.uv
        conv = self.fourier_starlet_base[starlet]
        for xi in range(0, 32):
            print(xi)
            for yi in range(0, x.shape[1]):
                pix = np.asarray([xi-32, yi-32])
    
                f_col = conv * np.exp((-1)*np.dot(p, pix)) / res.size
                x[xi, yi] = _shrink(np.real(np.sum(f_col*res)), lambda_cs)
                
        return x