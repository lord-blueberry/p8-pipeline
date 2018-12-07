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
     
    def init_zero_starlets(self):
        return np.zeros((self.starlet_levels + 1, self.pixel_count))
    
    def _nfft_approximation(self, vis, lambda_cs):
        starlets = self.init_zero_starlets()
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
 
    #assumes square image
    @staticmethod
    def _CD_cache(dimensions, uv, lambda_cs, active_set, starlet, vis_residual, x):
        x_img = np.reshape(x, dimensions)
        active_set_img = np.reshape(active_set, dimensions)
        center_pixel = math.floor(dimensions[0] / 2.0)
        
        cache = np.zeros((np.count_nonzero(active_set), vis_residual.size), dtype=np.complex128)
        cache_idx = 0
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
                    
                    cache[cache_idx] = f_col
                    cache_idx += 1
                    
                    #calc -b/(2a)
                    a = np.sum(np.square(f_r) + 2*f_r*f_i + np.square(f_i))
                    b = np.sum(f_r*res_r + f_r*res_i + f_i*res_r + f_i*res_i) 
                    x_new = b / a # this times -2 actually, but it cancels out the -1/2 of -b/(2a)
                    
                    x_new = _shrink(x_new + x_old, lambda_cs)
                    diff = x_new - x_old
                    if diff != 0.0:
                        x_img[xi, yi] = x_new
                        vis_residual = vis_residual - f_col*diff
        
        return vis_residual, x_img.flatten(), cache
    
    #assumes square image
    @staticmethod
    def _CD_cached(dimensions, lambda_cs, active_set, cache, vis_residual, x):
        x_img = np.reshape(x, dimensions)
        active_set_img = np.reshape(active_set, dimensions)
        
        cache_idx = 0
        for xi in range(0, x_img.shape[0]):
            for yi in range(0, x_img.shape[1]):
                if active_set_img[xi, yi] > 0.0:
                    x_old = x_img[xi, yi]
                    
                    f_col = cache[cache_idx]
                    f_r = np.real(f_col)
                    f_i = np.imag(f_col)
                    res_r = np.real(vis_residual)
                    res_i = np.imag(vis_residual)
                    cache_idx += 1
                    
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
    
    
    
    
    
    

    def optimize(self, lambda_cs, vis, x_starlet_init):
        vis_residual, x_starlets = self._nfft_approximation(vis, lambda_cs)
        active_sets = x_starlets.copy()
        print("after approx ", _magnitude(vis_residual))
        x_starlets = x_starlet_init + x_starlets
        
        uv = -2j * np.pi * self.data.uv
        for J in range(0, self.starlet_levels + 1):
            print("descent starlet ", J)
            active_set = active_sets[J]
            starlet = self.fourier_starlet_base[J]
            x = x_starlets[J]
            vis_residual, x = CoordinateDescent._CD(self.data.imsize, uv, lambda_cs, active_set, starlet, vis_residual, x)
            x_starlets[J] = x
            print("descent starlet ", J, " ", _magnitude(vis_residual))
        
        self.tmp_residuals = self.calc_accurate_residual_slow(self.data.imsize, active_sets, uv, starlets, vis_residual, x_starlets - x_starlet_init)
        print("accurate residuals ", _magnitude(self.tmp_residuals))
        self.tmp_starlets = x_starlets
        self.tmp_active = active_sets
        return vis_residual, x_starlets
    
    def calc_accurate_residual_slow(self, dimensions, active_sets, uv, starlets, vis, x_starlets):
        vis_residual = vis
        for J in range(0, self.starlet_levels + 1):
            active_set = active_sets[J]
            starlet = starlets[J]
            x = x_starlets[J]
            x_img = np.reshape(x, self.data.imsize)
            active_set_img = np.reshape(active_set, self.data.imsize)
            center_pixel = math.floor(dimensions[0] / 2.0)
            
            for xi in range(0, x_img.shape[0]):
                for yi in range(0, x_img.shape[1]):
                    if active_set_img[xi, yi] > 0.0:                        
                        uv_prod = np.dot(uv, np.asarray([xi - center_pixel, yi - center_pixel]))
                    
                        f_col = starlet * np.exp(uv_prod)
                        vis_residual = vis_residual - f_col*x_img[xi,yi]
        return vis_residual
    
    
    def optimize_cache(self, lambda_cs, vis_residual, x_starlet_init):
        vis_residual_approx, x_starlets = self._nfft_approximation(vis_residual, lambda_cs)
        active_sets = x_starlets.copy()
        print("after approx ", _magnitude(vis_residual))
        x_starlets = x_starlet_init + x_starlets
        
        caches = []
        uv = -2j * np.pi * self.data.uv
        for J in range(0, self.starlet_levels + 1):
            active_set = active_sets[J]
            starlet = self.fourier_starlet_base[J]
            x = x_starlets[J]
            vis_residual_approx, x, cache = CoordinateDescent._CD_cache(self.data.imsize, uv, lambda_cs, active_set, starlet, vis_residual_approx, x)
            caches.append(cache)
            x_starlets[J] = x
            print("descent starlet ", J, " ", _magnitude(vis_residual))
        
        
        self.tmp_residuals = self.calc_accurate_residual(self.data.imsize, active_sets, caches, vis_residual, x_starlets - x_starlet_init)
        print("accurate residuals ", _magnitude(self.tmp_residuals))
        self.tmp_starlets = x_starlets
        self.tmp_active = active_sets
        self.tmp_caches = caches
        return vis_residual, x_starlets
    
    def calc_accurate_residual(self, dimensions, active_sets, caches, vis, x_starlets):
        vis_residual = vis
        for J in range(0, self.starlet_levels + 1):
            active_set = active_sets[J]
            x = x_starlets[J]
            x_img = np.reshape(x, self.data.imsize)
            active_set_img = np.reshape(active_set, self.data.imsize)
            
            cache = caches[J]
            cache_idx = 0
            for xi in range(0, x_img.shape[0]):
                for yi in range(0, x_img.shape[1]):
                    if active_set_img[xi, yi] > 0.0:                        
                        f_col = cache[cache_idx]
                        cache_idx += 1
                        vis_residual = vis_residual - f_col*x_img[xi,yi]
        return vis_residual
            


    def calc_residual_img(self, lambda_cs):
        vis_residual = self.tmp_residuals
        vis_residual, x_starlets_approx = self._nfft_approximation(vis_residual, lambda_cs)
        
        return np.reshape(x_starlets_approx.sum(axis=0), self.data.imsize)
        

#debugging function

    def rerun_approx(self, lambda_cs):
        vis_residual = self.tmp_residuals
        vis_residual, x_starlets_approx = self._nfft_approximation(vis_residual, lambda_cs)
        
        x_starlets = self.tmp_starlets + x_starlets_approx
        
        self.tmp_residuals = vis_residual
        self.tmp_starlets = x_starlets
        self.tmp_active = x_starlets_approx.copy()
        return np.reshape(x_starlets.sum(axis=0), self.data.imsize)
        

    def rerun_inner_cd(self, lambda_cs):
        vis_residual = self.tmp_residuals
        x_starlets = self.tmp_starlets
        active_sets = self.tmp_active
        
        uv = -2j * np.pi * self.data.uv
        for J in range(0, self.starlet_levels + 1):
            print("descent starlet ", J)
            active_set = active_sets[J]
            starlet = self.fourier_starlet_base[J]
            x = x_starlets[J]
            vis_residual, x = CoordinateDescent._CD(self.data.imsize, uv, lambda_cs, active_set, starlet, vis_residual, x)
            x_starlets[J] = x
            print("descent starlet ", J, " ", _magnitude(vis_residual))
        
        self.tmp_residuals = vis_residual
        self.tmp_starlets = x_starlets
        self.tmp_active = active_sets
        
        return np.reshape(x_starlets.sum(axis=0), self.data.imsize)
    
    
    def rerun_inner_cd_cached(self, lambda_cs):
        vis_residual = self.tmp_residuals
        x_starlets = self.tmp_starlets
        active_sets = self.tmp_active
        caches = self.tmp_caches
        
        for J in range(0, self.starlet_levels + 1):
            print("descent starlet ", J)
            cache = caches[J]
            active_set = active_sets[J]
            starlet = self.fourier_starlet_base[J]
            x = x_starlets[J]
            vis_residual, x = CoordinateDescent._CD_cached(self.data.imsize, lambda_cs, active_set, cache, vis_residual, x)
            x_starlets[J] = x
            print("descent starlet ", J, " ", _magnitude(vis_residual))
        
        self.tmp_residuals = vis_residual
        self.tmp_starlets = x_starlets
        self.tmp_active = active_sets
        
        return np.reshape(x_starlets.sum(axis=0), self.data.imsize)
        






















        
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