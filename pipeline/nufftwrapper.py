#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  6 08:54:05 2018

@author: jon
"""

from pynfft import NFFT
import numpy as np

class nfft_wrapper:
    def __init__(self, data):
        self.normFactor = data.vis.size * 2.0 #SORT OF normalization. Weights are currently ignored.
        
        #create two pynfft objects, one for fft and one for ifft.
        #in theory one object should suffice, but the implementation tends to crash
        ifft_obj = NFFT(data.imsize, data.uv.shape[0])
        ifft_obj.x = data.uv.flatten()
        ifft_obj.precompute()
        self.ifft_obj = ifft_obj
        fft_obj = NFFT(data.imsize, data.uv.shape[0])
        fft_obj.x = data.uv.flatten()
        fft_obj.precompute()
        self.fft_obj = fft_obj
        
    def ifft(self, vis):
        self.ifft_obj.f = vis
        #needs a copy because it is a reference to the nfft's memory. Another call to ifft() would destroy the data
        return self.ifft_obj.adjoint().copy()  
    
    def ifft_normalized(self, vis):
        tmp = self.ifft(vis)
        return tmp / self.normFactor
        
    def fft(self, image):
        self.fft_obj.f_hat = image
        return self.fft_obj.trafo().copy()