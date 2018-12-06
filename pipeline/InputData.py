#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  5 14:16:17 2018

@author: jon
"""

class InputData:
    def __init__(self, vis, uv, weights, imsize, cellSize ):
        self.vis = vis
        self.uv = uv
        self.weights = weights
        self.imsize = imsize
        self.cellSize = cellSize
        self.stokes = "I" # Currently always intensity image
        
