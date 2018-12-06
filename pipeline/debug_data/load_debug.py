#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  6 11:41:29 2018

@author: jon
"""

import scipy.io as sio
from InputData import InputData
import numpy as np

import os


def load_debug():
    prefix = os.path.dirname(os.path.realpath(__file__))
    y = sio.loadmat(os.path.join(prefix, "y.mat"))['y'][0,0].reshape(307780)
    p = np.asarray(sio.loadmat(os.path.join(prefix, "p.mat"))["p"])
    #dirty = np.asarray(sio.loadmat(os.path.join(prefix, "dirty.mat"))["dirty"])
    
    return InputData(y, (0.5)*p, np.ones(y.size), (64,64), (1.0, 1.0))