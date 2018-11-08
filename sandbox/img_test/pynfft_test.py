#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  7 13:32:28 2018

@author: jon
"""
#import sys
#from pyuvdata import UVData
from pynfft import NFFT
import numpy as np
import matplotlib.pyplot as plt

from mslib import MS_jon

UV = MS_jon()
UV.read_ms("simkat64-default.ms")