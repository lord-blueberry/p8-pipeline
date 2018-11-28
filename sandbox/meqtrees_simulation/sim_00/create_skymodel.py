#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 19 07:19:59 2018

@author: jon
"""

from astropy.io import fits
import numpy as np
import math
import random as rnd
import matplotlib.pyplot as plt
import os

def gaussian(A, s_x,s_y, theta):
    a = np.cos(theta)**2/(2*s_x**2) + np.sin(theta)**2/(2*s_y**2)
    b = -np.sin(2*theta)/(4*s_x**2) + np.sin(2*theta)/(4*s_y**2)
    c = np.sin(theta)**2/(2*s_x**2) + np.cos(theta)**2/(2*s_y**2)
    
    size = math.trunc(max(s_x, s_y)*3.0)
    x = np.arange(-size, size, 1)
    x = x.reshape((x.size,1))
    y = np.transpose(x)
    
    z = np.ones((x.size,1))
    alpha = np.square(np.dot(x, np.transpose(z)))
    beta= np.dot(x,y)
    gamma = np.square(np.dot(z, y))

    return A*np.exp(-(a*alpha + 2*b*beta + c*gamma))

def insert(into, mat, x0, y0):
    half = mat.shape[0]//2 #(mat) is always square and divisible by 2
    
    #assume the insert can always fit in (into)
    xL = x0-half
    xH = x0+half
    yL = y0-half
    yH = y0+half
    into[xL:xH, yL:yH] = mat


def generate():
    data = np.zeros((1080,1080))
    insert(data, gaussian(0.2, 54.2, 46.3, math.radians(30)),492,533)
    insert(data, gaussian(0.6, 25.0, 13.3, math.radians(128.3)),880,593)
    insert(data, gaussian(0.4, 23.3, 21.8, math.radians(102.3)),694,823)
    
    rnd.seed(12348)
    data[rnd.randint(100,980), rnd.randint(100,980)] = 130.0
    data[rnd.randint(100,980), rnd.randint(100,980)] = 140
    data[rnd.randint(100,980), rnd.randint(100,980)] = 150
    data[rnd.randint(100,980), rnd.randint(100,980)] = 160
    data[rnd.randint(100,980), rnd.randint(100,980)] = 170
    data[rnd.randint(100,980), rnd.randint(100,980)] = 180
    data[rnd.randint(100,980), rnd.randint(100,980)] = 190
    
    data[500,530] = 540
    data[550, 512] = 530
    data[102, 202] = 520
    data[762, 668] = 510
    data[rnd.randint(100,980), rnd.randint(100,980)] = 330
    data[rnd.randint(100,980), rnd.randint(100,980)] = 440
    data[rnd.randint(100,980), rnd.randint(100,980)] = 550
    data[rnd.randint(100,980), rnd.randint(100,980)] = 660
    data[rnd.randint(100,980), rnd.randint(100,980)] = 770
    return data
    
    
def plotData(data):
    width=data.shape[0] # pixels
    height=data.shape[1]
    margin=50 # pixels
    dpi=100. # dots per inch
    
    figsize=((width+2*margin)/dpi, (height+2*margin)/dpi) # inches
    left = margin/dpi/figsize[0] #axes ratio
    bottom = margin/dpi/figsize[1]
    
    fig = plt.figure(figsize=figsize, dpi=dpi)
    fig.subplots_adjust(left=left, bottom=bottom, right=1.-left, top=1.-bottom)
    
    plt.imshow(np.flipud(data), interpolation="bilinear")
    plt.savefig('skymodel.png')

output_fits = "skymodel.fits"
if os.path.exists(output_fits):
    os.remove(output_fits)
    
gendata = generate()
hdul = fits.open("template.fits")
hdul[0].data[0,0] = gendata
hdul.writeto(output_fits)
plotData(gendata)
