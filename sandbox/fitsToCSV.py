from astropy.io import fits
import numpy as np
import math
import random as rnd
import matplotlib.pyplot as plt
import os

hdul = fits.open("template.fits")
data = hdul[0].data[0,0]

np.savetxt("tclean.pb", data, delimiter=",")
