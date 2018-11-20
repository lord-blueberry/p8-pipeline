#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 20 12:00:38 2018

@author: jon
"""

from astropy.io import fits
import numpy as np
import os
import math

cases_dir="cases"
for case_name in os.listdir(cases_dir):
    case_root = os.path.join(cases_dir,case_name)
    skymodel = fits.open(os.path.join(case_root, "skymodel.fits"))[0].data.reshape((1080,1080)).copy()
    
    with open(os.path.join(case_root, "results.csv"), "w+") as output:
        header = ["algorithm", "rmse", "nr_of_point_sources"]
        output.write(";".join(header) + "\n")
        
        for alg in os.listdir(os.path.join(case_root, "data")):
            file = os.path.join(case_root, "data", alg)
            alg_data = fits.open(file)[0].data.reshape((1080,1080))
            
            alg_name = os.path.splitext(alg)[0]
            rmse = math.sqrt(np.sum(np.square(alg_data - skymodel)) / alg_data.size)
            nr_point_sources = (alg_data > 50.0).sum()

            line = [alg_name, str(rmse), str(nr_point_sources)]
            output.write(";".join(line) + "\n")
            