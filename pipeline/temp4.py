#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 13 14:05:30 2018

@author: jon
"""
img = starlets[J]
start_lambda = 0.01

current_l = start_lambda
tmp = _shrink(img, current_l)
while np.count_nonzero(tmp) > max_full:
    current_l = current_l * 10.0
    tmp = _shrink(img, current_l)

nonzeros = np.count_nonzero(tmp)
if nonzeros < max_full and nonzeros >= 10:
    return tmp, current_l

current_l = save_l
lower_l = current_l / 10.0
diff = current_l - lower_l
while (nonzeros > max_full or nonzeros <= 10) and diff > 0.0001:
    diff = diff/2
    if nonzeros > max_full:
        current_l = current_l + diff
        print("raise lambda", current_l)   

    else:
        current_l = current_l - diff
        print("lower lambda", current_l)
    
         
    tmp = _shrink(img, current_l)
    nonzeros = np.count_nonzero(tmp)
    print("nonzeros", nonzeros)

return tmp, lower_l