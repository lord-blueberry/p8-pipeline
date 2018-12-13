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

diff = current_l - current_l / 10.0
while (nonzeros > max_full or nonzeros <= 10) and diff > 0.00001:
    diff = diff/2
    if nonzeros > max_full:
        current_l = current_l + diff
    else:
        current_l = current_l - diff
    tmp = _shrink(img, current_l)
    nonzeros = np.count_nonzero(tmp)

return tmp, current_l