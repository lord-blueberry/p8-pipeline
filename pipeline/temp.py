#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  7 02:56:26 2018

@author: jon
"""

idx=1
data = read_all_freq(prefix+bmark[idx]+"/simulation.ms", column[idx], size[idx], cell[idx])
nuft = nfft(data)
write_img(nuft.ifft_normalized(data.vis), bmark[idx]+"_dirty")

from algorithms.CoordinateDescent import CoordinateDescent as CD
cd_alg = CD(data,nuft , 4)


_, starlets = cd_alg._nfft_approximation(data.vis, 4.0)
img = starlets.sum(axis=0).reshape(data.imsize)
write_img(img, bmark[idx]+"_approx")

starlets_zero = cd_alg.init_zero_starlets()
residuals, starlets = cd_alg.optimize_cache(4.0, data.vis.copy(), starlets_zero)
img = starlets.sum(axis=0).reshape(data.imsize)
write_img(img, bmark[idx]+"_run1")

img = None
for i in range(0, 5):
    img = cd_alg.rerun_inner_cd_cached(4.0)
write_img(img, bmark[idx]+"_runX")

img = None
for i in range(0, 5):
    img = cd_alg.rerun_inner_cd_cached(2.0)
write_img(img, bmark[idx]+"_runX")

img = None
for i in range(0, 5):
    img = cd_alg.rerun_inner_cd_cached(1.0)
write_img(img, bmark[idx]+"_runX")


img = cd_alg.calc_residual_img(1.5)
write_img(img, bmark[idx]+"_residual")


residuals, starlets = cd_alg.optimize_cache(1.0, cd_alg.tmp_residuals, cd_alg.tmp_starlets)
img = starlets.sum(axis=0).reshape(data.imsize)
write_img(img, bmark[idx]+"_run2")



_, starlets = cd_alg._nfft_approximation(data.vis, 4.0)
img = starlets.sum(axis=0).reshape(data.imsize)
write_img(img, bmark[idx]+"_approx")

starlets_zero = cd_alg.init_zero_starlets()
residuals, starlets = cd_alg.optimize_cache(2.0, data.vis.copy(), starlets_zero)
img = starlets.sum(axis=0).reshape(data.imsize)
write_img(img, bmark[idx]+"_run1")