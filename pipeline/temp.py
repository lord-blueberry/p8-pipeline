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
from algorithms.CoordinateDescent import _shrink as shrink
cd_alg = CD(data, nuft, 2)

def starlet_img(starlets):
    return starlets.sum(axis=0).reshape(data.imsize)

def dump_starlets(starlets, name):
    for i in range(0, starlets.shape[0]):
        write_img(starlets[i].reshape(data.imsize), name+str(i))
        
def dump_starlets(starlets, name):
    reg = 0.01
    for i in range(0, starlets.shape[0]):
        write_img(shrink(starlets[i], reg).reshape(data.imsize), name+str(i))
        reg = reg * 10

_, starlets = cd_alg._nfft_approximation(data.vis, 1.0)
write_img(starlet_img(starlets), bmark[idx]+"_approx")
dump_starlets(starlets, "1_")

starlets_zero = cd_alg.init_zero_starlets()
residuals, starlets = cd_alg.optimize_cache(6000.0, data.vis.copy(), starlets_zero)

for i in range(0, 2):
    cd_alg.rerun_inner_cd_cached(500.0 )
    
for i in range(0, 2):
    cd_alg.rerun_inner_cd_cached(250.0 )

for i in range(0, 2):
    cd_alg.rerun_inner_cd_cached(100.0)

for i in range(0, 2):
     cd_alg.rerun_inner_cd_cached(50.0)

img = cd_alg.rerun_inner_cd_cached(20.0)
write_img(img, bmark[idx]+"step_1")

write_img(cd_alg.calc_residual_img(0.0), bmark[idx]+"residual")



residuals, starlets = cd_alg.optimize_cache(3000.0, cd_alg.tmp_residuals, cd_alg.tmp_starlets)
for i in range(0, 2):
    cd_alg.rerun_inner_cd_cached(500.0 )
    
for i in range(0, 2):
    cd_alg.rerun_inner_cd_cached(250.0 )

for i in range(0, 2):
    cd_alg.rerun_inner_cd_cached(100.0)

for i in range(0, 2):
     cd_alg.rerun_inner_cd_cached(50.0)

img = cd_alg.rerun_inner_cd_cached(20.0)
write_img(img, bmark[idx]+"step_2")

    img = cd_alg.rerun_inner_cd_cached(2.0)
write_img(img, bmark[idx]+"_runX")



active = cd_alg.tmp_active.sum(axis=0).reshape(data.imsize)
write_img(active, "active")
active[active > 0] = 1
write_img(active, "active_1")
