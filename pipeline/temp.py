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
        
def dump_starlets_weird(starlets, name):
    reg = 0.01
    for i in range(0, starlets.shape[0]):
        write_img(shrink(starlets[i], reg).reshape(data.imsize), name+str(i))
        reg = reg * 10

starlet_lvl = 2
lamb = 5.0
_, starlets = cd_alg._nfft_approximation(data.vis, lamb)
write_img(starlets[starlet_lvl].reshape(data.imsize), bmark[idx]+"_approx")

starlets_zero = cd_alg.init_zero_starlets()
vis_residual, active_set, cache, x_starlet = cd_alg.optimiz_single(lamb, starlet_lvl, data.vis.copy(), starlets_zero)

vis_residual, x_starlet = cd_alg.rerun_single(1000.00, active_set, cache, vis_residual, x_starlet)
vis_residual, x_starlet = cd_alg.rerun_single(1.00, active_set, cache, vis_residual, x_starlet)
for i in range(0, 10):
    vis_residual, x_starlet = cd_alg.rerun_single(1.0, active_set, cache, vis_residual, x_starlet)
    
write_img(x_starlet.reshape(data.imsize), bmark[idx]+"_run1")
img = x_starlet.reshape(data.imsize)
print(np.max(img))
print(img[256,219])


    
lamb = 1.0
_, starlets = cd_alg._nfft_approximation(vis_residual, lamb)
write_img(starlets[starlet_lvl].reshape(data.imsize), bmark[idx]+"_approx2")
starlets_zero[starlet_lvl] = x_starlet
vis_residual, active_set, cache, x_starlet = cd_alg.optimiz_single(lamb, starlet_lvl, vis_residual, starlets_zero)
vis_residual, x_starlet = cd_alg.rerun_single(1000.00, active_set, cache, vis_residual, x_starlet)
vis_residual, x_starlet = cd_alg.rerun_single(100.00, active_set, cache, vis_residual, x_starlet)
vis_residual, x_starlet = cd_alg.rerun_single(10.00, active_set, cache, vis_residual, x_starlet)
vis_residual, x_starlet = cd_alg.rerun_single(1.00, active_set, cache, vis_residual, x_starlet)
vis_residual, x_starlet = cd_alg.rerun_single(0.1, active_set, cache, vis_residual, x_starlet)
vis_residual, x_starlet = cd_alg.rerun_single(0.01, active_set, cache, vis_residual, x_starlet)
vis_residual, x_starlet = cd_alg.rerun_single(0.005, active_set, cache, vis_residual, x_starlet)
vis_residual, x_starlet = cd_alg.rerun_single(0.005, active_set, cache, vis_residual, x_starlet)

write_img(x_starlet.reshape(data.imsize), bmark[idx]+"_run2")

for i in range(0, 10):
    vis_residual, x_starlet = cd_alg.rerun_single(0.005, active_set, cache, vis_residual, x_starlet)

for i in range(0, 2):
    cd_alg.rerun_inner_cd_cached(250.0 )

for i in range(0, 2):
    cd_alg.rerun_inner_cd_cached(100.0)

for i in range(0, 2):
     cd_alg.rerun_inner_cd_cached(50.0)

img = cd_alg.rerun_inner_cd_cached(20.0)
write_img(img, bmark[idx]+"step_1")

write_img(cd_alg.calc_residual_img(0.0, vis_residual), bmark[idx]+"residual")




active = cd_alg.tmp_active.sum(axis=0).reshape(data.imsize)
write_img(active, "active")
active[active > 0] = 1
write_img(active, "active_1")
