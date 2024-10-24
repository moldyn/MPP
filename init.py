#!/usr/bin/env python

import os
import time
import inspect

import numpy as np
import matplotlib.pyplot as plt

import msmhelper as mh

import MPT



traj = np.loadtxt("/data/evaluation/MPP/stochastic_MPP_Felix/data_production/MPT/MPT/hp35.selected_contacts.gaussian10f_microstates_pcs5_p153", dtype=np.uint16)
feature_traj = np.loadtxt("/data/evaluation/MPP/stochastic_MPP_Felix/data_production/MPT/MPT/hp35.mindists2.gaussian10f.q")
multi_feature_traj = np.loadtxt("/data/evaluation/MPP/stochastic_MPP_Felix/data_production/MPT/MPT/hp35.mindists2")
cluster_file = "/data/evaluation/MPP/stochastic_MPP_Felix/data_production/MPT/MPT/hp35.mindist2.mosaic_clusters"

out_base = "/data/evaluation/MPP/stochastic_MPP_Felix/data_production/MPT/MPT/"
out = out_base + "img/hp35_dendrogram_det_dc.pdf"
ob = "/home/fg149/Dokumente/data_production/rep_lukas_fnc/fnc_weighting_function/"
toy = "/home/fg149/Dokumente/data_production/rep_lukas_fnc/toy1/"

lagtime = 50

# traj = np.loadtxt(toy + "traj")
# feature_traj = np.loadtxt(toy + "feature_traj")
# lagtime = 1

mpt_kernel = MPT.kernel.MPTKernel()
# lmpt_kernel = MPT.kernel.MPTKernel()
#kl_kernel = MPT.kernel.KLKernel()
p_kl = {
    "a": 0.2112667860907813,
    "b": 0.5136487914187732,
    "c": 0.7643062872125638,
    "e": 4.47446985609192,
    "f": 3.969819128552328,
}

p_js = {
    "a": -0.3993436536464809,
    "b": 0.996549865269283,        
    "c": 0.3260350317329467,
    "e": 18.078028246050454,
    "f": 3.436942994680956,
}

p_js_4827 = {
    "a": -0.45076656088480827,
    "b": 0.5886234915324093,
    "c": 0.12830697697407034,
    "e": 7.9914733535407,
    "f": 10.762802821534988,
}

p_js_4875 = {
    "a": -0.31390263446885075,
    "b": 0.7685878238249264,
    "c": 0.13041883715594693,
    "e": 5.682928465388565,
    "f": 1.4272292032636014,
}

p_js_loss = {
    "a": 0.6061912866743351,
    "b": 0.5945954305784469,
    "c": 0.9815886167424418,
    "e": 14.666986771456534,
    "f": 5.681534627126421,
}

kl_kernel = MPT.kernel.MPTKernel(similarity="KL", a=p_kl["a"], b=p_kl["b"], c=p_kl["c"], e=p_kl["e"], f=p_kl["f"])
# js_kernel = MPT.kernel.MPTKernel(similarity="JS", a=p_js["a"], b=p_js["b"], c=p_js["c"], e=p_js["e"], f=p_js["f"])
js_kernel = MPT.kernel.MPTKernel(
    similarity="JS",
    a=p_js_4875["a"],
    b=p_js_4875["b"],
    c=p_js_4875["c"],
    e=p_js_4875["e"],
    f=p_js_4875["f"]
)
js2_kernel = MPT.kernel.MPTKernel(
    similarity="JS",
    a=p_js_loss["a"],
    b=p_js_loss["b"],
    c=p_js_loss["c"],
    e=p_js_loss["e"],
    f=p_js_loss["f"]
)
# smpt_kernel = MPT.kernel.SMPTKernel(method="p", param=1, c=0.15)
lfeature_kernel = MPT.kernel.FeatureKernel(feature_traj, traj, sigma=0.05, b=2)
feature_kernel = MPT.kernel.FeatureKernel(multi_feature_traj, traj, sigma=0.05, b=2)
kfeature_kernel = MPT.kernel.FeatureKernel(multi_feature_traj, traj, sigma=0.05, b=2)

mpt = MPT.MPT(traj, lagtime, feature_traj, macrostate_thresholds=(0.005, 0.5))
mpt.mpt(mpt_kernel)

mpt_fnc = MPT.MPT(traj, lagtime, feature_traj, macrostate_thresholds=(0.005, 0.5))
mpt_fnc.mpt(mpt_kernel, feature_kernel=feature_kernel)
# mpt_fnc.mpt(MPT.kernel.MPTKernel(a=0, b=0, c=1), feature_kernel=lfeature_kernel)

# mpt_kl = MPT.MPT(traj, lagtime, feature_traj, macrostate_thresholds=(0.005, 0.5))
# mpt_kl.mpt(kl_kernel)
#
# mpt_fnc_kl = MPT.MPT(traj, lagtime, feature_traj, macrostate_thresholds=(0.005, 0.5))
# mpt_fnc_kl.mpt(kl_kernel, feature_kernel=feature_kernel)
#
# mpt_js = MPT.MPT(traj, lagtime, feature_traj, macrostate_thresholds=(0.005, 0.5))
# mpt_js.mpt(js_kernel)
#
# mpt_fnc_js = MPT.MPT(traj, lagtime, feature_traj, macrostate_thresholds=(0.005, 0.5))
# mpt_fnc_js.mpt(js_kernel, feature_kernel=feature_kernel)
#
# mpt_js2 = MPT.MPT(traj, lagtime, feature_traj, macrostate_thresholds=(0.005, 0.5))
# mpt_js2.mpt(js2_kernel)
#
# mpt_fnc_js2 = MPT.MPT(traj, lagtime, feature_traj, macrostate_thresholds=(0.005, 0.5))
# mpt_fnc_js2.mpt(js2_kernel, feature_kernel=feature_kernel)

# print(mpt_fnc.timescales[0, 0] / mpt.timescales[0, 0])
# print("KL")
# print(mpt_kl.timescales[0, 0] / mpt.timescales[0, 0])
# print(mpt_fnc_kl.timescales[0, 0] / mpt.timescales[0, 0])
# print("JS")
# print(mpt_js.timescales[0, 0] / mpt.timescales[0, 0])
# print(mpt_fnc_js.timescales[0, 0] / mpt.timescales[0, 0])
# print("JS2")
# print(mpt_js2.timescales[0, 0] / mpt.timescales[0, 0])
# print(mpt_fnc_js2.timescales[0, 0] / mpt.timescales[0, 0])

# mpt_fnc.plot("/home/fg149/Dokumente/data_production/tmp_dev/det_fnc_tmp.pdf")
# mpt.plot("/home/fg149/Dokumente/data_production/tmp_dev/det_tmp.pdf")
# mpt_fnc_kl.plot("/home/fg149/Dokumente/data_production/tmp_dev/det_fnc_kl.pdf")
