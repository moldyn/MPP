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
kl_kernel = MPT.kernel.MPTKernel(kullback_leibler=True)
# smpt_kernel = MPT.kernel.SMPTKernel(method="p", param=1, c=0.15)
feature_kernel = MPT.kernel.FeatureKernel(feature_traj, traj, sigma=0.05, b=2)
lfeature_kernel = MPT.kernel.FeatureKernel(feature_traj, traj, sigma=0.05, b=2)
kfeature_kernel = MPT.kernel.FeatureKernel(feature_traj, traj, sigma=0.05, b=2)

mpt = MPT.MPT(traj, lagtime, feature_traj, macrostate_thresholds=(0.005, 0.5))
mpt.mpt(mpt_kernel)

mpt_fnc = MPT.MPT(traj, lagtime, feature_traj, macrostate_thresholds=(0.005, 0.5))
mpt_fnc.mpt(mpt_kernel, feature_kernel=feature_kernel)

mpt_kl = MPT.MPT(traj, lagtime, feature_traj, macrostate_thresholds=(0.005, 0.5))
mpt_kl.mpt(kl_kernel)

mpt_fnc_kl = MPT.MPT(traj, lagtime, feature_traj, macrostate_thresholds=(0.005, 0.59))
mpt_fnc_kl.mpt(kl_kernel, feature_kernel=feature_kernel)

print(mpt_fnc.timescales[0, 0] / mpt.timescales[0, 0])
print(mpt_kl.timescales[0, 0] / mpt.timescales[0, 0])
print(mpt_fnc_kl.timescales[0, 0] / mpt.timescales[0, 0])

# mpt_fnc.plot("/home/fg149/Dokumente/data_production/tmp_dev/det_fnc_tmp.pdf")
# mpt.plot("/home/fg149/Dokumente/data_production/tmp_dev/det_tmp.pdf")
# mpt_fnc_kl.plot("/home/fg149/Dokumente/data_production/tmp_dev/det_fnc_kl.pdf")
