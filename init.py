#!/usr/bin/env python

import os
import time
import inspect

import numpy as np
import matplotlib.pyplot as plt

import msmhelper as mh
from MPT.MPT_MCMC_fnc import q_macrostates

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
lmpt_kernel = MPT.kernel.MPTKernel()
kmpt_kernel = MPT.kernel.KLKernel()
smpt_kernel = MPT.kernel.SMPTKernel(method="p", param=1, c=0.15)
feature_kernel = MPT.kernel.FeatureKernel(feature_traj, traj, sigma=0.05, b=2)
lfeature_kernel = MPT.kernel.FeatureKernel(feature_traj, traj, sigma=0.05, b=2)
kfeature_kernel = MPT.kernel.FeatureKernel(feature_traj, traj, sigma=0.05, b=2)

mpt = MPT.MPT(traj, lagtime)
mpt.mpt(mpt_kernel)
mpt.add_feature(feature_traj)
mpt.assign_macrostates(0.005, 0.5)
mpt.calc_timescales()

mpt_nd = MPT.MPT(traj, lagtime)
mpt_nd.mpt(mpt_kernel)
mpt_nd.add_feature(feature_traj)
mpt_nd.assign_macrostates(0.005, 0.5, dyn_correct=False)
mpt_nd.calc_timescales()

mpt_fnc = MPT.MPT(traj, lagtime)
mpt_fnc.mpt(mpt_kernel, feature_kernel=feature_kernel)
mpt_fnc.add_feature(feature_traj)
mpt_fnc.assign_macrostates(0.005, 0.5)
mpt_fnc.calc_timescales()

