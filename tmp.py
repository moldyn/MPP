#!/usr/bin/env python

import os
import time
import inspect

import numpy as np
import matplotlib.pyplot as plt

import MPT

traj = np.loadtxt("/data/evaluation/MPP/stochastic_MPP_Felix/data_production/MPT/MPT/hp35.selected_contacts.gaussian10f_microstates_pcs5_p153", dtype=np.uint16)
feature_traj = np.loadtxt("/data/evaluation/MPP/stochastic_MPP_Felix/data_production/MPT/MPT/hp35.mindists2.gaussian10f.q")
multi_feature_traj = np.loadtxt("/data/evaluation/MPP/stochastic_MPP_Felix/data_production/MPT/MPT/hp35.mindists2")

out_base = "/data/evaluation/MPP/stochastic_MPP_Felix/data_production/MPT/MPT/"
out = out_base + "img/hp35_dendrogram_det_dc.pdf"
ob = "/home/fg149/Dokumente/data_production/rep_lukas_fnc/fnc_weighting_function/"

lagtime = 50

mpt_kernel = MPT.kernel.MPTKernel()
smpt_kernel = MPT.kernel.SMPTKernel(method="p", param=1, c=0.15)
feature_kernel = MPT.kernel.FeatureKernel(feature_traj, traj, sigma=0.05)
lfeature_kernel = MPT.kernel.FeatureKernel(feature_traj, traj, sigma=0.05)

use_old = False
mpt = MPT.MPT(traj, lagtime, use_old)
mpt.mpt(mpt_kernel, feature_kernel=feature_kernel)
mpt.add_feature(feature_traj)
mpt.assign_macrostates(0.005, 0.5)
