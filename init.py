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
out = out_base + "img/hp35_n2_1k_its.pdf"

lagtime = 50

mpt_kernel = MPT.kernel.MPTKernel()
smpt_kernel = MPT.kernel.SMPTKernel(method="p", param=1, c=0.15)
feature_kernel = MPT.kernel.FeatureKernel(feature_traj, traj, sigma=0.05)
# fk1 = MPT.kernel.FeatureKernel(feature_traj, traj, sigma=0.13)
# fk2 = MPT.kernel.FeatureKernel(feature_traj, traj, sigma=0.1)
# fk3 = MPT.kernel.FeatureKernel(feature_traj, traj, sigma=0.05)
#feature_kernel = MPT.kernel.FeatureKernel(multi_feature_traj, traj, sigma=0.5)

mpt = MPT.MPT(traj, lagtime)
mpt.mpt(mpt_kernel)
#mpt.mpt(mpt_kernel, feature_kernel=feature_kernel)
mpt.add_feature(feature_traj)
mpt.assign_macrostates(0.005, 0.5)
mpt.calc_timescales()
# #mpt.plot(out)
#mpt.plot_tmat(out_base + "img/hp35_det_macrotmat.pdf")
#mpt.plot_tmat_times(out_base + "img/hp35_det_macrotmat_times.pdf")

# smpt = MPT.MPT(traj, lagtime)
# smpt.mpt(smpt_kernel, feature_kernel=feature_kernel, n=10)
# smpt.add_feature(feature_traj)
# smpt.assign_macrostates(0.005, 0.5)
# smpt.calc_timescales()


