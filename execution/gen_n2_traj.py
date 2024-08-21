#!/usr/bin/env python

import sys
import os
sys.path.append("/data/evaluation/MPP/stochastic_MPP_Felix/tools/MPT")

import numpy as np

import MPT


traj = np.loadtxt("/home/fg149/Dokumente/data_production/MPT/MPT/hp35.selected_contacts.gaussian10f_microstates_pcs5_p153", dtype=int)
feature_traj = np.loadtxt("/home/fg149/Dokumente/data_production/MPT/MPT/hp35.mindists2.gaussian10f.q")
out_base = "/home/fg149/Dokumente/data_production/MPT/MPT/"

lagtime = 50
smpt_kernel = MPT.kernel.SMPTKernel(method="n", param=2)

mpt = MPT.MPT(traj, lagtime)
mpt.mpt(smpt_kernel, n=50)
mpt.add_feature(feature_traj)
mpt.assign_macrostates(0.005, 0.5)

o = out_base + "macrotrajs_n2_50.npy"
if os.path.exists(o):
    ar = np.load(o)
    car = np.concatenate((ar, mpt.macrotraj.T))
else:
    car = mpt.macrotraj.T

np.save(o, car)
