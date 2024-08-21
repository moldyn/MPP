#!/usr/bin/env python

import os
import numpy as np

import MPT

traj = np.loadtxt("/home/fg149/Dokumente/data_production/MPT/MPT/hp35.selected_contacts.gaussian10f_microstates_pcs5_p153", dtype=int)
feature_traj = np.loadtxt("/home/fg149/Dokumente/data_production/MPT/MPT/hp35.mindists2.gaussian10f.q")
out_base = "/home/fg149/Dokumente/data_production/MPT/MPT/"

lagtime = 50
smpt_kernel = MPT.kernel.SMPTKernel(method="n", param=2)

mpt = MPT.MPT(traj, lagtime)
mpt.mpt(smpt_kernel, n=100)
mpt.add_feature(feature_traj)
mpt.assign_macrostates(0.005, 0.5)
mpt.macro_to_micro_feature()

o = out_base + "mf.npy"
if os.path.exists(o):
    ar = np.load(o)
    car = np.concatenate((ar, mpt.micro_feature.T))
else:
    car = mpt.micro_feature.T

np.save(o, car)

# with open(o, "ab") as f:
#     np.savetxt(f, mpt.micro_feature)
