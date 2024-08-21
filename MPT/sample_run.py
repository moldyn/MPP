#!/usr/bin/env python3

import numpy as np
import MPT
from plot_dendrogram import plot

traj = np.loadtxt(
    "/path/to/hp35.selected_contacts.gaussian10f"
            "_microstates_pcs5_p153",
    dtype=int
)
feature_traj = np.loadtxt(
    "/path/to/hp35.mindists2.gaussian10f.q"
)

# save as .pdf
out = "/path/to/save/dendrogram"
lagtime = 50 # 10 ns
n_macrostates = 4

mpt_kernel = MPT.kernel.MPTKernel()
# sMPT kernel:
# smpt_kernel = MPT.kernel.SMPTKernel(method="n", param=2)
mpt = MPT.MPT(traj, lagtime)
# SMPT allows running multiple clusterings with the same
# parameters
# mpt = MPT.SMPT(traj, lagtime)
mpt.mpt(mpt_kernel)
mpt.set_macrostates(n_macrostates)
mpt.add_feature(feature_traj)
mpt.plot(out)

