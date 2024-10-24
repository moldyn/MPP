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
# required for contacts representation:
# multi_feature_traj = np.loadtxt(
#     "/path/to//hp35.mindists2"
# )

# save as .pdf
out = "/path/to/save/dendrogram"
lagtime = 50 # 50 frames = 10 ns

# Deterministic MPT
mpt_kernel = MPT.kernel.MPTKernel()

mpt = MPT.MPT(traj, lagtime, feature_traj)
mpt.mpt(mpt_kernel)
# mpt.plot(out)

# list of np.array of dimension [n_macrostates, n_microstates], type bool
# mpt.macrostate_assignment
#
# list of np.array of dimension [n_microstates], type int (index of macrostate for each microstate)
# mpt.macrostates_map

