#!/usr/bin/env python

import numpy as np

import MPT
from plot import plot_macro_feature

mpt_kernel = MPT.kernel.MPTKernel()

traj = np.loadtxt("/home/fg149/Dokumente/data_production/MPT/MPT/hp35.selected_contacts.gaussian10f_microstates_pcs5_p153", dtype=int)
feature_traj = np.loadtxt("/home/fg149/Dokumente/data_production/MPT/MPT/hp35.mindists2.gaussian10f.q")
lagtime = 50

out_base = "/home/fg149/Dokumente/data_production/MPT/MPT/"
i = out_base + "mf.npy"
o = out_base + "img/stoch_hist_test.pdf"

mpt = MPT.MPT(traj, lagtime)
mpt.mpt(mpt_kernel)
mpt.add_feature(feature_traj)
mpt.assign_macrostates(0.005, 0.5)


micro_features = np.load(i).T
weights = np.repeat(np.expand_dims(mpt.full_pop[0][:mpt.n_states], 1), micro_features.shape[1], axis=1)
plot_macro_feature(micro_features, o, [(
    mpt.macrostate_assignment[0],
    mpt.macrostate_feature[0],
    "r",
    "Reference",
    mpt.full_pop[0][:mpt.n_states],
)], pop=weights)
