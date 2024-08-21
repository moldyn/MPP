#!/usr/bin/env python3

import numpy as np
import MPT
from plot import plot_dendrogram
from core import assign_macrostates
import time

import prettypyplot as pplt
from matplotlib import pyplot as plt
import msmhelper as mh
from sklearn.decomposition import PCA

traj = np.loadtxt("/home/fg149/Dokumente/data_production/MPT/MPT/hp35.selected_contacts.gaussian10f_microstates_pcs5_p153", dtype=int)
feature_traj = np.loadtxt("/home/fg149/Dokumente/data_production/MPT/MPT/hp35.mindists2.gaussian10f.q")
multi_feature_traj = np.loadtxt("/home/fg149/Dokumente/data_production/MPT/MPT/hp35.mindists2")
out_base = "/home/fg149/Dokumente/data_production/MPT/MPT/"
out = out_base + "img/hp35_dendrogram_kl_fnc.pdf"
out2 = out_base + "img/hp35_dendrogram_det.pdf"
out3 = out_base + "img/hp35_dendrogram_test.pdf"
#out_it = "/home/fg149/Dokumente/data_production/MPT/MPT/hp35_implied_timescales_det.pdf"
out_it = out_base + "hp35_implied_timescales_test2.pdf"
out_sc = out_base + "img/stoch_hist.pdf"
lagtime = 50 # 10 ns
n_macrostates = 12

#Z, full_pop = MPT.mpt(traj, lagtime, kernel=MPT.smpt_kernel, method="p", param=.5)
#Z, full_pop = MPT.mpt(traj, lagtime, kernel=MPT.kernel.MPTKernel(), method="n", param=2)
#kernel = MPT.kernel.MPTKernel()
mpt_kernel = MPT.kernel.MPTKernel()
kl_kernel = MPT.kernel.KLKernel()
smpt_kernel = MPT.kernel.SMPTKernel(method="n", param=2)
smpt_kernel2 = MPT.kernel.SMPTKernel(method="p", param=0.5)
# Z, full_pop = MPT.mpt(traj, lagtime, kernel=kernel)
#Z_n = Z[:-n_macrostates+1, :2]

c1 = [2, 1, 9, 5]
c2 = [12, 13, 8, 11, 7, 10]
c3 = [18, 17, 19, 20]
c4 = [26, 24, 28]
c5 = [33, 29, 34]
c6 = [40, 39, 41, 38]
c7 = [16, 14, 15]
contacts = c1 + c2 + c3 + c4 + c5 + c6 + c7

feature_kernel = MPT.kernel.FeatureKernel(multi_feature_traj, traj, sigma=0.5)
pca_feature_kernel = MPT.kernel.PCAFeatureKernel(multi_feature_traj, traj, sigma=0.5)
kl_feature_kernel = MPT.kernel.MultiFeatureKullbackLeiblerKernel(multi_feature_traj, traj, sigma=1, features=contacts)

# pca = PCA(n_components=3)
# mft = pca.fit_transform(multi_feature_traj)

mpt = MPT.MPT(traj, lagtime)
# mpt.mpt(mpt_kernel)#, feature_kernel=pca_feature_kernel)
# mpt.mpt(mpt_kernel, feature_kernel=kl_feature_kernel)
mpt.mpt(mpt_kernel, feature_kernel=feature_kernel)
#mpt.mpt(kl_kernel)
mpt.add_feature(feature_traj)
#mpt.add_feature(multi_feature_traj)
# mpt.add_feature(mft)

# first is min population, second is min Q
mpt.assign_macrostates(0.005, 0.5)
#np.savetxt(out_base + "hp35_det_macrotraj", mpt.macrotraj, fmt="%.0f")
mpt.plot(out3)

#mpt.plot(out)
# mpt1 = MPT.MPT(traj, lagtime)
# mpt1.mpt(smpt_kernel, n=5)#, feature_kernel=feature_kernel)
# mpt1.add_feature(feature_traj)
# mpt1.assign_macrostates(0.005, 0.5)
# mpt1.macro_to_micro_feature()
# mpt1.plot_macro_feature(out_sc, [(
#     mpt.macrostate_assignment[0],
#     mpt.macrostate_feature[0],
#     "r",
#     "Reference",
# )])
#mpt1.plot_time_scales(out_it)

# Plot heatmaps comparing all stochastic clusterings
# S, n = mpt1 * mpt1
# for i in range(10):
#     d0 = ((np.diff(S[i], axis=0)) ** 2).mean() / S[i].shape[0]
#     d1 = ((np.diff(S[i], axis=1)) ** 2).mean() / S[i].shape[1]
#
#     #title = f"clustering {i}; mean {n[i]*100:.2f}"
#     #title = f"clustering {i}; mean {np.sqrt((d0+d1)/2)*100:.2f}"
#     title = f"clustering {i}; mean {np.sqrt((d0+d1)/2):.2f}"
#     MPT.plot_heatmap(S[i], out_base + f"img/h/{i}.png", title)
#     plt.close()


# mpt2 = MPT.SMPT(traj, lagtime)
# mpt2.mpt(smpt_kernel2)#, feature_kernel=feature_kernel)
# mpt2.add_feature(feature_traj)
# mpt2.assign_macrostates(0.005, 0.5)


# trajs = [mpt.traj, mpt.macrotraj]#, mpt1.macrotraj]#, mpt2.macrotraj]
# titles = ["microstate traj", "MPT"]#, "SMPTn2"]#, "SMPTp50"]
# trajs = [mpt.traj]#, mpt.macrotraj, mpt1.macrotraj]#, mpt2.macrotraj]
# titles = ["microstate traj"]#, "MPT", "SMPTn2"]#, "SMPTp50"]
lagtimes = np.arange(1, 201, 5)

#MPT.plot_implied_timescales(trajs, lagtimes, out_it, titles[1:2], first_ref=True)#, frame_length=0.2)


#pplt.use_style(figsize=(6, 2.5), latex=False, colors='pastel_autumn')

#lagtimes = np.arange(1, 150, 5)
#fig, axs = plt.subplots(2, 2)
#for ax, traj in zip(axs.flatten(), [mpt.traj, mpt.macrotraj, mpt1.macrotraj, mpt2.macrotraj]):
# for traj in [mpt.traj, mpt.macrotraj, mpt1.macrotraj, mpt2.macrotraj]:
#     it = mh.msm.implied_timescales(traj, lagtimes, ntimescales=3)
#     print(it.sum(axis=1))
#     print()

#     plot_impl_times(it, ax)
#     ax.set_yscale("log")
#     ax.set_xticklabels(["0", "10", "20", "30"])
#
# plt.show()

# smpt = MPT.SMPT(traj, lagtime)
# smpt.mpt(smpt_kernel, n=10)
# smpt.set_macrostates(n_macrostates)
# smpt.add_feature(feature_traj)

# smpt2 = MPT.SMPT(traj, lagtime)
# smpt2.mpt(smpt_kernel2, 10)
# smpt2.set_macrostates(n_macrostates - 1)
# smpt2.add_feature(feature_traj)

##### Eval stoch

#MPT.evaluate_stochastic_clustering(mpt1, mpt, out2)

#mpt.plot(out)


# implied timescales:
# mh.msm.timescales.implied_timescales([mpt2._macrotraj.astype(int), mpt3._macrotraj.astype(int), mpt._macrotraj.astype(int)], [1, 5, 20, lagtime, 200])
