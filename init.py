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
multi_feature_traj = np.loadtxt("/data/evaluation/MPP/stochastic_MPP_Felix/data_source/hp35.mindist2.gaussian10f")
multi_feature_traj_bool = multi_feature_traj < 0.45
cluster_file = "/data/evaluation/MPP/stochastic_MPP_Felix/data_production/MPT/MPT/hp35.mindist2.mosaic_clusters"

out_base = "/data/evaluation/MPP/stochastic_MPP_Felix/data_production/MPT/MPT/"
out = out_base + "img/hp35_dendrogram_det_dc.pdf"
out_dev = "/home/fg149/Dokumente/data_production/tmp_dev/"
out_thesis = "/data/evaluation/MPP/stochastic_MPP_Felix/data_production/thesis/"
name = "hp35_stoch_n2_KL_JS_contacts/"

top = "/home/fg149/Dokumente/data_source/villin_crystal_number360K.pdb"
# xtc_traj = "/tmp/pnas2012-2f4k-360K-protein-fit.xtc"
xtc_traj = "/home/fg149/Dokumente/data_source/pnas2012-2f4k-360K-protein-fit.xtc"

o = "/home/fg149/Dokumente/tmp/latex/"

lagtime = 50


# feature_kernel = MPT.kernel.FeatureKernel(feature_traj, traj, sigma=0.05, b=2)
# klfeature_kernel = MPT.kernel.MultiFeatureKernel(multi_feature_traj, traj, similarity="KL")
# jsfeature_kernel = MPT.kernel.MultiFeatureKernel(multi_feature_traj_bool, traj, similarity="JS")


# k1 = MPT.kernel.MPTKernel(similarity="KL", a=0, b=1, c=1, term="*")
k1 = MPT.kernel.MPTKernel()
m1 = MPT.MPT(traj, lagtime, feature_traj, macrostate_thresholds=(0.005, 0.5))
m1.mpt(k1)
# m1.mpt(k1, feature_kernel=jsfeature_kernel)
# m1.topology_file = top
# m1.xtc_trajectory_file = xtc_traj
# helices = np.array([[3, 10], [14, 19], [22, 32]])

# name = "hp35_det_KL/"


# m1.plot(out_thesis + name + "dendrogram.pdf")
# m1.plot_implied_timescales(out_thesis + name + "timescales.pdf")
# m1.plot_sankey(out_thesis + name + "sankey.pdf")
# m1.plot_contact_rep(multi_feature_traj, cluster_file, out_thesis + name + "contact_rep.pdf")

# m1.plot(out_thesis + name + "dendrogram.pgf", scale=1)
# m1.plot_implied_timescales(out_thesis + name + "timescales.pgf", scale=1)
# m1.plot_sankey(out_thesis + name + "sankey.pgf", scale=1)
# m1.plot_contact_rep(multi_feature_traj, cluster_file, out_thesis + name + "contact_rep.pgf", scale=0.8)
# m1.plot_implied_timescales(o + "timescales_.pgf", scale=1)

# m1.plot_implied_timescales(out_dev + "ts.pdf")
# MPT.plot.plot_implied_timescales([m1.reference.macrotraj[:, 0], m1.macrotraj[:, 0]], np.arange(1, 152, 5), out_dev + "ts.pdf", first_ref=True)

# MPT.plot.report(
#     m1,
#     multi_feature_traj,
#     cluster_file,
#     out_base + "img_v2/hp35_det_KL",
#     helices,
# )


# ff = np.zeros((1093, 1093))
# ix = np.ix_(np.arange(547), np.arange(547))
# ff[ix] = m1.tmat
# m = np.zeros(1093).astype(bool)
# m[:547] = True
# s = 272
# fjs = jsfeature_kernel.apply(ff[s], s, m)
# fkl = klfeature_kernel.apply(ff[s], s, m)
# fjs.sort()
# fkl.sort()
#
# plt.plot(np.arange(547), fkl, label="KL")
# plt.plot(np.arange(547), fjs, label="JS")
# plt.legend()
# plt.show()

