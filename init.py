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
multi_feature_traj = np.loadtxt("/data/evaluation/MPP/stochastic_MPP_Felix/data_production/MPT/MPT/hp35.mindists2")
cluster_file = "/data/evaluation/MPP/stochastic_MPP_Felix/data_production/MPT/MPT/hp35.mindist2.mosaic_clusters"

out_base = "/data/evaluation/MPP/stochastic_MPP_Felix/data_production/MPT/MPT/"
out = out_base + "img/hp35_dendrogram_det_dc.pdf"
ob = "/home/fg149/Dokumente/data_production/rep_lukas_fnc/fnc_weighting_function/"
toy = "/home/fg149/Dokumente/data_production/rep_lukas_fnc/toy1/"

top = "/home/fg149/Dokumente/data_source/villin_crystal_number360K.pdb"
# xtc_traj = "/tmp/pnas2012-2f4k-360K-protein-fit.xtc"
xtc_traj = "/home/fg149/Dokumente/data_source/pnas2012-2f4k-360K-protein-fit.xtc"

lagtime = 50

# traj = np.loadtxt(toy + "traj")
# feature_traj = np.loadtxt(toy + "feature_traj")
# lagtime = 1

mpt_kernel = MPT.kernel.MPTKernel()
# lmpt_kernel = MPT.kernel.MPTKernel()

kl_kernel = MPT.kernel.MPTKernel(similarity="KL", b=10.75, c=47.5)
js_kernel = MPT.kernel.MPTKernel(similarity="JS", b=30, c=0)
# js_kernel = MPT.kernel.MPTKernel(similarity="JS", a=p_js["a"], b=p_js["b"], c=p_js["c"], e=p_js["e"], f=p_js["f"])
# smpt_kernel = MPT.kernel.SMPTKernel(method="p", param=1, c=0.15)

feature_kernel = MPT.kernel.FeatureKernel(feature_traj, traj, sigma=0.05, b=2)
klfeature_kernel = MPT.kernel.MultiFeatureKernel(multi_feature_traj, traj, similarity="KL")
jsfeature_kernel = MPT.kernel.MultiFeatureKernel(multi_feature_traj, traj, similarity="JS")

# mpt = MPT.MPT(traj, lagtime, feature_traj, macrostate_thresholds=(0.005, 0.5))
# mpt.mpt(mpt_kernel)
#
# mpt_fnc = MPT.MPT(traj, lagtime, feature_traj, macrostate_thresholds=(0.005, 0.5))
# mpt_fnc.mpt(mpt_kernel, feature_kernel=feature_kernel)
#
# mpt_kl = MPT.MPT(traj, lagtime, feature_traj, macrostate_thresholds=(0.005, 0.5))
# mpt_kl.mpt(kl_kernel, feature_kernel=klfeature_kernel)
#
# mpt_js = MPT.MPT(traj, lagtime, feature_traj, macrostate_thresholds=(0.005, 0.5))
# mpt_js.mpt(js_kernel)

# k1 = MPT.kernel.MPTKernel(similarity="JS", b=55, c=0)
# k1 = MPT.kernel.MPTKernel(similarity="KL", b=11, c=0)
k1 = MPT.kernel.MPTKernel(similarity="KL", b=1, c=0)
k1.a = 0
# k1 = MPT.kernel.MPTKernel()
# k2 = MPT.kernel.MPTKernel(similarity="JS", b=70.1, c=92.5)
m1 = MPT.MPT(traj, lagtime, feature_traj, macrostate_thresholds=(0.005, 0.5))
# m2 = MPT.MPT(traj, lagtime, feature_traj, macrostate_thresholds=(0.005, 0.5))
m1.mpt(k1)
# m1.mpt(k1, feature_kernel=klfeature_kernel)
# m2.mpt(k2, feature_kernel=jsfeature_kernel)
# MPT.plot.report(m1, multi_feature_traj, cluster_file, "/home/fg149/Dokumente/data_production/tmp_dev/r1")
# MPT.plot.report(m2, multi_feature_traj, cluster_file, "/home/fg149/Dokumente/data_production/tmp_dev/r2")
# m1.print_rel(multi_feature_traj)
# m2.print_rel(multi_feature_traj)
m1.topology_file = top
m1.xtc_trajectory_file = xtc_traj
helices = np.array([[3, 10], [14, 19], [22, 32]])
# m2.topology_file = top
# m2.xtc_trajectory_file = xtc_traj


MPT.plot.report(
    m1,
    multi_feature_traj,
    cluster_file,
    out_base + "img_v2/hp35_det_KL",
    helices,
)
