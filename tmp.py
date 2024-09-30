#!/usr/bin/env python

import os
import time
import inspect

import numpy as np
import matplotlib.pyplot as plt

# import MPT
#
# traj = np.loadtxt("/data/evaluation/MPP/stochastic_MPP_Felix/data_production/MPT/MPT/hp35.selected_contacts.gaussian10f_microstates_pcs5_p153", dtype=np.uint16)
# feature_traj = np.loadtxt("/data/evaluation/MPP/stochastic_MPP_Felix/data_production/MPT/MPT/hp35.mindists2.gaussian10f.q")
# multi_feature_traj = np.loadtxt("/data/evaluation/MPP/stochastic_MPP_Felix/data_production/MPT/MPT/hp35.mindists2")
#
# out_base = "/data/evaluation/MPP/stochastic_MPP_Felix/data_production/MPT/MPT/"
# out = out_base + "img/hp35_dendrogram_det_dc.pdf"
# ob = "/home/fg149/Dokumente/data_production/rep_lukas_fnc/fnc_weighting_function/"
#
# lagtime = 50
#
# mpt_kernel = MPT.kernel.MPTKernel()
# smpt_kernel = MPT.kernel.SMPTKernel(method="p", param=1, c=0.15)
# feature_kernel = MPT.kernel.FeatureKernel(feature_traj, traj, sigma=0.05)
# lfeature_kernel = MPT.kernel.FeatureKernel(feature_traj, traj, sigma=0.05)
#
# use_old = False
# mpt = MPT.MPT(traj, lagtime, use_old)
# mpt.mpt(mpt_kernel, feature_kernel=feature_kernel)
# mpt.add_feature(feature_traj)
# mpt.assign_macrostates(0.005, 0.5)

o = "/home/fg149/Dokumente/data_production/rep_lukas_fnc/debug/"
t_fg = np.loadtxt(o + "fg/trans")
t_ld = np.loadtxt(o + "ld/trans")
w_fg = np.loadtxt(o + "fg/wf")
w_ld = np.loadtxt(o + "ld/wf")

def get_list(t):
    l = []
    n = int(np.sqrt(t.shape[0] * 2))
    cs0 = 0
    cs1 = n
    for i in range(1, n + 1):
        l.append(np.array(t[cs0:cs1]))
        cs0 = cs1
        cs1 += n - i
    return l

def compare_lists(l1, l2):
    s = []
    for i in range(len(l1)):
        s.append((~np.isclose(np.sort(l1[i]), np.sort(l2[i]))).sum())
    return np.array(s)

def compare_lists_pr(l1, l2):
    s = []
    for i in range(len(l1)):
        s.append((~np.isclose(np.sort(l1[i]), np.sort(l2[i]), rtol=1e-07)).sum())
    return np.array(s)

l_fg = get_list(t_fg)
l_ld = get_list(t_ld)
lw_fg = get_list(w_fg)
lw_ld = get_list(w_ld)
