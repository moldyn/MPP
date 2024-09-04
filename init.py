#!/usr/bin/env python

import os
import time
import inspect

import numpy as np
import matplotlib.pyplot as plt

import msmhelper as mh
from MPT.MPT_MCMC_fnc import q_macrostates

import MPT

def print_m(m):
    s = "  "
    for i in range(m.shape[1]):
        if (i-5)%10 == 0:
            s += "5"
        elif i % 10 == 0:
            if i != 0:
                s += str(i)[:-1]
            else:
                s += "0"
        else:
            s += " "
    print(s)
            
    for k, i in enumerate(m):
        s = f"{k} "
        for j in i:
            if j:
                s += "|"
            else:
                s += "-"
        print(s)

def ppops(m, p):
    for l, k in enumerate(m):
        sm = 0
        cur = 0
        prev = 0
        for i, s in enumerate(k):
            prev = cur
            cur = s
            if cur:
                sm += p[i]
            elif prev and not cur:
                print(f"{l}-{i}: {sm}")
                sm = 0
    print(f"{l}-{i}: {sm}")

def ra(pop, macrostate_assignment):
    indices_to_exclude = set()
    for mi, m in enumerate(macrostate_assignment.astype(bool)):
        print(f"macrostate {mi}")
        prev = 0
        cur = 0
        max_idx = set()
        cur_idx = set()
        for i, s in enumerate(m):
            prev = cur
            cur = s
            if cur:
                cur_idx.add(i)
            if (prev and not cur) or (cur and i + 1 == m.shape[0]):
                if pop[list(cur_idx)].sum() > pop[list(max_idx)].sum():
                    indices_to_exclude.update(max_idx)
                    max_idx = cur_idx
                    print(f"max_idx: {max_idx}: {pop[list(max_idx)].sum()}")
                else:
                    indices_to_exclude.update(cur_idx)
                cur_idx = set()

            # if len(cur_idx) > 0:
            #     print(f"{cur_idx}: {pop[list(cur_idx)].sum()}")
        print()

    print(f"indices to exclude: {indices_to_exclude}")
    return list(indices_to_exclude)

def merge_states(f, p, states):
    return (
        f[states] * p[states]
    ) / p[states].sum()


traj = np.loadtxt("/data/evaluation/MPP/stochastic_MPP_Felix/data_production/MPT/MPT/hp35.selected_contacts.gaussian10f_microstates_pcs5_p153", dtype=np.uint16)
feature_traj = np.loadtxt("/data/evaluation/MPP/stochastic_MPP_Felix/data_production/MPT/MPT/hp35.mindists2.gaussian10f.q")
multi_feature_traj = np.loadtxt("/data/evaluation/MPP/stochastic_MPP_Felix/data_production/MPT/MPT/hp35.mindists2")

out_base = "/data/evaluation/MPP/stochastic_MPP_Felix/data_production/MPT/MPT/"
out = out_base + "img/hp35_dendrogram_det_dc.pdf"
ob = "/home/fg149/Dokumente/data_production/rep_lukas_fnc/fnc_weighting_function/"

lagtime = 50

mpt_kernel = MPT.kernel.MPTKernel()
smpt_kernel = MPT.kernel.SMPTKernel(method="p", param=1, c=0.15)
feature_kernel = MPT.kernel.FeatureKernel(feature_traj, traj, sigma=0.05)
lfeature_kernel = MPT.kernel.FeatureKernel(feature_traj, traj, sigma=0.05)

use_old = False
mpt = MPT.MPT(traj, lagtime, use_old)
# mpt.mpt(mpt_kernel)
mpt.mpt(mpt_kernel, feature_kernel=feature_kernel)
mpt.add_feature(feature_traj)
#mpt.assign_macrostates(0.005, 0.5, dyn_correct=True)
mpt.assign_macrostates(0.005, 0.5)
# mpt.plot(ob + "ld_det.pdf")

use_old = True
lmpt = MPT.MPT(traj, lagtime, use_old)
# mpt.mpt(mpt_kernel)
lmpt.mpt(mpt_kernel, feature_kernel=lfeature_kernel)
lmpt.add_feature(feature_traj)
#mpt.assign_macrostates(0.005, 0.5, dyn_correct=True)
lmpt.assign_macrostates(0.005, 0.5)
lmpt.plot(ob + "hp35_det_lukas_updated.pdf")

f = feature_kernel.full_feature[mpt.n_states:]
fl = lfeature_kernel.full_feature[mpt.n_states:]
# print(f[:10] - fl[:10])

# o = [l.name for l in mpt.tree[0].leaves]
# ma = mpt.macrostate_assignment[0][:, o]
# ro = np.arange(mpt.n_states)[o]
# ieo = MPT.utils.get_microstates_to_reassign(mpt.full_pop[0], ma)
# ie = ro[ieo]



#ma[:, ie] = 0
#nma, inter_ma, inter_tmat = MPT.utils.reassign_states(mpt.tmat, mpt.full_pop[0, :mpt.n_states], ma)


# n_macrostates = mpt.n_macrostates[0]
# ma = np.array([np.where(mpt.macrostate_assignment[0][:, i])[0][0]+1 for i in range(547)])[o]
# mi = np.arange(0, 547)
# pops = mpt.full_pop[0][:mpt.n_states][o]
# dcm = mpp_plus_dyn_cor(macrostates=ma, microstates=mi, n_macrostates=n_macrostates, pops=pops, traj=traj, tlag=lagtime)
# print(dcm)
#
# mmpt = mpt
# mmpt.assign_macrostates(0.005, 0.5, dyn_correct=True)
# mmpt.calc_timescales()
# print(mmpt.timescales[0])
# madc = np.array([np.where(mmpt.macrostate_assignment[0][:, i])[0][0]+1 for i in range(547)])[o]
# print(madc)


# mpt.assign_macrostates(0.005, 0.5, dyn_correct=True)
# mpt.plot("/home/fg149/Dokumente/data_production/tmp_dev/with_dc.pdf")
# ma2 = mpt.macrostate_assignment[0]

# #p = np.random.randint(1, 1000, 20)
# p = np.array([
#     364, 989, 812, 482, 406, 804, 608, 592,  11, 583,
#     158, 717, 163, 734,  25, 626, 443, 144, 387,  63
# ])
# s = np.random.randint(0, 3, 20)
# # m = np.zeros((3, 20))
# # m[s, np.arange(m.shape[1])] = 1
# # m = m.astype(bool)
# m = np.array([
#     [1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1],
#     [0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0],
#     [0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0],
# ])

# NOTE:
# Analyisis of macrostates.py/mpp_plus_dyn_cor
# - macrostates: microstate 1-based index for each mircostate
# - microstates: 1-based indices for each microstate (1-547 effectively)
# - n_macrostates: int (here: 12)
# - pops: populations for each microstate
# - traj: microstate trajectory
# - tlag: lagtime, 50
#
#



# (base) [fg149@bmd13 tools]$ python -i init.py 
# Clustering ...
# 100%|██████████████████████████████████████████████████████████████████████| 1/1 [00:01<00:00,  1.30s/it]
# Assigning macrostates ...
# 100%|██████████████████████████████████████████████████████████████████████| 1/1 [00:01<00:00,  1.04s/it]
# Clustering ...
#   0%|                                                                              | 0/1 [00:00<?, ?it/s]target_state: 11
# merged_states: []
# target_states: []
# q target state: 0.42812195144900267
# 100%|██████████████████████████████████████████████████████████████████████| 1/1 [00:01<00:00,  1.60s/it]
# Assigning macrostates ...
# 100%|██████████████████████████████████████████████████████████████████████| 1/1 [00:03<00:00,  3.39s/it]
# [[ 5.28137135e-03]
#  [ 9.92017766e-05]
#  [ 5.87318959e-05]
#  [-3.46659397e-03]
#  [ 2.25667549e-03]
#  [-2.12983265e-02]
#  [ 4.37382206e-02]
#  [ 1.38164614e-02]
#  [-7.95953697e-03]
#  [ 8.27920698e-03]]
# >>> f[0]
# array([0.43340332])
# >>> fl[0]
# array([0.42812195])
# >>> f1 = feature_kernel.full_feature[[272, 11]]
# >>> f1
# array([[0.4737269 ],
#        [0.42812195]])
# >>> p1 = feature_kernel.full_pop[[272, 11]]
# >>> p
# Traceback (most recent call last):
#   File "<stdin>", line 1, in <module>
# NameError: name 'p' is not defined. Did you mean: 'np'?
# >>> p1
# array([ 696, 5314], dtype=uint32)
# >>> p1 = feature_kernel.full_pop[[272, 11]][:, 0]
# Traceback (most recent call last):
#   File "<stdin>", line 1, in <module>
# IndexError: too many indices for array: array is 1-dimensional, but 2 were indexed
# >>> p1
# array([ 696, 5314], dtype=uint32)
# >>> f1 = feature_kernel.full_feature[[272, 11]][:, 0]
# >>> f1
# array([0.4737269 , 0.42812195])
# >>> (f1 * p1}.sum() / p1.sum()
#   File "<stdin>", line 1
#     (f1 * p1}.sum() / p1.sum()
#             ^
# SyntaxError: closing parenthesis '}' does not match opening parenthesis '('
# >>> (f1 * p1).sum() / p1.sum()
# 0.4334033227953411
# >>> lp1 = lfeature_kernel.full_pop[[272, 11]]
# >>> lf1 = lfeature_kernel.full_feature[[272, 11]][:, 0]
# >>> (lf1 * lp1).sum() / lp1.sum()
# 0.4334033227953411
# >>> lf1.mean()
# 0.45092442400036337
# >>> qs = feature_kernel.full_feature[:547]
# >>> from MPT.MPT_MCMC_fnc import q_macrostates
# >>> qs = feature_kernel.full_feature[:547].copy()
# >>> q_macrostates(11, [], [], qs)
# 0.42812195144900267
# >>> (feature_kernel.full_feature[:547] - qs).sum()
# 0.0
# >>> np.where([] == 11)[0]
# <stdin>:1: DeprecationWarning: Calling nonzero on 0d arrays is deprecated, as it behaves surprisingly. Use `atleast_1d(cond).nonzero()` if the old behavior was intended. If the context of this warning is of the form `arr[nonzero(cond)]`, just use `arr[cond]`.
# array([], dtype=int64)
# >>> state = 11
# >>> merged_states = []
# >>> target_states = []
# >>> q_macrostates(state, merged_states, target_states, qs)
# 0.42812195144900267
# >>> np.where(target_states == state)[0]
# array([], dtype=int64)
# >>> [merged_states[i] for i in np.where(target_states == state)[0]]
# []
# >>> id_mi_cl = np.where(target_states == state)[0]
# >>> [merged_states[i] for i in id_mi_cl]
# []
# >>> mi_cl = [merged_states[i] for i in id_mi_cl]
# >>> mi_cl.append(state)
# >>> mi_cl
# [11]
# >>> 
