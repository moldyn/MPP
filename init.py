#!/usr/bin/env python

import os
import time
import inspect

import numpy as np
import matplotlib.pyplot as plt

import msmhelper as mh

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

def state_sequences(macrostates, state):
    """Get continuous index sequences of macrostate in mstate assignment."""
    state_idx = np.where(macrostates == state)[0]
    idx_jump = state_idx[1:] - state_idx[:-1] != 1
    return np.array_split(
        state_idx,
        np.nonzero(idx_jump)[0] + 1,
    )

def mpp_plus_dyn_cor(
    *,
    macrostates,
    microstates,
    n_macrostates,
    pops,
    traj,
    tlag,
):
    """Apply MPP+ step2: Dynamically correct minor branches."""
    # fix dynamically missassigned single-state branches
    # identify them
    dyn_corr_macrostates = macrostates.copy()
    for mstate in np.unique(macrostates):
        idx_sequences = state_sequences(macrostates, mstate)
        if len(idx_sequences) > 1:
            highest_pop_sequence = np.argmax([
                np.sum([
                    pops[s] for s in microstates[seq]
                ]) for seq in idx_sequences
            ])
            idx_sequences = [
                seq for idx, seq in enumerate(idx_sequences)
                if idx != highest_pop_sequence
            ]
            for seq in idx_sequences:
                largest_state = np.max(dyn_corr_macrostates)
                for newstate, seq_idx in enumerate(
                    seq,
                    largest_state + 1,
                ):
                    print(f"seq: {seq_idx}; newstate: {newstate}")
                    dyn_corr_macrostates[seq_idx] = newstate
    print(dyn_corr_macrostates)

    # dynamically reassign all new state to previous macrostates
    mstates = np.unique(dyn_corr_macrostates)
    # while len(mstates) > n_macrostates:
    for _ in range(len(mstates) - n_macrostates):
        tmat, mstates = mh.msm.estimate_markov_model(
            mh.shift_data(traj, microstates, dyn_corr_macrostates),
            lagtime=tlag,
        )

        # sort new states by increasing metastability
        qs = np.diag(tmat)[n_macrostates:]
        deletestate = mstates[n_macrostates:][np.argsort(qs)[0]]
        # deletestate = n_macrostates + np.argsort(qs)[0] + 1
        # idx_sort = np.argsort(qs)
        # newstates = mstates[n_macrostates:][idx_sort]
        # deletestate = newstates[0]

        # reassign them
        idx = np.where(mstates == deletestate)[0][0]
        # idx = deletestate - 1
        print(f"delstate: {deletestate}; mstates: {mstates}")
        idxs_to = np.argsort(tmat[idx])[::-1]

        # for idx_to in idxs_to:
        #     if idx_to == idx:
        #         continue
        #     dyn_corr_macrostates[
        #         dyn_corr_macrostates == deletestate
        #     ] = mstates[idx_to]
        #     break
        dyn_corr_macrostates[
            dyn_corr_macrostates == deletestate
        ] = idxs_to[1] + 1 if idx == idxs_to[0] else idxs_to[0] + 1

        # mstates = np.unique(dyn_corr_macrostates)

    return dyn_corr_macrostates


traj = np.loadtxt("/data/evaluation/MPP/stochastic_MPP_Felix/data_production/MPT/MPT/hp35.selected_contacts.gaussian10f_microstates_pcs5_p153", dtype=np.uint16)
feature_traj = np.loadtxt("/data/evaluation/MPP/stochastic_MPP_Felix/data_production/MPT/MPT/hp35.mindists2.gaussian10f.q")
multi_feature_traj = np.loadtxt("/data/evaluation/MPP/stochastic_MPP_Felix/data_production/MPT/MPT/hp35.mindists2")

out_base = "/data/evaluation/MPP/stochastic_MPP_Felix/data_production/MPT/MPT/"
out = out_base + "img/hp35_dendrogram_det_dc.pdf"

lagtime = 50

mpt_kernel = MPT.kernel.MPTKernel()
smpt_kernel = MPT.kernel.SMPTKernel(method="p", param=1, c=0.15)
feature_kernel = MPT.kernel.FeatureKernel(feature_traj, traj, sigma=0.05)

mpt = MPT.MPT(traj, lagtime)
mpt.mpt(mpt_kernel)
#mpt.mpt(mpt_kernel, feature_kernel=feature_kernel)
mpt.add_feature(feature_traj)
#mpt.assign_macrostates(0.005, 0.5, dyn_correct=True)
mpt.assign_macrostates(0.005, 0.5)
# mpt.plot("/home/fg149/Dokumente/data_production/tmp_dev/no_dc.pdf")
mpt.calc_timescales()
print(mpt.timescales[0])
#mpt.plot(out)

o = [l.name for l in mpt.tree[0].leaves]
ma = mpt.macrostate_assignment[0][:, o]
ro = np.arange(mpt.n_states)[o]
ieo = MPT.utils.get_microstates_to_reassign(mpt.full_pop[0], ma)
ie = ro[ieo]



#ma[:, ie] = 0
#nma, inter_ma, inter_tmat = MPT.utils.reassign_states(mpt.tmat, mpt.full_pop[0, :mpt.n_states], ma)

n_macrostates = mpt.n_macrostates[0]
ma = np.array([np.where(mpt.macrostate_assignment[0][:, i])[0][0]+1 for i in range(547)])[o]
mi = np.arange(0, 547)
pops = mpt.full_pop[0][:mpt.n_states][o]
dcm = mpp_plus_dyn_cor(macrostates=ma, microstates=mi, n_macrostates=n_macrostates, pops=pops, traj=traj, tlag=lagtime)
print(dcm)

mpt.assign_macrostates(0.005, 0.5, dyn_correct=True)
mpt.calc_timescales()
print(mpt.timescales[0])
madc = np.array([np.where(mpt.macrostate_assignment[0][:, i])[0][0]+1 for i in range(547)])[o]
print(madc)

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

