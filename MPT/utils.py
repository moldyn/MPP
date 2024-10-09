"""
utils.py
========

Utilities for MPT.
"""

__all__ = [
    "apply_feature"
]

import numpy as np
from numba import njit
from itertools import combinations
from typing import List
from numpy.typing import NDArray
import msmhelper as mh

@njit
def feature_mean(traj: np.ndarray, feature: np.ndarray):
    """
    traj (np.ndarray): state trajectory
    feature (np.ndarray): feature trajectory
    """
    states = np.unique(traj)
    feature_means = np.zeros(len(states))

    for state in states:
        feature_means[state-1] = feature[traj==state].mean()

    return feature_means

#@njit
def apply_feature(tmat: np.ndarray, traj: np.ndarray, feature: np.ndarray):
    """
    Apply a feature to a transition matrix according to the exponetial function
    of report 7 by LD:

    s(ij) = exp(-a(|qi - qj|)^b)
    p_score(i)|j = p(i)|j * s(ij)
    """
    a = 30
    b = 2
    feature_means = feature_mean(traj, feature)
    # combinations of i and j
    ij = np.array(list(combinations(np.arange(feature_means.shape[0]), 2)))
    # calculate the absulute difference
    abs_qij = np.abs(np.diff(feature_means[ij])[:, 0])
    # exponential term
    s_ij = np.exp(-a * abs_qij ** b)
    s_mat = np.ones(tmat.shape)
    s_mat[ij[:, 0], ij[:, 1]] = s_ij
    s_mat[ij[:, 1], ij[:, 0]] = s_ij
    # apply to transition matrix
    w_tmat = tmat * s_mat
    # return normalize weighted transition matrix
    return w_tmat / np.expand_dims(w_tmat.sum(axis=1), -1)

    
def get_micro(
        Z: NDArray[np.float_],
        i: int,
        microstates: List[int]
    ) -> List[int]:
    """
    get_micro
    ----------
    Recursively find all microstates belonging to state i.

    Z (np.ndarray): Z matrix as of cluster
    i (int): state to find all microstates for
    microstates (list):
    """
    i = int(i)
    l, r = Z[i][:2].astype(int)
    if l <= Z.shape[0]:
        microstates.append(l)
    else:
        microstates = get_micro(Z, l - Z.shape[0] - 1, microstates)
    if r <= Z.shape[0]:
        microstates.append(r)
    else:
        microstates = get_micro(Z, r - Z.shape[0] - 1, microstates)
    return microstates

def translate_traj(
        traj: NDArray[np.int_],
        map: NDArray[np.int_]
    ) -> NDArray[np.int_]:
    """
    Transform trajectory to other state names.

    traj (NDArray[np.int_]): original state trajectory
    map (NDArray[np.int_]): index is original state, value at that position is
            new value

    returns translated trajectory
    """
    macrostates = np.unique(map)
    if map.max() < 2**8:
        macrotraj_type = np.uint8
    elif map.max() < 2**16:
        macrotraj_type = np.uint16
    else:
        macrotraj_type = np.uint32

    macrotraj = np.zeros(traj.shape, dtype=macrotraj_type)
    for macrostate in macrostates:
        macrotraj[np.isin(traj, np.where(map==macrostate)[0]+1)] = macrostate
    return macrotraj + 1

def macro_traj(
        Z: NDArray[np.float_],
        traj: NDArray[np.int_],
        n_macrostates: int
    ) -> NDArray[np.int_]:
    """
    Create macrostate trajectory from Z matrix, microstate trajectory and
    number of macrostates.

    Z (NDArray[np.float_]): Z matrix as of cluster
    traj (NDArray[np.int_v]): microstate trajectory
    n_macrostates (int): number of macrostates to create

    returns: NDArray[np.int_] macrostate trajectory
    """
    macrostate_map = macrostates_from_Z(Z, n_macrostates)
    macro_traj = np.zeros(traj.shape[0], dtype=traj.dtype.type)
    for macrostate in np.unique(macrostate_map):
        microstates = np.where(macrostate_map==macrostate)[0]
        macro_traj[np.isin(traj, microstates)] = macrostate
    return macro_traj

def macro_tmat(tmat, macrostate_assignment, pop):
    """
    transform a transition matrix from microstates to macrostates
    """
    n_macrostates = macrostate_assignment.shape[0]
    m_tmat = np.zeros((n_macrostates, n_macrostates), dtype=tmat.dtype.type)
    for i, ms in enumerate(macrostate_assignment):
        for j, other_ms in enumerate(macrostate_assignment):
            m_tmat[i, j] = (tmat[ms][:, other_ms] * np.expand_dims(pop[ms], -1)).sum()
    return m_tmat / m_tmat.sum(axis=0)

#@njit
def similarity(ma1, ma2):
    # number of macrostates
    n_m1 = ma1.shape[0]
    n_m2 = ma2.shape[0]
    mat = np.zeros((n_m1, n_m2))
    # cast macrostate assignments to bool
    # ma1b = ma1.astype(bool)
    # ma2b = ma2.astype(bool)
    for i in range(n_m1):
        for j in range(n_m2):
            intersect = np.logical_and(ma1[i], ma2[j]).sum()
            union = np.logical_or(ma1[i], ma2[j]).sum()
            # mat[i, j] = intersect / union
            # mat[i, j] = intersect / ma1[i].sum()
            # mat[i, j] = ((intersect - union) / union) ** 2
            mat[i, j] = intersect

    return mat

def get_grid_format(n):
    sqrt = np.sqrt(n)
    y = int(sqrt)
    x = y
    if x < sqrt:
        x += 2
    if (x-1) * y >= n:
        x -= 1
    return x, y

def gmrq(tmat):
    # Generalized matrix Rayleigh quotient
    val, vec = np.linalg(tmat)
    return val[:, :3].sum()

def sparse_to_matrix(sparse):
    if not isinstance(sparse, np.ndarray):
        sparse = np.array(sparse)
    size = int(0.5 * (1 + np.sqrt(8 * sparse.shape[0] + 1)))
    o = np.array(list(combinations(range(size), 2))).T
    a = np.ones((size, size))
    a[o[0], o[1]] = a[o[1], o[0]] = sparse
    return a

# @njit
# def Z_to_linkage(Z):
#     l = Z[:, :, :3].copy()
#     for r, z in enumerate(l):
#         for i, row in enumerate(z):
#             mask = np.where(l[r, :, :2]==i+Z.shape[1]+1)
#             l[r, :, :2][mask] = row[1]
#     l[:, :, :2] += 1
#     return l
def Z_to_linkage(Z):
    l = Z[:, :3].copy()
    for i, row in enumerate(l):
        mask = np.where(l[:, :2]==i+Z.shape[0]+1)
        l[:, :2][mask] = row[1]
    l[:, :2] += 1
    return l

def linkage_to_Z(linkage, pop):
    linkage = np.array(linkage)
    n_states = linkage.shape[0] + 1
    Z = np.zeros((linkage.shape[0], 4))
    Z[:, :3] = linkage[:, :3]
    Z[:, :2] -= 1

    full_pop = np.zeros(2 * n_states - 1, dtype=pop.dtype.type)
    full_pop[:n_states] = pop
    for i, l in enumerate(linkage[:-1]):
        new_state = n_states + i
        old_state = Z[i, 1]
        full_pop[new_state] = full_pop[[Z[i, 0].astype(int), int(old_state)]].sum()
        Z[i+1:, :2][np.where(Z[i+1:, :2] == old_state)] = new_state
    full_pop[-1] = full_pop[Z[i+1, :2].astype(int)].sum()
    Z[:, 3] = full_pop[n_states:]
    return Z, full_pop

def get_microstates_to_reassign(pop, macrostate_assignment):
    indices_to_exclude = set()
    for mi, m in enumerate(macrostate_assignment):
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
                else:
                    indices_to_exclude.update(cur_idx)
                cur_idx = set()

    return list(indices_to_exclude)

def merge_states(tmat, states, new_state, full_pop):
    full_pop[new_state] = full_pop[states].sum()

    tmat[new_state] = (
        tmat[states] * full_pop[states, np.newaxis]
    ).sum(axis=0) / full_pop[new_state]

    tmat[:, new_state] = tmat[:, states].sum(axis=1)

    # tmat[new_state, new_state] = tmat[new_state, states].sum()

    # Set all probabilities that have been merged to 0
    # tmat[new_state, states] = 0
    # tmat[states, new_state] = 0
    tmat[:, states] = 0
    tmat[states, :] = 0
    return tmat, full_pop

def merge_states_(tmat, states, new_state, full_pop, traj):
    # NOTE:
    # implement here merging using mh.msm.estimate_markov_model
    # macrostate assignment required of some kind
    full_pop[new_state] = full_pop[states].sum()

    tmat[new_state] = (
        tmat[states] * full_pop[states, np.newaxis]
    ).sum(axis=0) / full_pop[new_state]

    tmat[:, new_state] = tmat[:, states].sum(axis=1)

    tmat[new_state, new_state] = tmat[new_state, states].sum()

    # Set all probabilities that have been merged to 0
    tmat[new_state, states] = 0
    tmat[states, new_state] = 0
    return tmat, full_pop

def get_macrostate_tmat_from_assignment(tmat, pop, macrostate_assignment):
    """
    tmat: initial transitition matrix (NxN)
    pop: microstate population (N)
    """
    #dims = (macrostate_assignment.shape[0], macrostate_assignment.shape[0])
    n_states = tmat.shape[0]
    dim = sum(macrostate_assignment.shape)

    #macro_tmat = np.zeros(tmat.shape, tmat.dtype.type)
    #macro_tmat = np.zeros(dims, tmat.dtype.type)
    full_macro_tmat = np.zeros((dim, dim), tmat.dtype.type)
    full_macro_tmat[:n_states][:, :n_states] = tmat
    #macro_pop = np.zeros(macrostate_assignment.shape[0], pop.dtype.type)
    full_pop = np.zeros(dim, pop.dtype.type)
    full_pop[:n_states] = pop
    #macro_pop = np.zeros(tmat.shape[0], pop.dtype.type)
    # full_mask = np.full(dim, False)
    # full_mask[:n_states] = True
    for i, m in enumerate(macrostate_assignment):
        full_macro_tmat, full_pop = merge_states(full_macro_tmat, np.where(m)[0], i + n_states, full_pop)
    return full_macro_tmat[n_states:, n_states:], full_pop[n_states:]

def reassign_states__(tmat, pop, macrostate_assignment, traj):
    """
    tmat: initial transitition matrix (NxN)
    pop: microstate population (N)
    macrostate_assignment: (MxN) M: number of macrostates, N: number of
            microstates; Microstates that are newly assigned have no
            macrostate here
    """
    states_to_assign = np.where(macrostate_assignment.sum(axis=0) == 0)[0]
    n_macrostates = macrostate_assignment.shape[0]
    sa_dim = states_to_assign.shape[0]
    inter_dim = n_macrostates + sa_dim

    inter_ma = np.zeros((inter_dim, macrostate_assignment.shape[1]), macrostate_assignment.dtype.type)
    inter_ma[:n_macrostates] = macrostate_assignment
    inter_ma[np.arange(n_macrostates, inter_dim), states_to_assign] = True

    # print("inter_ma")
    # print(inter_ma)

    inter_tmat, inter_pop = get_macrostate_tmat_from_assignment(tmat, pop, inter_ma)
    print((inter_tmat > 0).astype(int))

    full_inter_pop = np.zeros(inter_dim + sa_dim, dtype=inter_pop.dtype.type)
    full_inter_pop[:inter_dim] = inter_pop[-inter_dim:]

    full_inter_tmat = np.zeros((inter_dim + sa_dim, inter_dim + sa_dim), dtype=inter_tmat.dtype.type)
    # full_inter_tmat[:inter_dim][:, :inter_dim] = inter_tmat[-inter_dim:][:, -inter_dim]
    full_inter_tmat[:inter_dim][:, :inter_dim] = inter_tmat
    print((full_inter_tmat > 0).astype(int))

    # microstate index
    merge_order = np.argsort(np.diag(tmat[states_to_assign]))
    print(merge_order)

    # full_inter_tmat index
    merging_states = np.argsort(np.diag(full_inter_tmat)[n_macrostates:inter_dim]) + n_macrostates
    print(merging_states)
    print("----------")
    tmat_mask = np.full(inter_dim + sa_dim, False)
    tmat_mask[:n_macrostates] = True

    macrostate_order = np.arange(n_macrostates)
    for i, (state, state_to_merge) in enumerate(zip(states_to_assign[merge_order], merging_states), start=inter_dim):
        # # 1-based macrostates
        # ma = np.array([np.where(inter_ma[:, j])[0][0] for j in range(macrostate_assignment.shape[1])])
        # f_tmat, mstates = mh.msm.estimate_markov_model(
        #     mh.shift_data(traj, np.arange(macrostate_assignment.shape[1]), ma),
        #     lagtime=50,
        # )
        #
        # state = np.argsort(np.diag(f_tmat[n_macrostates:]))[0]
        # target = np.argsort(f_tmat[state, :n_macrostates])[-1]
        # macrostate_assignment[target, states_to_assign[state]] = True


        # state_to_merge = np.argsort(np.diag(full_inter_tmat)[n_macrostates:inter_dim])[0] + n_macrostates
        # Only states to merge
        s = np.argsort((full_inter_tmat * ~np.diag(np.full(inter_dim + sa_dim, True)))[state_to_merge, tmat_mask])
        print(f"sorting: {s}")
        print(f"{np.sort(full_inter_tmat[state_to_merge, tmat_mask])}")
        target = s[-1] + (~tmat_mask[:s[-1]]).sum()
        # NOTE:
        # Mind the order of macrostate indices
        print(f"target: {target}")
        macrostate_assignment[s[-1], state] = True
        full_inter_tmat, full_inter_pop = merge_states(full_inter_tmat, [state_to_merge, target], i, full_inter_pop)
        full_inter_tmat[target] = full_inter_tmat[i]
        full_inter_tmat[:, target] = full_inter_tmat[:, i]
        tmat_mask[s[-1]] = False
        tmat_mask[i] = True
        # print(tmat_mask.astype(np.uint8))


    # # 1-based macrostates
    # ma = np.array([np.where(macrostate_assignment[:, i])[0][0]+1 for i in range(macrostate_assignment.shape[1])])
    # tmat, mstates = mh.msm.estimate_markov_model(
    #     mh.shift_data(traj, microstates, ma),
    #     lagtime=50,
    # )


    # for i, state in enumerate(states_to_assign, start=n_macrostates):
    #     macrostate = np.argmax(inter_tmat[i, :n_macrostates])
    #     # print(macrostate)
    #     # print(inter_tmat[i, :n_macrostates])
    #     macrostate_assignment[macrostate, state] = True
    
    return macrostate_assignment


def reassign_states_(tmat, pop, macrostate_assignment):
    """
    tmat: initial transitition matrix (NxN)
    pop: microstate population (N)
    macrostate_assignment: (MxN) M: number of macrostates, N: number of
            microstates; Microstates that are newly assigned have no
            macrostate here
    """
    states_to_assign = np.where(macrostate_assignment.sum(axis=0) == 0)[0]
    n_macrostates = macrostate_assignment.shape[0]
    sa_dim = states_to_assign.shape[0]
    inter_dim = n_macrostates + sa_dim

    inter_ma = np.zeros((inter_dim, macrostate_assignment.shape[1]), macrostate_assignment.dtype.type)
    inter_ma[:n_macrostates] = macrostate_assignment
    inter_ma[np.arange(n_macrostates, inter_dim), states_to_assign] = True

    inter_tmat, inter_pop = get_macrostate_tmat_from_assignment(tmat, pop, inter_ma)
    # print(states_to_assign)
    # print(inter_tmat.shape)

    # microstate index
    merge_order = np.argsort(np.diag(tmat)[states_to_assign])
    states_to_assign_ordered = states_to_assign[merge_order]

    tmp_tmat = np.zeros((inter_dim+1, inter_dim+1))
    tmp_tmat[:-1][:, :-1] = inter_tmat

    tmp_pop = np.zeros(inter_dim+1)
    tmp_pop[:-1] = inter_pop

    for state in merge_order:
        target = np.argsort(tmp_tmat[state + n_macrostates, :n_macrostates])[-1]
        # print(tmp_tmat[state + n_macrostates, state + n_macrostates])
        # print(tmp_tmat[state + n_macrostates, target])
        # print(tmp_tmat[target, target])
        # print()
        tmp_tmat, tmp_pop = merge_states(tmp_tmat, [state + n_macrostates, target], -1, tmp_pop)
        tmp_tmat[target] = tmp_tmat[-1]
        tmp_tmat[:, target] = tmp_tmat[:, -1]
        tmp_pop[target] = tmp_pop[-1]
        macrostate_assignment[target, states_to_assign[state]] = True
    return macrostate_assignment

def state_sequences(macrostates, state):
    """Get continuous index sequences of macrostate in mstate assignment."""
    state_idx = np.where(macrostates == state)[0]
    idx_jump = state_idx[1:] - state_idx[:-1] != 1
    return np.array_split(
        state_idx,
        np.nonzero(idx_jump)[0] + 1,
    )

def reassign_states(
    tmat,
    pop,
    macrostate_assignment,
    traj,
    macrostates,
    tlag=50,
    # *,
    # macrostates,
    # microstates,
    # n_macrostates,
    # pops,
    # traj,
    # tlag,
):
    """Apply MPP+ step2: Dynamically correct minor branches."""
    n_macrostates = macrostate_assignment.shape[0]
    microstates = np.arange(macrostate_assignment.shape[1])
    pops = pop




    # # fix dynamically missassigned single-state branches
    # # identify them
    # dyn_corr_macrostates = macrostates.copy()
    # for mstate in np.unique(macrostates):
    #     idx_sequences = state_sequences(macrostates, mstate)
    #     if len(idx_sequences) > 1:
    #         highest_pop_sequence = np.argmax([
    #             np.sum([
    #                 pops[s] for s in microstates[seq]
    #             ]) for seq in idx_sequences
    #         ])
    #         idx_sequences = [
    #             seq for idx, seq in enumerate(idx_sequences)
    #             if idx != highest_pop_sequence
    #         ]
    #         for seq in idx_sequences:
    #             largest_state = np.max(dyn_corr_macrostates)
    #             for newstate, seq_idx in enumerate(
    #                 seq,
    #                 largest_state + 1,
    #             ):
    #                 dyn_corr_macrostates[seq_idx] = newstate
    #
    # # dynamically reassign all new state to previous macrostates
    # mstates = np.unique(dyn_corr_macrostates)
    # print(len(mstates) - n_macrostates)
    # while len(mstates) > n_macrostates:
    # # for _ in range(len(mstates) - n_macrostates):
    #     tmat, mstates = mh.msm.estimate_markov_model(
    #         mh.shift_data(traj, microstates, dyn_corr_macrostates),
    #         lagtime=tlag,
    #     )
    #
    #     # sort new states by increasing metastability
    #     qs = np.diag(tmat)[n_macrostates:]
    #     deletestate = mstates[n_macrostates:][np.argsort(qs)[0]]
    #     print(f"deletestate: {deletestate}")
    #
    #     # reassign them
    #     idx = np.where(mstates == deletestate)[0][0]
    #     print(f"idx: {idx}")
    #     idxs_to = np.argsort(tmat[idx])[::-1]
    #
    #     dyn_corr_macrostates[
    #         dyn_corr_macrostates == deletestate
    #     ] = idxs_to[1] + 1 if idx == idxs_to[0] else idxs_to[0] + 1
    #
    #     mstates = np.unique(dyn_corr_macrostates)
    #     print(len(mstates))
    #
    # new_macro = np.zeros(macrostate_assignment.shape[1], dtype=np.uint32)
    # for i, j in enumerate(np.unique(dyn_corr_macrostates)):
    #     new_macro[np.where(dyn_corr_macrostates == j)[0]] = i
    #
    # print(new_macro.max())
    # print(macrostate_assignment.shape)
    # print(dyn_corr_macrostates.shape)
    # print(dyn_corr_macrostates.max())
    # print(len(np.unique(dyn_corr_macrostates)))
        
    dyn_corr_macrostates = macrostates[:]
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
                    dyn_corr_macrostates[seq_idx] = newstate

    # dynamically reassign all new state to previous macrostates
    mstates = np.unique(dyn_corr_macrostates)
    while len(mstates) > n_macrostates:
        tmat, mstates = mh.msm.estimate_markov_model(
            mh.shift_data(traj, microstates, dyn_corr_macrostates),
            lagtime=tlag,
        )

        # sort new states by increasing metastability
        qs = np.diag(tmat)[n_macrostates:]
        idx_sort = np.argsort(qs)
        newstates = mstates[n_macrostates:][idx_sort]

        deletestate = newstates[0]

        # reassign them
        idx = np.where(mstates == deletestate)[0][0]
        idxs_to = np.argsort(tmat[idx])[::-1]
        for idx_to in idxs_to:
            if idx_to == idx:
                continue
            dyn_corr_macrostates[
                dyn_corr_macrostates == deletestate
            ] = mstates[idx_to]
            break

        mstates = np.unique(dyn_corr_macrostates)



    # new_macro = np.zeros(macrostate_assignment.shape[1], dtype=np.uint32)
    # for i, j in enumerate(np.unique(dyn_corr_macrostates)):
    #     new_macro[np.where(dyn_corr_macrostates == j)[0]] = i

    # return dyn_corr_macrostates
    new_ma = np.zeros(macrostate_assignment.shape, dtype=macrostate_assignment.dtype.type)
    new_ma[dyn_corr_macrostates-1, np.arange(dyn_corr_macrostates.shape[0])] = True
    # new_ma[new_macro, np.arange(dyn_corr_macrostates.shape[0])] = True
    # new_ma[np.arange(dyn_corr_macrostates.shape[0]), new_macro] = True
    return new_ma

def dim(n):
    return int(n * (n+3) / 2)

def ld_tmat(Z, pop, tmat):
    linkage = Z_to_linkage(Z)
    tm = tmat.copy()
    po = pop.copy()

    mask = np.full(pop.shape, True)
    n = pop.shape[0]
    dims = np.zeros(n, dtype=int)
    dims[1:] = np.arange(n, 1, -1)
    c_dims = np.cumsum(dims)
    tps = np.zeros(dim(n-1))

    for i, (o, t) in enumerate(linkage[:, :2].astype(int)):
        tps[c_dims[i]:c_dims[i+1]] = tm[o-1][mask]
        tm, po = merge_states(tm, [o-1, t-1], t-1, po)
        print(tm.sum())
        mask[o-1] = False

    return tm, po, tps
