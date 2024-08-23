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

# def macrostates_from_Z(
#         Z: NDArray[np.float_],
#         n_macrostates: int
#     ) -> (NDArray[np.int_], List[List[np.int_]]):
#     """
#     macrostates_from_Z
#     ------------------
#     Create map for microstates and macrostates. Index corresponds to state
#     index.
#
#     Z (NDArray[np.float_]): Z matrix as of cluster
#     n_macrostates (int): Number of macrostates to create.
#
#     returns two maps: micro -> macro, macro -> micro
#     """
#     n = Z.shape[0] + 1
#     macrostate_map = np.zeros(n, dtype=int)
#     Z_macro_flat = Z[-n_macrostates+1:, :2].flatten()
#     macrostate_roots = np.sort(Z_macro_flat)[:n_macrostates]
#     macrostates = []
#     for macrostate in macrostate_roots:
#         if macrostate < n:
#             macrostates.append([int(macrostate)])
#         else:
#             macrostates.append(get_micro(Z[:, :2].astype(int), int(macrostate-n), []))
#
#     for i, microstates in enumerate(macrostates):
#         for microstate in microstates:
#             macrostate_map[microstate] = i
#
#     return macrostate_map, macrostates

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
    for i, ms in enumerate(macrostate_assignment.astype(bool)):
        for j, other_ms in enumerate(macrostate_assignment.astype(bool)):
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

@njit
def Z_to_linkage(Z):
    l = Z[:, :, :3].copy()
    for r, z in enumerate(l):
        for i, row in enumerate(z):
            mask = np.where(l[r, :, :2]==i+Z.shape[1]+1)
            l[r, :, :2][mask] = row[1]
    l[:, :, :2] += 1
    return l

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
            if not prev and cur:
                cur_idx.add(i)
            elif prev and cur:
                cur_idx.add(i)
            elif prev and not cur:
                print(f"pops: current: {pop[list(cur_idx)].sum()} max: {pop[list(max_idx)].sum()}")
                if pop[list(cur_idx)].sum() > pop[list(max_idx)].sum():
                    indices_to_exclude.update(max_idx)
                    max_idx = cur_idx
                else:
                    indices_to_exclude.update(cur_idx)
                cur_idx = set()
    print(indices_to_exclude)
    return list(indices_to_exclude)

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
    full_pop = np.zeros(sum(macrostate_assignment.shape), pop.dtype.type)
    full_pop[:n_states] = pop
    #macro_pop = np.zeros(tmat.shape[0], pop.dtype.type)
    # full_mask = np.full(dim, False)
    # full_mask[:n_states] = True
    for i, m in enumerate(macrostate_assignment.astype(bool)):
        states = np.full(dim, False)
        states[:n_states][m] = True

        new_state = i + n_states
        # Calculate population of current macrostate
        full_pop[new_state] = pop[m].sum()

        full_macro_tmat[new_state] = (
            full_macro_tmat[states] * full_pop[states, np.newaxis]
        ).sum(axis=0) / full_pop[new_state]

        full_macro_tmat[:, new_state] = full_macro_tmat[:, states].sum(axis=1)

        full_macro_tmat[new_state, new_state] = full_macro_tmat[new_state, states].sum()

        # Set all probabilities that have been merged to 0
        full_macro_tmat[new_state, states] = 0
        full_macro_tmat[states, new_state] = 0
    return full_macro_tmat[n_states:, n_states:], full_pop

def reassign_states(tmat, pop, macrostate_assignment):
    """
    tmat: initial transitition matrix (NxN)
    pop: microstate population (N)
    macrostate_assignment: (MxN) M: number of macrostates, N: number of
            microstates; Microstates that are newly assigned have no
            macrostate here
    """
    states_to_assign = np.where(macrostate_assignment.sum(axis=0) == 0)[0]
    ma_dim = macrostate_assignment.shape[0]
    sa_dim = states_to_assign.shape[0]
    inter_dim = ma_dim + sa_dim

    inter_ma = np.zeros((inter_dim, macrostate_assignment.shape[1]), macrostate_assignment.dtype.type)
    inter_ma[:ma_dim] = macrostate_assignment
    inter_ma[np.arange(ma_dim, inter_dim), states_to_assign] = 1

    # print("inter_ma")
    # print(inter_ma)

    inter_tmat, inter_pop = get_macrostate_tmat_from_assignment(tmat, pop, inter_ma)

    for i, state in enumerate(states_to_assign, start=ma_dim):
        macrostate = np.argmax(inter_tmat[i, :ma_dim])
        # print(macrostate)
        # print(inter_tmat[i, :ma_dim])
        macrostate_assignment[macrostate, state] = 1
    
    return macrostate_assignment

