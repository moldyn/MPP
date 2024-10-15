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
import scipy as scy
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

def merge_states(tmat, states, new_state, full_pop):
    full_pop[new_state] = full_pop[states].sum()

    tmat[new_state] = (
        tmat[states] * full_pop[states, np.newaxis]
    ).sum(axis=0) / full_pop[new_state]

    tmat[:, new_state] = tmat[:, states].sum(axis=1)
    tmat[:, states] = 0
    tmat[states, :] = 0
    return tmat, full_pop

def get_macrostate_tmat_from_assignment(tmat, pop, macrostate_assignment):
    """
    tmat: initial transitition matrix (NxN)
    pop: microstate population (N)
    """
    n_states = tmat.shape[0]
    dim = sum(macrostate_assignment.shape)

    full_macro_tmat = np.zeros((dim, dim), tmat.dtype.type)
    full_macro_tmat[:n_states][:, :n_states] = tmat
    full_pop = np.zeros(dim, pop.dtype.type)
    full_pop[:n_states] = pop
    for i, m in enumerate(macrostate_assignment):
        full_macro_tmat, full_pop = merge_states(full_macro_tmat, np.where(m)[0], i + n_states, full_pop)
    return full_macro_tmat[n_states:, n_states:], full_pop[n_states:]

def dim(n):
    return int(n * (n+3) / 2)

def get_macrostate_assignment_from_tree(tree):
    macrostate_order = [l.assigned_macrostate.name for l in tree.leaves]
    macrostates = {l.assigned_macrostate for l in tree.leaves}
    q_ma = np.array([(m.name, m.feature) for m in macrostates])
    ma_order = np.argsort(q_ma[:, 1])[::-1]
    # Dict to translate from n+i numbering to actual macrostate numbers.
    full2real = {f: r for r, f in enumerate(q_ma[ma_order, 0])}
    macrostate_assignment = np.full((len(macrostates), len(macrostate_order)), False)
    macrostate_assignment[[full2real[m] for m in macrostate_order], np.arange(len(macrostate_order))] = True
    reorder_microstates = np.zeros(len(macrostate_order), dtype=int)
    reorder_microstates[[l.name for l in tree.leaves]] = np.arange(len(macrostate_order))
    return macrostate_assignment[:, reorder_microstates]

def similarity(ref, sto):
    """Return similarity of two clusterings"""
    # Similarity matrix
    S = np.zeros((3, ref.n_macrostates[0], sto.n_runs))
    
    for n_i in range(sto.n_runs):
        ref_ma = ref.macrostate_assignment[0].astype(bool)
        sto_ma = sto.macrostate_assignment[n_i].astype(bool)
        for i in range(ref.n_macrostates[0]):
            for j in range(sto.n_macrostates[n_i]):
                intersect = (np.logical_and(ref_ma[i], sto_ma[j]) * ref.full_pop[0, :ref.n_states]).sum()
                union = (np.logical_or(ref_ma[i], sto_ma[j]) * ref.full_pop[0, :ref.n_states]).sum()
                # union
                S[0, i, n_i] = max(S[0, i, n_i], intersect / union)
                # reference
                S[1, i, n_i] = max(S[1, i, n_i], intersect / (ref_ma[i] * ref.full_pop[0, :ref.n_states]).sum())
                # clustering
                S[2, i, n_i] = max(S[2, i, n_i], intersect / (sto_ma[j] * ref.full_pop[0, :ref.n_states]).sum())
    return S

def kullback_leibler_probability(transitions, tmat, epsilon=1e-6):
    """Return Kallback-Leibler probability"""
    tmat = tmat.copy()
    np.fill_diagonal(tmat, transitions)
    tmat += epsilon
    smoothed_tmat = tmat / np.expand_dims(tmat.sum(axis=1), axis=1)
    kl = scy.stats.entropy(transitions, smoothed_tmat, axis=1)
    exp_kl = np.exp(-kl)
    p = exp_kl / exp_kl.sum()
    return p
