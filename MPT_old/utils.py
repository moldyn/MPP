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

    
