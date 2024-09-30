"""
core.py
=======

Core functions for MPT class
"""

__all__ = [
    "cluster",
]

import numpy as np
from typing import Callable, List
from numpy.typing import NDArray

import MPT.utils as utils
import MPT.kernel as kern
from MPT.macrostates import macrotraj_calc
from MPT.MPT_MCMC_fnc import MPT_MCMC, q_of_states

def cluster(
        tmat: NDArray[np.float_],
        pop: NDArray[np.int_],
        kernel: Callable[
            [NDArray[np.float_], NDArray[np.int_], NDArray[np.bool_]],
            [np.int_, np.int_, NDArray[np.bool_]]
        ]=kern.MPTKernel(),
        feature_kernel = 1,
    ) -> (NDArray[np.float_], NDArray[np.int_]):
    """
    cluster
    -------
    Perform full clustering for a transition matrix, given populations and a
    kernel.

    tmat (NDArray[np.float_]): transition matrix, e. g. from
            mh.msm.estimate_markov_model
    pop (NDArray[np.float_]): populations of microstates
    kernel: kernel object that determines the next merge

    returns Z (np.ndarray), full_pop (np.ndarray):
        The Z matrix holds the full merging of microstates:
            0: origin state
            1: target state
            2: distance between origin and target
            3: joint population
            i: Z[i, 0] and Z[i, 1] are combined to cluster n + i
            reference: scipy.cluster.hierarchy.linkage
        full_pop holds all state populations from state 0 to n + i
    """
    n = tmat.shape[0]

    full_tmat = np.zeros((2 * n - 1, 2 * n - 1), dtype=tmat.dtype.type)
    full_tmat[:n, :n] = tmat

    #full_pop = np.zeros(2 * n - 1, dtype=numba.uint64)
    full_pop = np.zeros(2 * n - 1, dtype=pop.dtype.type)
    full_pop[:n] = pop

    if tmat.shape[0] < 2**8:
        states_type = np.uint8
    elif tmat.shape[0] < 2**16:
        states_type = np.uint16
    else:
        states_type = np.uint32
    
    # complete linkage
    #full_states = np.zeros((2 * n - 1, 2), dtype=numba.uint32)
    full_states = np.zeros((2 * n - 1, 2), dtype=states_type)
    full_states[:n, 0] = np.arange(0, n)

    mask = np.full(2 * n - 1, False)
    mask[:n] = True

    # 0: state a
    # 1: state b
    # 2: distance between a and b
    # 3: population
    # i: Z[i, 0] and Z[i, 1] are combined to cluster n + i
    Z = np.zeros((n-1, 4), dtype=np.float32)

    # if feature_kernel != 1:
    #     feature_kernel.reset()
    for i in range(n-1):
        # Index of new state
        new_state = n + i

        # Use feature only for determination of target state
        if feature_kernel != 1:
            # state, target_state, mask = kernel(feature_kernel * full_tmat, full_states, mask)
            state, target_state, mask = kernel(full_tmat, full_states, mask, feature_kernel)
            feature_kernel.update(state, target_state, new_state)
        else:
            state, target_state, mask = kernel(full_tmat, full_states, mask)

        metastability = full_tmat[state, state]
        # Merge states in transition matrix
        full_tmat, full_pop = utils.merge_states(full_tmat, [state, target_state], new_state, full_pop)

        # Update state linkage
        full_states[state, 1] = new_state
        full_states[target_state, 1] = new_state
        full_states[new_state:, 0] = new_state

        Z[i] = [state, target_state, metastability, full_pop[new_state]]

        # Update mask
        mask[new_state] = True
        mask[target_state] = False

    return Z, full_pop

def cluster_mpt_mcmc(
        tmat: NDArray[np.float_],
        pop: NDArray[np.int_],
        traj,
        kernel: Callable[
            [NDArray[np.float_], NDArray[np.int_], NDArray[np.bool_]],
            [np.int_, np.int_, NDArray[np.bool_]]
        ]=kern.MPTKernel(),
        feature_kernel = 1,
    ) -> (NDArray[np.float_], NDArray[np.int_]):
    if feature_kernel == 1:
        sigma = 0.13
        b = 0
    else:
        sigma = feature_kernel.sigma
        b = feature_kernel.b

    c = kernel.c

    pop_norm = pop.sum()
    pop = pop / pop_norm
    if feature_kernel == 1:
        q_states = np.ones(pop.shape)
    else:
        q_states = q_of_states(traj, feature_kernel.feature_traj)

    linkage, feature = MPT_MCMC(
        tmat,
        pop,
        sigma,
        b,
        c,
        q_states,
    )
    if feature_kernel != 1:
        feature_kernel.full_feature[:len(linkage) + 1, 0] = q_states
        feature_kernel.full_feature[len(linkage) + 1:, 0] = feature
    Z, full_pop = utils.linkage_to_Z(linkage, pop)
    return Z, full_pop * pop_norm

def assign_macrostates(Z, full_pop, pop_thr, q_min):
    """
    Z: Z matrix
    full_pop: populations for all states incl intermediate states
    pop_thr: minimum population per macrostate (0 ... 1)
    q_min: minimum self transition probability for that state
    """
    pop_norm = full_pop / Z[-1, 3]
    macrostates = split(Z, 2 * Z.shape[0], [], pop_norm, [], pop_thr, q_min)
    # macrostate assignment
    ma = np.zeros((len(macrostates), Z.shape[0]+1), dtype=bool)
    for macrostate, microstates in enumerate(macrostates[::-1]):
        ma[macrostate, microstates] = 1
    return ma

def assign_macrostates_mcalc(Z, full_pop, pop_thr, q_min, tlag, traj, q_of_t):
    linkage = utils.Z_to_linkage(Z)
    microstates, dc_macrostates, macrotraj = macrotraj_calc(
        linkage, tlag, pop_thr, q_min, traj, q_of_t,
    )
    ma = np.zeros((dc_macrostates.max(), dc_macrostates.shape[0]), dtype=bool)
    ma[dc_macrostates-1, microstates-1] = True
    return ma

def split(Z, state, macrostates, full_pop, overflow: list, pop_thr, q_min):
    """
    Z: Z matrix
    state: state that is to split
    macrostates: list of macrostates (list of lists of microstates)
    full_pop: full population list incl indermediate states
    overflow: list of microstates that remain in macrostate
    pop_thr: population threshold
    q_min: self transition probability threshold
    """
    n_states = Z.shape[0] + 1
    q_condition = Z[state - n_states, 2] > q_min
    a, b = Z[state - n_states, :2].astype(int)

    # both states greater than population threshold
    if ( full_pop[a] > pop_thr ) & ( full_pop[b] > pop_thr ) & q_condition:
        # first process smaller state
        if full_pop[a] < full_pop[b]:
            c = a
            d = b
        else:
            c = b
            d = a

        # distinguish microstates from intermediate states
        if c < n_states:
            macrostates.append([c])
        else:
            macrostates = split(Z, c, macrostates, full_pop, [], pop_thr, q_min)
        if d < n_states:
            macrostates.append([d] + overflow)
        else:
            macrostates = split(Z, d, macrostates, full_pop, overflow, pop_thr, q_min)

    # one state greater than population threshold
    elif (( full_pop[a] > pop_thr ) ^ ( full_pop[b] > pop_thr )) & q_condition:
        # first process smaller state
        if full_pop[a] < full_pop[b]:
            c = a
            d = b
        else:
            c = b
            d = a

        # distinguish microstates from intermediate states
        if c < n_states:
            overflow.append(c)
        else:
            overflow += utils.get_micro(Z, c - n_states, [])
        if d < n_states:
            macrostates.append([d] + overflow)
        else:
            macrostates = split(Z, d, macrostates, full_pop, overflow, pop_thr, q_min)

    # both states smaller than population threshold
    else:
        # distinguish microstates from intermediate states
        if a < n_states:
            overflow.append(a)
        else:
            overflow += utils.get_micro(Z, a - n_states, [])
        if b < n_states:
            overflow.append(b)
        else:
            overflow += utils.get_micro(Z, b - n_states, [])
        macrostates.append(overflow)
    return macrostates
