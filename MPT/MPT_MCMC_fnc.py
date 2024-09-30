import numpy as np
from numpy import random
import msmhelper as mh
from msmhelper.msm import row_normalize_matrix as normalize
from tqdm import tqdm
from MPT.macrostates import macrotraj_calc


def get_qmin(tmat: np.array, pop: int):
    """Get index and value of lowest self transition probability.
    If two states equally unstable, the lower populated is chosen.

    Args:
        matrix (np.array): transition probability matrix
        pop (np.array): array of state populations

    Returns:
        int, float: Index and self transition probability of state to merge
    """
    stabilities = np.diagonal(tmat)
    merge_idx = np.where(np.amin(stabilities) == stabilities)[0]
    if len(merge_idx) > 1:
        (
            merge_idx_min_pop
        ) = np.where(np.amin(pop[merge_idx]) == pop[merge_idx])[0]
        merge_idx = merge_idx[merge_idx_min_pop[0]]
    else:
        merge_idx = merge_idx[0]
    qmin = stabilities[merge_idx]
    return merge_idx, qmin


def q_of_states(traj, q_of_t):
    """Fraction of native contacts list for all microstates in trajectory

    Args:
        traj (list): state trajectory
        q_of_t (list): fraction of native contacts trajectory

    Returns:
        list[list]: lists of all fraction of native contacts of each microstate
    """
    return np.array([
        q_of_t[traj == state].mean() for state in np.unique(traj)
    ])


def q_macrostates_(state, merged_states, target_states, q_of_t_states):
    """Generate fraction of native contacts values for macrostates

    Args:
        state (int): microstate to calculate FNC for
        merged_states (list[int]): list of all merged states
        target_states (list[int]): list of all taregt states
        q_of_t_states (list): lists of all FNC of each state

    Returns:
        float: fraction of native contacts of state
    """
    idx_micros_cluster = np.where(target_states == state)[0]
    micros_cluster = [merged_states[idx] for idx in idx_micros_cluster]
    micros_cluster.append(state)
    return q_of_t_states[micros_cluster].mean()

def q_macrostates(state, merged_states, target_states, q_of_t_states, pop):
    """Generate fraction of native contacts values for macrostates

    Args:
        state (int): microstate to calculate FNC for
        merged_states (list[int]): list of all merged states
        target_states (list[int]): list of all taregt states
        q_of_t_states (list): lists of all FNC of each state

    Returns:
        float: fraction of native contacts of state
    """
    # state = target_state
    # NOTE:
    # If both, merge state and target state are already compound states, not all microstates are considered
    # e. g. in merge 36, microstate 199 is not considered
    # >>> l = MPT.utils.Z_to_linkage(mpt.Z[0])
    # >>> ll = MPT.utils.Z_to_linkage(lmpt.Z[0])
    # >>> l[:37, :2].astype(int)[[10, 17, 24, 36]]
    # array([[ 89,  38],
    #        [200,  27],
    #        [106,  38],
    #        [ 27,  38]])
    # >>> ll[:37, :2].astype(int)[[10, 17, 24, 36]]
    # array([[ 89,  38],
    #        [200,  27],
    #        [106,  38],
    #        [ 27,  38]])
    # >>> z[:37, :2].astype(int)[[10, 17, 24, 36]]
    # array([[ 88,  37],
    #        [199,  26],
    #        [105, 557],
    #        [564, 571]])
    
    # idx_micros_cluster = np.where(target_states == state)[0]
    # micros_cluster = [merged_states[idx] for idx in idx_micros_cluster]
    # micros_cluster.append(state)

    micros_cluster = find_states(state, merged_states, target_states)
    q_ma =  (q_of_t_states[micros_cluster] * pop[micros_cluster]).sum() / pop[micros_cluster].sum()
    # print(f"target state: {state}")
    # print(merged_states)
    # print(target_states)
    # print(micros_cluster)
    # print()
    #
    return q_ma

def find_states(state, ms, ts):
    state_list = list(set(rec(state, np.array(ms), np.array(ts), [])))
    state_list.append(state)
    # print(len(state_list))
    return state_list

def rec(state, ms, ts, states):
    mss = list(ms[ts == state])
    states += mss
    for s in mss:
        states = list(set(states + rec(s, ms, ts, states)))
    return states


def modify_prob(prob_distr, merge_idx, q_states, variance, exponent):
    """Calculate fraction of native contacts scored probability distribution

    Args:
        prob_distr (list): probability distribution
        merge_idx (int): Index of the state which gets merged next
        q_states (list): fraction of native contacts of states
        variance (float): variance of FNC distribution
        exponent (int): exponent of FNC distribution

    Returns:
        list: FNC scored probability distribution
    """
    weighting_factors = np.exp(
        -(abs(q_states[merge_idx] - q_states)**exponent)/(2*variance**2)
    )
    # with open("/home/fg149/Dokumente/data_production/rep_lukas_fnc/debug/ld/trans", "a") as f:
    #     np.savetxt(f, prob_distr)
    # with open("/home/fg149/Dokumente/data_production/rep_lukas_fnc/debug/ld/wf", "a") as f:
    #     np.savetxt(f, weighting_factors)
    prob_distr_mod = prob_distr * weighting_factors
    return prob_distr_mod / np.sum(prob_distr_mod)
    # return prob_distr


def trans_states_MCMC(
    tmat,
    microstates,
    merge_idx,
    q_states,
    variance,
    exponent,
    cut_prob
):
    """Find microstates to merge and associated indices in the matrix.
    Uses MCMC approach to stochastically determine target states.
    Probability distributions can be scored with fraction of native contacts
    Choosing probability cutoff 1.0 makes clustering deterministic

    Args:
        tmat (np.array): transition probability matrix
        microstates (np.array): miucrostates which have not been merged yet
        merge_idx (int): Index of the state which gets merged next
        q_states (list): fraction of native contacts of states
        variance (float): variance of FNC distribution
        exponent (int): exponent of FNC distribution
        cut_prob (float): cutoff for low probabilities

    Returns:
        int, int int: merged and taregt microstate, index of target state
    """
    merged_state = microstates[merge_idx]
    prob_distr = tmat[merge_idx].copy()
    prob_distr[merge_idx] = 0

    if exponent != 0:
        prob_distr_mod = modify_prob(
            prob_distr,
            merge_idx,
            q_states,
            variance,
            exponent
        )
    else:
        prob_distr_mod = prob_distr

    # print(f"ld: {np.sort(prob_distr_mod)[::-1][:3]}")
    if cut_prob < 1:
        prob_distr_mod[prob_distr_mod <= cut_prob * np.max(prob_distr_mod)] = 0

        csum = np.cumsum(prob_distr_mod / np.sum(prob_distr_mod))
        random_number = random.rand()
        if csum[-1] > random_number:
            target_idx = np.where(
                csum-random_number > 0, csum-random_number, np.inf
            ).argmin()
        else:
            target_idx = np.argmax(csum)

        # prob_distr_mod = np.nan_to_num(prob_distr_mod)
        # target_idx = np.random.choice(
        #     np.arange(prob_distr_mod.shape[0]),
        #     p=prob_distr_mod / np.sum(prob_distr_mod)
        # )
    else:
        target_idx = np.argmax(prob_distr_mod)

    # with open("/home/fg149/Dokumente/data_production/rep_lukas_fnc/debug/ld/trans", "a") as f:
    #     np.savetxt(f, prob_distr_mod[np.delete(np.arange(len(prob_distr_mod)), merge_idx)])
    target_state = microstates[target_idx]
    return merged_state, target_state, target_idx


def reduction(tmat, pop, merge_idx, target_idx):
    """Adjusts transition matrix and population to the merged states

    Args:
        tmat (np.array): transition probability matrix
        pop (np.array): population of microstates
        idx_i (int): Index of state to merge
        idx_j (int): Index of target state

    Returns:
        np.array, np.array: Adjusted transition matrix and population
    """
    p_i = pop[merge_idx]
    p_j = pop[target_idx]
    P_ij = p_i + p_j
    tmat[target_idx] = (tmat[merge_idx] * p_i + tmat[target_idx] * p_j) / P_ij
    tmat[:, target_idx] = tmat[:, merge_idx] + tmat[:, target_idx]
    tmat = np.delete(tmat, merge_idx, axis=0)
    red_tmat = np.delete(tmat, merge_idx, axis=1)
    pop[target_idx] = P_ij
    red_pop = np.delete(pop, merge_idx)
    return red_tmat, red_pop


def MPT_MCMC(tmat, init_pop, variance, exponent, cut_prob, q_of_t_states):
    """Iteratively cluster the least stable state with stochastically chosen
    target state

    Args:
        matrix (np.array): row-normalized transition matrix of microstates
        init_pop (np.array): equilibrium populations of transition matrix
        variance (float): variance of FNC distribution
        exponent (int): exponent of FNC distribution
        cut_prob (float): cutoff for low probabilities
        q_of_t_states (list[list]): fraction of native contacts of microstates

    Returns:
        list: linkages in form of a n-1 by 3 matrix
    """
    popi = init_pop.copy()

    microstates = np.arange(0, len(tmat))
    transitions = []

    target_states = []
    merged_states = []

    q_states = [q_macrostates(
        state,
        merged_states,
        target_states,
        q_of_t_states,
        popi,
    ) for state in microstates]

    feature = np.zeros(len(tmat) - 1)
    for i in range(len(tmat) - 1):
        merge_idx, qmin = get_qmin(tmat, init_pop)

        (
            merged_state,
            target_state,
            target_idx,
        ) = trans_states_MCMC(
            tmat, microstates,
            merge_idx,
            q_states,
            variance,
            exponent,
            cut_prob
        )

        transitions.append([
            int(merged_state + 1),
            int(target_state + 1),
            qmin
        ])

        merged_states.append(merged_state)
        target_states.append(target_state)

        q_states[target_idx] = q_macrostates(
            target_state,
            merged_states,
            target_states,
            q_of_t_states,
            popi,
        )
        feature[i] = q_states[target_idx]
        # print(f"{merged_state} {target_state}")
        # print(q_states[target_idx])
        q_states = np.delete(q_states, merge_idx)

        # merged_states.append(merged_state)
        # target_states.append(target_state)
        tmat, red_pop = reduction(tmat, init_pop, merge_idx, target_idx)
        init_pop = red_pop
        microstates = np.delete(microstates, merge_idx)
    if len(tmat) != 1:
        raise TypeError('Matrix is not fully reducible in n-1 steps')
    return transitions, feature

