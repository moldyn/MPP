import click
import numpy as np
from numpy import random
import msmhelper as mh
from msmhelper.msm import row_normalize_matrix as normalize
from tqdm import tqdm
from macrostates import macrotraj_calc


@click.command(no_args_is_help='-h')
@click.option(
    '--tlag',
    required=True,
    type=click.IntRange(min=1),
    help='Lagtime in frames',
)
@click.option(
    '--cut-params',
    default=(0.005, 0.2),
    type=click.FloatRange(min=0, max=0.99),
    nargs=2,
    help='Tuple defining the cut parameters, (pop, q) both in [0, 1).',
)
@click.option(
    '--state-traj',
    'state_traj',
    required=True,
    type=click.Path(exists=True),
    help='Used to apply MPP+ automated lumping.',
)
@click.option(
    '--fraction-of-native-contacts',
    'qtraj',
    required=True,
    type=click.Path(exists=True),
    help='File of holding fraction of native contacts Q.',
)
@click.option(
    '--iterations',
    required=True,
    type=click.IntRange(min=1),
    help='Iteration count of MCMC Clusterings'
)
@click.option(
    '--cut-prob',
    required=True,
    default=0.0,
    type=click.FloatRange(min=0, max=0.99),
    help='Cut Probability relative to maximum'
)
@click.option(
    '--state-count',
    required=True,
    type=click.IntRange(min=1),
    help='Number of considered probability maxima'
)
def MPT_MCMC_Macrostates(
    state_traj,
    tlag,
    iterations,
    cut_params,
    qtraj,
    cut_prob,
    state_count
):
    """Run stochastic MPT Algorithm from trajectory

    Args:
        state_traj (str): directory of trajectory
        lagtime (int): lagtime of trajectory
        iterations (int): number of clusterings
        cut_params (tuple(float, float)): population and metastability cutoff
        qtraj (str): directory of fraction of native contacts
        cut_prob (float): Probability cutoff relative to maximum probability
        state_count(int): Amount of considered probability maxima

    Returns:
        None:
    """

    traj = np.loadtxt(state_traj, dtype=int, comments='#')
    q_of_t = np.loadtxt(qtraj, dtype=float, comments='#')
    pop_thr, qmin_thr = cut_params
    matrix, permutation = mh.msm.estimate_markov_model(traj, tlag)
    tmat = normalize(matrix)
    states, pop = np.unique(traj, return_counts=True)
    pop = pop/np.sum(pop)

    formats = ['%d', '%d', '%.15f']
    for i in tqdm(range(iterations)):
        transitions = MPT_MCMC(
            tmat,
            pop,
            cut_prob,
            state_count
        )
        decimals = len(str(iterations))
        linkage_file = state_traj + f'_MCMC_{i:0>{decimals}}_'\
            f'{state_count}max_cut{cut_prob}_linkage.dat'
        macros_file = f'{linkage_file}_q.pop{pop_thr:.3f}_qmin{qmin_thr:.2f}'
        (
            microstates,
            dyn_corr_macrostates,
            macrotraj
        ) = macrotraj_calc(transitions, tlag, pop_thr, qmin_thr, traj, q_of_t)

        np.savetxt(linkage_file, transitions, delimiter=' ', fmt=formats)

        np.savetxt(
            macros_file + '.macrotraj',
            macrotraj,
            header='macrostates',
            fmt='%.0f'
        )

        np.savetxt(
            macros_file + '.macrostates',
            np.array([microstates, dyn_corr_macrostates]).T,
            header='microstates macrostates',
            fmt='%.0f'
        )


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
        merge_idx = merge_idx[merge_idx_min_pop]
    qmin = stabilities[int(merge_idx)]
    return int(merge_idx), qmin


def trans_states_MCMC(
    tmat: np.array,
    microstates: np.array,
    merge_idx: int,
    cut_prob, state_count,
):
    """Find microstates to merge and associated indices in the matrix.
    Uses MCMC approach to stochastically determine target states

    Args:
        tmat (np.array): transition probability matrix
        microstates (np.array): microstates which have not been merged yet
        idx_i (int): Index of the state which gets merged next

    Returns:
        int, int int: merged and taregt microstate, index of target state
    """
    merged_state = microstates[merge_idx]
    prob_distr = tmat[merge_idx].copy()
    prob_distr[merge_idx] = 0.0

    idx_largest_prob = np.argsort(prob_distr)[-state_count:]
    mask = np.zeros_like(prob_distr, dtype=bool)
    mask[idx_largest_prob] = True
    prob_distr[~mask] = 0
    prob_distr = np.where(
        prob_distr > cut_prob * np.max(prob_distr), prob_distr, 0.0
    )

    csum = np.cumsum(prob_distr / np.sum(prob_distr))
    random_number = random.rand()
    if csum[-1] > random_number:
        target_idx = np.where(
            csum-random_number > 0, csum-random_number, np.inf
        ).argmin()
    else:
        target_idx = np.argmax(csum)
    target_state = microstates[target_idx]
    return merged_state, target_state, target_idx


def reduction(tmat: np.array, pop: np.array, merge_idx: int, target_idx: int):
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


def MPT_MCMC(
    tmat: np.array,
    init_pop: np.array,
    cut_prob, state_count
):
    """Iteratively cluster the least stable state with stochastically chosen
    target state

    Args:
        matrix (np.array): row-normalized transition matrix of microstates
        init_pop (np.array): equilibrium populations of transition matrix

    Returns:
        list: linkages in form of a n-1 by 3 matrix
    """
    microstates = np.arange(0, len(tmat))
    transitions = []

    for _ in range(len(tmat) - 1):
        merge_idx, qmin = get_qmin(tmat, init_pop)

        (
            merged_state,
            target_state,
            target_idx,
        ) = trans_states_MCMC(
            tmat, microstates,
            merge_idx,
            cut_prob,
            state_count
        )
        transitions.append([
            int(merged_state + 1),
            int(target_state + 1),
            qmin
        ])

        tmat, red_pop = reduction(tmat, init_pop, merge_idx, target_idx)
        init_pop = red_pop
        microstates = np.delete(microstates, merge_idx)
    if len(tmat) != 1:
        raise TypeError('Matrix is not fully reducible in n-1 steps')
    return transitions


if __name__ == '__main__':
    MPT_MCMC_Macrostates()
