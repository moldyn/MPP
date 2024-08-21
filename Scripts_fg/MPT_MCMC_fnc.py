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
    default=1,
    type=click.IntRange(min=1),
    help='Iteration count of MCMC Clusterings'
)
@click.option(
    '--variance',
    default=0.05,
    type=click.FloatRange(min=0),
    help='Variance of distribution for FNC'
)
@click.option(
    '--exponent',
    default=0,
    type=click.IntRange(min=0),
    help='Exponent of distribution for FNC, default is no scoring'
)
@click.option(
    '--cut-prob',
    default=1.0,
    type=click.FloatRange(max=1.0),
    help='cut for probabilities, default is for deterministic clustering'
)
def MPT_MCMC_Macrostates(
    state_traj,
    tlag,
    iterations,
    cut_params,
    qtraj,
    variance,
    exponent,
    cut_prob,
):
    """Run stochastic MPT Algorithm from trajectory

    Args:
        traj_dir (str): directory of trajectory
        lagtime (int): lagtime of directory
        iterations (int): number of clusterings
        cut_params (_type_): cut parameters qpop, qmin to identify macrostates
        qtraj (str): fraction of native contacts trajectory
        variance (float): variance of FNC distribution
        exponent (int): exponent of FNC distribution
        cut_prob (float): cutoff for low probabilities

    Returns:
        None
    """
    traj = np.loadtxt(state_traj, dtype=int, comments='#')
    q_of_t = np.loadtxt(qtraj, dtype=float, comments='#')
    pop_thr, qmin_thr = cut_params
    q_of_t_states = q_of_states(traj, q_of_t)

    matrix, permutation = mh.msm.estimate_markov_model(traj, tlag)
    tmat = normalize(matrix)
    states, pop = np.unique(traj, return_counts=True)
    pop = pop/np.sum(pop)

    formats = ['%d', '%d', '%.15f']
    for i in tqdm(range(iterations)):
        transitions = MPT_MCMC(
            tmat,
            pop,
            variance,
            exponent,
            cut_prob,
            q_of_t_states,
        )
        decimals = len(str(iterations))

        if cut_prob == 1.0 and exponent == 0:
            linkage_file = state_traj + '_linkage.dat'
        elif cut_prob == 1.0 and exponent != 0:
            linkage_file = state_traj + f'_var{variance:.2f}'\
                f'_exp{exponent}_linkage.dat'
        else:
            linkage_file = state_traj + f'_MCMC_{i:0>{decimals}}_var'\
                f'{variance:.2f}_exp{exponent}_cut{cut_prob:.2f}_linkage.dat'

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
        print(merge_idx, merge_idx_min_pop, merge_idx_min_pop[0])
        merge_idx = merge_idx[merge_idx_min_pop[0]]
        print(merge_idx)
    qmin = stabilities[int(merge_idx)]
    return int(merge_idx), qmin


def q_of_states(traj, q_of_t):
    """Fraction of native contacts list for all microstates in trajectory

    Args:
        traj (list): state trajectory
        q_of_t (list): fraction of native contacts trajectory

    Returns:
        list[list]: lists of all fraction of native contacts of each microstate
    """
    q_of_t_states = []
    for state in np.unique(traj):
        q_of_t_states.append([q_of_t[traj == state]])
    return q_of_t_states


def q_macrostates(state, merged_states, target_states, q_of_t_states):
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
    if state not in micros_cluster:
        micros_cluster.append(state)
    q_of_t_state = []
    for idx in micros_cluster:
        q_of_t_state.extend(q_of_t_states[idx][0])
    q_state = np.mean(q_of_t_state)
    return q_state


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
    weighting_factors = np.array([
        np.exp(-(abs(q_states[merge_idx] - q_state)**exponent)/(2*variance**2))
        for q_state in q_states])
    prob_distr_mod = np.array(prob_distr.copy()) * weighting_factors
    return prob_distr_mod / np.sum(prob_distr_mod)


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
    prob_distr[merge_idx] = 0.0

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

    if cut_prob < 1.0:
        prob_distr_mod = np.where(
            prob_distr_mod > cut_prob * np.max(prob_distr_mod),
            prob_distr_mod, 0.0
        )

        csum = np.cumsum(prob_distr_mod / np.sum(prob_distr_mod))
        random_number = random.rand()
        if csum[-1] > random_number:
            target_idx = np.where(
                csum-random_number > 0, csum-random_number, np.inf
            ).argmin()
        else:
            target_idx = np.argmax(csum)
    else:
        target_idx = np.argmax(prob_distr_mod)

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
    microstates = np.arange(0, len(tmat))
    transitions = []

    target_states = []
    merged_states = []

    q_states = [q_macrostates(
        state,
        merged_states,
        target_states,
        q_of_t_states
    ) for state in microstates]

    for _ in range(len(tmat) - 1):
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

        q_states[target_idx] = q_macrostates(
            target_state,
            merged_states,
            target_states,
            q_of_t_states
        )
        q_states = np.delete(q_states, merge_idx)

        merged_states.append(merged_state)
        target_states.append(target_state)
        tmat, red_pop = reduction(tmat, init_pop, merge_idx, target_idx)
        init_pop = red_pop
        microstates = np.delete(microstates, merge_idx)
    if len(tmat) != 1:
        raise TypeError('Matrix is not fully reducible in n-1 steps')
    return transitions


if __name__ == '__main__':
    fnc_dir = '/data/MPP_MC/Lukas/Datasets/HP35_reference_data/'\
        'hp35.mindists2.gaussian10f.q'
    traj_dir = '/data/MPP_MC/Lukas/Datasets/HP35_reference_data/'\
        'hp35.selected_contacts.gaussian10f_microstates_pcs5_p153'
    tlag = 50
    pop_thr, qmin_thr = cut_params = 0.005, 0.50

    iterations = 1
    cut_prob = 1.0
    variance = 0.05
    exponent = 2

    MPT_MCMC_Macrostates()
