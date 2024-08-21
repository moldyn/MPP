import click
import numpy as np
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
def MPT_Macrostates(
    state_traj: str,
    tlag: int,
    cut_params: tuple[float, float],
    qtraj: str
):
    """Run MPT Algorithm from a trajectory

    Args:
        state_traj (str): directory of trajectory
        lagtime (int): lagtime of trajectory
        cut parameters (tuple): cut parameters for macrostates
        qtraj (str): directory of fraction of native contacts

    Returns:
        None:
    """
    traj = np.loadtxt(state_traj, dtype=int, comments='#')
    # Fraction of native contacts for each frame
    q_of_t = np.loadtxt(qtraj, dtype=float, comments='#')
    # min population and min fnc
    pop_thr, qmin_thr = cut_params

    tmat, states = mh.msm.estimate_markov_model(traj, tlag)
#    tmat = normalize(matrix)
    _, pop = np.unique(traj, return_counts=True)
    pop = pop/pop.sum()
    transitions = MPT(tmat, pop, states)

    formats = ['%d', '%d', '%.15f']
    linkage_file = state_traj + '_linkage.dat'
    macros_file = f'{linkage_file}_q.pop{pop_thr:.3f}_qmin{qmin_thr:.2f}'

    # microstates: list of microstates, order corresponds to dyn_corr_macrostates
    # dyn_corr_macrostates: list of macrostates, order matches the one of microstates
    # together: linkage
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


def get_qmin(tmat: np.array, pop: np.array):
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
        merge_min = merge_idx[merge_idx_min_pop]
    else:
        merge_min = merge_idx[0]
    return merge_min, stabilities[merge_min]


def trans_states(tmat: np.array, microstates: np.array, merge_idx: int):
    """Find microstates to merge and associated indices in the matrix.

    Args:
        tmat (np.array): transition probability matrix
        microstates (np.array): microstates which have not been merged yet
        idx_i (int): Index of the state which gets merged next

    Returns:
        int, int, int: merged and target microstate, index of target state
    """
    merged_state = microstates[merge_idx]
    max_vals = np.argsort(-tmat[merge_idx])
    target_state = microstates[max_vals[0]]
    if target_state == merged_state:
        target_state = microstates[max_vals[1]]
    target_idx = np.where(target_state == microstates)[0][0]
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
    # Why weighting only once with population?
    tmat[target_idx] = (tmat[merge_idx] * p_i + tmat[target_idx] * p_j) / P_ij
    tmat[:, target_idx] = tmat[:, merge_idx] + tmat[:, target_idx]
    tmat = np.delete(tmat, merge_idx, axis=0)
    red_tmat = np.delete(tmat, merge_idx, axis=1)
    pop[target_idx] = P_ij
    red_pop = np.delete(pop, merge_idx)
    return red_tmat, red_pop


def MPT(tmat: np.array, init_pop: np.array, states: np.array):
    """Iteratively cluster the least stable state with dynamically nearest
    target state

    Args:
        tmat (np.array): row-normalized transition matrix of microstates
        init_pop (np.array): equilibrium populations of transition matrix
        states (np.array): index to state assignment

    Returns:
        list: linkages in form of a n-1 by 3 matrix
    """
    microstates = np.arange(0, len(tmat))
    transitions = []
    for _ in tqdm(range(len(tmat) - 1)):
        # state to merge (has least self transition probability, in case there
        # are several states, the least populated one is chosen), qmin is
        # corresponding self transition probability
        merge_idx, qmin = get_qmin(tmat, init_pop)

        (
            merged_state,
            target_state,
            target_idx
        ) = trans_states(tmat, microstates, merge_idx)

        transitions.append([
            states[merged_state],
            states[target_state],
            qmin
        ])

        tmat, red_pop = reduction(tmat, init_pop, merge_idx, target_idx)
        init_pop = red_pop
        # Delete merged microstate from remainding microstates
        microstates = np.delete(microstates, merge_idx)

    if len(tmat) != 1:  # check wheter matrix is fully reduced
        raise TypeError('Matrix is not fully reducible in n-1 steps')

    return transitions


if __name__ == '__main__':
    MPT_Macrostates()
