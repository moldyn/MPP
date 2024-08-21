import click
import msmhelper as mh
import numpy as np
from scipy.cluster.hierarchy import dendrogram


def macrotraj(
    linkage_file,
    tlag=50,
    cut_params=(0.005, 0.5),
    state_traj,
    qtraj,
):
    # parse input and create output basename
    pop_thr, qmin_thr = cut_params
    output_file = f'{linkage_file}_q.pop{pop_thr:.3f}_qmin{qmin_thr:.2f}'

    # load transitions and sort them
    transitions = np.loadtxt(linkage_file, comments='#')
    traj = np.loadtxt(state_traj)
    q_of_t = np.loadtxt(qtraj, dtype=np.float32)

    (
        microstates,
        dyn_corr_macrostates,
        macrotraj
    ) = macrotraj_calc(transitions, tlag, pop_thr, qmin_thr, traj, q_of_t)

    # save microstate -> macrostate
    mh.savetxt(f'{output_file}.macrostates', np.array([microstates, dyn_corr_macrostates]).T, header='microstates macrostates')
    # save macrostate trajectory
    mh.savetxt(f'{output_file}.macrotraj', macrotraj, header='macrostates')


def macrotraj_calc(transitions, tlag, pop_thr, qmin_thr, traj, q_of_t):
    (
        linkage_mat,
        states_idx_to_microstates,
        states_idx_to_rootstates,
    ) = _transitions_to_linkage(transitions, qmin=0)

    # get states
    nstates = len(linkage_mat) + 1
    # Bullshit
    states = np.unique(linkage_mat[:, :2].astype(int))

    # estimate population of states
    microstates, counts = np.unique(traj, return_counts=True)
    pops = counts / len(traj)
    pops = {
        idx_state: np.sum([
            pops[microstates == state]
            for state in states_idx_to_microstates[idx_state]
        ])
        for idx_state in states
    }
    pops[2 * (nstates - 1)] = 1.0

    # find optimal cut
    macrostates, macrostates_assignment = mpp_plus_cut(
        states_idx_to_rootstates=states_idx_to_rootstates,
        states_idx_to_microstates=states_idx_to_microstates,
        linkage_mat=linkage_mat,
        microstates=microstates,
        pops=pops,
        pop_thr=pop_thr,
        qmin_thr=qmin_thr,
    )
    n_macrostates = len(macrostates_assignment)

    dendrogram_dict = _dendrogram(
        linkage_mat=linkage_mat,
    )

    # permute macrostate assignment and label them
    macrostates_assignment = macrostates_assignment.T[
        dendrogram_dict['leaves']
    ].T
    macrostates = macrostates[dendrogram_dict['leaves']]
    microstates = microstates[dendrogram_dict['leaves']]

    # apply dynamical correction of minor branches
    dyn_corr_macrostates = mpp_plus_dyn_cor(
        macrostates=macrostates,
        microstates=microstates,
        n_macrostates=n_macrostates,
        pops=pops,
        traj=traj,
        tlag=tlag,
    )

    # rename macrostates by fraction of native contacts
    macrotraj = mh.shift_data(traj, microstates, dyn_corr_macrostates)
    macrostates_q = [
        1 - _fraction_of_native_contacts([state], q_of_t, macrotraj)
        for state in np.unique(macrostates)
    ]
    macroperm = np.unique(dyn_corr_macrostates)[np.argsort(macrostates_q)]
    dyn_corr_macrostates = mh.shift_data(
        dyn_corr_macrostates, macroperm, np.unique(dyn_corr_macrostates),
    )

    macrotraj = mh.shift_data(traj, microstates, dyn_corr_macrostates)

    return microstates, dyn_corr_macrostates, macrotraj


def _fraction_of_native_contacts(states, q_of_t, traj):
    """Mean fraction of native contacts per state."""
    if len(states):
        mask = np.full(q_of_t.shape[0], False)
        for state in states:
            mask = np.logical_or(
                mask,
                traj == state,
            )
        cs = q_of_t[mask]
    else:
        cs = q_of_t
    return np.mean(cs)


def _transitions_to_linkage(trans, *, qmin=0.0):
    """Convert transition matrix to linkage matrix.

    Parameters
    ----------
    transitions: ndarray of shape (nstates - 1, 3)
        Three column: merged state, remaining state, qmin lebel.

    qmin: float [0, 1]
        Qmin cut-off. Returns only sublinkage-matrix.

    """
    transitions = np.copy(trans)

    # Already sorted !!!
    # sort by merging qmin level
#    transitions = transitions[
#        np.argsort(transitions[:, 2])
#    ]


    # qmin is 0, thus, a full True array
    # create linkage matrix
    mask_qmin = transitions[:, 2] > qmin

    # Since mask_qmin is full True, this is number of microstates
#    nstates_qmin = np.count_nonzero(mask_qmin) + 1
    # much easier possible
#    linkage_mat = np.zeros((nstates_qmin - 1, 4))
    linkage_mat = np.zeros((transitions.shape[0], 4))

    # Effectively, subtract one from state indices to match matrix indices (1 based to 0 based index)
    # replace state names by their indices
    transitions_idx, states_idx = mh.rename_by_index(
        transitions[:, :2][mask_qmin].astype(int),
        return_permutation=True,
    )
    transitions[:, :2][mask_qmin] = transitions_idx
    linkage_mat[:, :3] = transitions[mask_qmin]

    # holds for each state (index) a list corresponding to the microstates
    # it consist of.
    states_idx_to_microstates = {
        idx: [
            state,
            *transitions[~mask_qmin][:, 0][
                transitions[~mask_qmin][:, 1] == state
            ].astype(int),
        ]
        for idx, state in enumerate(states_idx)
    }
    # ... has the same effect as
    states_idx_to_microstates = {i: j for i, j in enumerate(states_idx)}

    # ... thus, the indices in the lists of the following dict are just 1 lower (0 based instead of 1 based)
    states_idx_to_rootstates = {
        idx: [idx]
        for idx, _ in enumerate(states_idx)
    }

    for idx, nextstate in enumerate(
        range(nstates_qmin, 2 * nstates_qmin - 1),
    ):
        statefrom, stateto = linkage_mat[idx, :2].astype(int)
        states_idx_to_microstates[nextstate] = [
            *states_idx_to_microstates[stateto],
            *states_idx_to_microstates[statefrom],
        ]
        states_idx_to_rootstates[nextstate] = [
            *states_idx_to_rootstates[stateto],
            *states_idx_to_rootstates[statefrom],
        ]

        states = linkage_mat[idx, :2].astype(int)
        for state in states:
            linkage_mat[idx + 1:, :2][
                linkage_mat[idx + 1:, :2] == state
            ] = nextstate

    # Attention: linkage_mat state ids are shifted by one to make array ids!!!
    return (
        linkage_mat,
        states_idx_to_microstates,
        states_idx_to_rootstates,
    )


def _dendrogram(linkage_mat):
    dendrogram_dict = dendrogram(
        linkage_mat,
        leaf_rotation=90,
        get_leaves=True,
        no_plot=True,
    )
    return dendrogram_dict


def state_sequences(macrostates, state):
    """Get continuous index sequences of macrostate in mstate assignment."""
    state_idx = np.where(macrostates == state)[0]
    idx_jump = state_idx[1:] - state_idx[:-1] != 1
    return np.array_split(
        state_idx,
        np.nonzero(idx_jump)[0] + 1,
    )


def mpp_plus_cut(
    *,
    states_idx_to_rootstates,
    states_idx_to_microstates,
    linkage_mat,
    microstates,
    pops,
    pop_thr,
    qmin_thr,
):
    """Apply MPP+ step1: Identify branches."""
    nstates = len(linkage_mat) + 1

    # Dude, what're you doing??
    macrostates_set = [
        set(states_idx_to_rootstates[2 * (nstates - 1)]),
    ]
    macrostates_leaf_set = [
        set(states_idx_to_microstates[2 * (nstates - 1)]),
    ]
    for state_i, state_j, qmin in reversed(linkage_mat[:, :3]):
        if pops[state_i] > pop_thr and qmin > qmin_thr:
            mstate_i = set(states_idx_to_rootstates[state_i])
            macrostates_set = [
                mstate - mstate_i
                for mstate in macrostates_set
            ]
            macrostates_set.append(mstate_i)

            mstate_leaf_i = set(states_idx_to_microstates[state_i])
            macrostates_leaf_set = [
                mstate - mstate_leaf_i
                for mstate in macrostates_leaf_set
            ]
            macrostates_leaf_set.append(mstate_leaf_i)

    n_macrostates = len(macrostates_set)
    macrostates_assignment = np.zeros((n_macrostates, nstates))
    for idx, mstate in enumerate(macrostates_set):
        macrostates_assignment[idx][list(mstate)] = 1

    macrostates = np.empty(len(microstates), dtype=np.int64)
    for idx, microstate in enumerate(microstates):
        for idx_m, macroset in enumerate(macrostates_leaf_set):
            if microstate in macroset:
                macrostates[idx] = idx_m + 1
                break
        else:
            print(f'{microstate} not in macrostate')

    return macrostates, macrostates_assignment


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

    return dyn_corr_macrostates


if __name__ == '__main__':
    macrotraj()
