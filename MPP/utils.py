"""
utils.py
========

Utilities for MPP.
"""

import numpy as np
import numpy.typing as npt
import mdtraj as md
from tqdm import tqdm

from collections.abc import Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import MPP


def translate_trajectory(
    trajectory: npt.NDArray[np.int_], state_map: npt.NDArray[np.int_]
) -> npt.NDArray[np.int_]:
    """
    Transform trajectory to other state names.

    Parameters
    ----------
    trajectory : ndarray of int
        Original state trajectory.
    state_map : ndarray of int
        Index is original state; value at that position is the new state name.

    Returns
    -------
    ndarray of int
        Translated trajectory.
    """
    macrostates = np.unique(state_map)
    if state_map.max() < 2**8:
        macrostate_trajectory_type = np.uint8
    elif state_map.max() < 2**16:
        macrostate_trajectory_type = np.uint16
    else:
        macrostate_trajectory_type = np.uint32

    macrostate_trajectory = np.zeros(trajectory.shape, dtype=macrostate_trajectory_type)
    for macrostate in macrostates:
        macrostate_trajectory[np.isin(trajectory, np.where(state_map == macrostate)[0])] = (
            macrostate
        )
    return macrostate_trajectory


def macrostate_tmat(tmat, macrostate_assignment, pop):
    """Transform a microstate transition matrix to macrostate resolution.

    Aggregates transition probabilities by population-weighting within each
    macrostate, then normalizes column-wise.

    Parameters
    ----------
    tmat : ndarray of float, shape (n_microstates, n_microstates)
        Microstate transition matrix.
    macrostate_assignment : ndarray of bool, shape (n_macrostates, n_microstates)
        Boolean mask assigning each microstate to a macrostate.
    pop : ndarray of int or float, shape (n_microstates,)
        Microstate populations.

    Returns
    -------
    ndarray of float, shape (n_macrostates, n_macrostates)
        Column-normalized macrostate transition matrix.
    """
    n_macrostates = macrostate_assignment.shape[0]
    m_tmat = np.zeros((n_macrostates, n_macrostates), dtype=tmat.dtype.type)
    for i, macrostate_mask in enumerate(macrostate_assignment):
        for j, other_macrostate_mask in enumerate(macrostate_assignment):
            m_tmat[i, j] = (tmat[macrostate_mask][:, other_macrostate_mask] * np.expand_dims(pop[macrostate_mask], -1)).sum()
    return m_tmat / m_tmat.sum(axis=0)


def get_grid_format(n):
    """Compute a near-square grid layout for n panels.

    Parameters
    ----------
    n : int
        Number of panels to arrange.

    Returns
    -------
    n_rows : int
        Number of rows.
    n_cols : int
        Number of columns.
    """
    sqrt = np.sqrt(n)
    y = int(sqrt)
    x = y
    if x < sqrt:
        x += 2
    if (x - 1) * y >= n:
        x -= 1
    return x, y


def gmrq(tmat):
    """Compute the generalized matrix Rayleigh quotient for a set of transition matrices.

    Returns the sum of the 2nd through 4th largest eigenvalues for each
    transition matrix.

    Parameters
    ----------
    tmat : ndarray of float, shape (n_runs, n_states, n_states)
        Array of transition matrices.

    Returns
    -------
    ndarray of float, shape (n_runs,)
        GMRQ value for each transition matrix.
    """
    q = np.zeros(len(tmat))
    for i, t in enumerate(tmat):
        val, vec = np.linalg.eig(t)
        val.sort()
        q[i] = val[-4:-1].sum()
    return q


def gmrq2(tmat):
    """Compute the sum of squares of the 2nd through 4th largest eigenvalues.

    Parameters
    ----------
    tmat : ndarray of float, shape (n_runs, n_states, n_states)
        Array of transition matrices.

    Returns
    -------
    ndarray of float, shape (n_runs,)
        Sum of squared eigenvalues for each transition matrix.
    """
    q = np.zeros(len(tmat))
    for i, t in enumerate(tmat):
        val, vec = np.linalg.eig(t)
        val.sort()
        q[i] = (val[-4:-1] ** 2).sum()
    return q


def Z_to_linkage(Z):
    """Convert an MPP Z matrix to legacy linkage format.

    In the MPP Z matrix, each merge produces a new intermediate state with
    index ``n_states + i``. The legacy linkage format does not introduce new
    indices; instead, the merged state inherits ``state_b``'s index. This
    function replaces all references to intermediate indices with the
    corresponding ``state_b`` index and converts to 1-based indexing.

    Parameters
    ----------
    Z : ndarray of float, shape (n_states-1, 4)
        Z matrix in MPP format with columns
        ``[state_a, state_b, metastability, joint_pop]``, using 0-based
        indices where merged states receive new indices ``n_states + i``.

    Returns
    -------
    ndarray of float, shape (n_states-1, 3)
        Legacy linkage matrix with columns ``[state_a, state_b, distance]``
        using 1-based indices, where the merged state is identified by
        ``state_b``'s index rather than a new intermediate index.
    """
    linkage = Z[:, :3].copy()
    for i, row in enumerate(linkage):
        mask = np.where(linkage[:, :2] == i + Z.shape[0] + 1)
        linkage[:, :2][mask] = row[1]
    linkage[:, :2] += 1
    return linkage


def linkage_to_Z(linkage, pop):
    """Convert a legacy linkage matrix to MPP Z matrix format.

    In the legacy linkage format, the merged state inherits ``state_b``'s
    index rather than receiving a new intermediate index. This function
    converts to the MPP Z matrix format by replacing those reused indices
    with proper intermediate indices ``n_states + i`` and appending the
    joint population as the fourth column.

    Parameters
    ----------
    linkage : array-like, shape (n_states-1, 3)
        Legacy linkage matrix with columns ``[state_a, state_b, distance]``
        using 1-based indices, where the merged state is identified by
        ``state_b``'s index.
    pop : ndarray of int, shape (n_states,)
        Microstate populations.

    Returns
    -------
    Z : ndarray of float, shape (n_states-1, 4)
        Z matrix in MPP format with columns
        ``[state_a, state_b, metastability, joint_pop]``, using 0-based
        indices where merged states receive new indices ``n_states + i``.
    full_pop : ndarray of int, shape (2*n_states-1,)
        Population array for all microstates and intermediate cluster states.
    """
    linkage = np.array(linkage)
    n_states = linkage.shape[0] + 1
    Z = np.zeros((linkage.shape[0], 4))
    Z[:, :3] = linkage[:, :3]
    Z[:, :2] -= 1

    full_pop = np.zeros(2 * n_states - 1, dtype=pop.dtype.type)
    full_pop[:n_states] = pop
    for i in range(len(linkage[:-1])):
        new_state = n_states + i
        old_state = Z[i, 1]
        full_pop[new_state] = full_pop[[Z[i, 0].astype(int), int(old_state)]].sum()
        Z[i + 1 :, :2][np.where(Z[i + 1 :, :2] == old_state)] = new_state
    full_pop[-1] = full_pop[Z[i + 1, :2].astype(int)].sum()
    Z[:, 3] = full_pop[n_states:]
    return Z, full_pop


def merge_states(tmat, states, new_state, full_pop, reset_states=True):
    """Merge two states into a new state in the full transition matrix.

    Updates the full transition matrix in-place by combining rows and columns
    of the merged states using population-weighted averaging (outgoing
    transitions) and summation (incoming transitions).

    Parameters
    ----------
    tmat : ndarray of float, shape (2*n_states-1, 2*n_states-1)
        Full transition matrix, modified in-place.
    states : list or ndarray of int
        Indices of the two states to merge.
    new_state : int
        Index of the resulting merged state.
    full_pop : ndarray of int
        Population array for all states, modified in-place.
    reset_states : bool, optional
        If True, zero out the rows and columns of the merged states.
        (default True)

    Returns
    -------
    tmat : ndarray of float
        Updated transition matrix.
    full_pop : ndarray of int
        Updated population array.
    """
    full_pop[new_state] = full_pop[states].sum()

    tmat[new_state] = (tmat[states] * full_pop[states, np.newaxis]).sum(
        axis=0
    ) / full_pop[new_state]

    tmat[:, new_state] = tmat[:, states].sum(axis=1)
    if reset_states:
        tmat[:, states] = 0
        tmat[states, :] = 0
    return tmat, full_pop


def calc_full_tmat(tmat, pop, Z):
    """Compute the full transition matrix for all states and merges encoded in Z.

    Replays the merging sequence encoded in ``Z`` to build a full transition
    matrix of shape ``(2*n_states-1, 2*n_states-1)`` for each run, including
    all intermediate cluster states.

    Parameters
    ----------
    tmat : ndarray of float, shape (n_states, n_states)
        Microstate transition matrix.
    pop : ndarray of int, shape (n_states,)
        Microstate populations.
    Z : ndarray of float, shape (n_states-1, 4) or (n_runs, n_states-1, 4)
        Z matrix encoding the merging sequence. If 2D, treated as a single run.

    Returns
    -------
    full_tmat : ndarray of float, shape (n_runs, 2*n_states-1, 2*n_states-1)
        Full transition matrix for each run including intermediate states.
    full_pop : ndarray of uint32, shape (n_runs, 2*n_states-1)
        Full population array for each run.
    """
    # Ensure that Z is 3D
    if Z.ndim == 2:
        Z = Z.reshape((1, *Z.shape))

    # Initialize full_tmat and full_pop
    n_states = tmat.shape[0]
    full_dim = 2 * n_states - 1
    n_runs = Z.shape[0]
    full_tmat = np.empty((n_runs, full_dim, full_dim))
    full_pop = np.empty((n_runs, full_dim), dtype=np.uint32)

    full_tmat[:, :n_states, :n_states] = tmat
    full_pop[:, :n_states] = pop

    for run, z in enumerate(Z):
        for i, (origin, target) in enumerate(z[:, :2].astype(int)):
            full_tmat[run], full_pop[run] = merge_states(
                full_tmat[run],
                [origin, target],
                n_states + i,
                full_pop[run],
                reset_states=False,
            )
    return full_tmat, full_pop


def Z_to_mask(Z):
    """
    Calculate the mask for each lumping step.

    Parameters
    ----------
    Z : ndarray of float, shape (N, 4)
        Z matrix encoding the merging sequence.

    Returns
    -------
    ndarray of bool, shape (N, 2*n-1)
        Boolean mask for each lumping step.
    """
    n1 = Z.shape[0]
    n = n1 + 1
    m = np.zeros((n1, 2 * n - 1), dtype=bool)
    m[0, :n] = True
    for k, (i, j) in enumerate(Z[:-1, :2].astype(int)):
        m[k + 1] = m[k]
        m[k + 1, [i, j]] = False
        m[k + 1, k + n] = True
    return m


def get_macrostate_assignment_from_tree(tree):
    """Extract macrostate assignment from the lumping tree.

    Parses the binary lumping tree to produce a boolean assignment matrix
    mapping each macrostate to its constituent microstates, ordered by
    decreasing metastability.

    Parameters
    ----------
    tree : BinaryTreeNode
        Root of the lumping tree as produced by ``core.cluster``.

    Returns
    -------
    ndarray of bool, shape (n_macrostates, n_microstates)
        Macrostate assignment matrix. Entry ``[i, j]`` is True if microstate
        ``j`` belongs to macrostate ``i``. Macrostates are ordered by
        decreasing metastability.
    """
    macrostate_order = [l.assigned_macrostate.name for l in tree.leaves]
    macrostates = {l.assigned_macrostate for l in tree.leaves}
    q_ma = np.array([(m.name, m.feature) for m in macrostates])
    ma_order = np.argsort(q_ma[:, 1])[::-1]
    # Dict to translate from n+i numbering to actual macrostate numbers.
    full2real = {f: r for r, f in enumerate(q_ma[ma_order, 0])}
    macrostate_assignment = np.full((len(macrostates), len(macrostate_order)), False)
    macrostate_assignment[
        [full2real[m] for m in macrostate_order], np.arange(len(macrostate_order))
    ] = True
    reorder_microstates = np.zeros(len(macrostate_order), dtype=int)
    reorder_microstates[[l.name for l in tree.leaves]] = np.arange(
        len(macrostate_order)
    )
    return macrostate_assignment[:, reorder_microstates]


def similarity(ref, sto):
    """Compute pairwise similarity between a reference and stochastic lumping.

    For each macrostate in the reference lumping and each run in the stochastic
    lumping, computes three population-weighted overlap measures: Jaccard
    (union-based), recall (reference-based), and precision (lumping-based).

    Parameters
    ----------
    ref : MPP.Lumping
        Reference lumping (deterministic, single run).
    sto : MPP.Lumping
        Stochastic lumping with multiple runs.

    Returns
    -------
    ndarray of float, shape (3, n_macrostates_ref, n_runs)
        Axis 0: ``[Jaccard similarity, recall, precision]``.
        Axis 1: reference macrostates.
        Axis 2: stochastic runs.
    """
    # Similarity matrix
    S = np.zeros((3, ref.n_macrostates[0], sto.n_runs))

    for run_index in range(sto.n_runs):
        ref_ma = ref.macrostate_assignment[0].astype(bool)
        sto_ma = sto.macrostate_assignment[run_index].astype(bool)
        for i in range(ref.n_macrostates[0]):
            for j in range(sto.n_macrostates[run_index]):
                intersect = (
                    np.logical_and(ref_ma[i], sto_ma[j])
                    * ref.full_pop[0, : ref.n_states]
                ).sum()
                union = (
                    np.logical_or(ref_ma[i], sto_ma[j])
                    * ref.full_pop[0, : ref.n_states]
                ).sum()
                # union
                S[0, i, run_index] = max(S[0, i, run_index], intersect / union)
                # reference
                S[1, i, run_index] = max(
                    S[1, i, run_index],
                    intersect / (ref_ma[i] * ref.full_pop[0, : ref.n_states]).sum(),
                )
                # lumping
                S[2, i, run_index] = max(
                    S[2, i, run_index],
                    intersect / (sto_ma[j] * ref.full_pop[0, : ref.n_states]).sum(),
                )
    return S


def shannon_entropy(p):
    """Compute the normalized Shannon entropy of a probability distribution.

    Parameters
    ----------
    p : ndarray of float
        Non-negative values; normalized to sum to 1 internally.

    Returns
    -------
    float
        Shannon entropy normalized by ``log(n)``, where ``n`` is the length
        of ``p``. Returns values in ``[0, 1]``.
    """
    p = p / sum(p)
    return -(p * np.log(p)).sum() / np.log(p.shape[0])


def weighting_function(dq):
    """Transform divergences to similarity weights using a Gaussian kernel.

    For a single value, returns ``exp(-dq)``. For multiple values, applies a
    Gaussian kernel: ``exp(-dq^2 / (2 * var(dq)))``.

    Parameters
    ----------
    dq : ndarray of float
        Array of divergence or distance values.

    Returns
    -------
    ndarray of float
        Similarity weights; larger divergence yields smaller weight.
    """
    if dq.shape[0] == 1:
        return np.exp(-dq)
    sigma2 = np.var(dq)
    return np.exp(-(dq**2) / (2 * sigma2))


### RMSD #####################################################################


def argmedian(x):
    """Return the index of the approximate median value in an array.

    Parameters
    ----------
    x : ndarray
        Input array.

    Returns
    -------
    int
        Index of the approximate median element.
    """
    return np.argpartition(x, len(x) // 2)[len(x) // 2]


def load_trajectory(
    topfile,
    trajectoryfile,
    atom_selection: str = "all",
    frames: npt.NDArray = None,
    stride: int = 1,
):
    """Load an MD trajectory from XTC and topology files.

    Parameters
    ----------
    topfile : str or path-like
        Path to the topology file.
    trajectoryfile : str or path-like
        Path to the XTC trajectory file.
    atom_selection : str, optional
        MDTraj atom selection string. (default ``'all'``)
    frames : ndarray of int, optional
        Specific frame indices to load. If None, loads all frames with the
        given stride. (default None)
    stride : int, optional
        Load every ``stride``-th frame when ``frames`` is None. If ``frames``
        is provided, frame indices are multiplied by ``stride``. (default 1)

    Returns
    -------
    md.Trajectory
        Loaded trajectory.
    """
    print("Loading trajectory...")
    top = md.load_topology(topfile)
    if frames is None:
        return md.load_xtc(
            trajectoryfile,
            top=top,
            atom_indices=top.select(atom_selection),
            stride=stride,
        )
    else:
        if stride > 1:
            frames *= stride
        return md.join(
            [
                md.load_xtc(
                    trajectoryfile,
                    top=top,
                    atom_indices=top.select(atom_selection),
                    frame=frame,
                )
                for frame in frames
            ]
        )


def load_mean_frames(topfile, trajectoryfile, mean_frames, dt=0.1):
    """Load mean representative frames from an XTC trajectory.

    Parameters
    ----------
    topfile : str or path-like
        Path to the topology file.
    trajectoryfile : str or path-like
        Path to the XTC trajectory file.
    mean_frames : list of md.Trajectory
        Single-frame trajectories whose time stamps indicate which frames to
        load.
    dt : float, optional
        Time step in ps used to convert time stamps to frame indices.
        (default 0.1)

    Returns
    -------
    md.Trajectory
        Trajectory containing only the requested mean frames.
    """
    top = md.load_topology(topfile)
    idxs = [int(frame.time[0]) / dt for frame in mean_frames]
    trajectory = md.join(
        [md.load_xtc(trajectoryfile, top=top, frame=frame) for frame in idxs]
    )
    return trajectory


def find_mean_frame(trajectory, estimator=np.argmin):
    """Find a representative mean frame from a collection of trajectory frames.

    Selects the frame with the smallest (or otherwise estimated) mean RMSD to
    all other frames in the joined trajectory.

    Parameters
    ----------
    trajectory : list of md.Trajectory or md.Trajectory
        Input trajectory frames.
    estimator : callable, optional
        Function applied to the mean RMSD array to select the representative
        frame index. (default ``numpy.argmin``)

    Returns
    -------
    mean_frame : md.Trajectory
        The selected representative frame.
    index : ndarray
        Index of the selected frame within the joined trajectory.
    """
    trajectory = md.join(trajectory)
    mean_rmsd = np.array([estimate_rmsd(frame, trajectory) for frame in trajectory])
    index_mean_frame = estimator(mean_rmsd)
    mean_frame = trajectory[index_mean_frame]
    return mean_frame, np.array(index_mean_frame)


def estimate_rmsd(frame, trajectory):
    """Compute mean RMSD of a reference frame against a trajectory.

    Parameters
    ----------
    frame : md.Trajectory
        Reference frame.
    trajectory : md.Trajectory
        Trajectory to compute RMSD against.

    Returns
    -------
    float
        Mean RMSD over all frames in the trajectory.
    """
    rmsd = md.rmsd(
        trajectory,
        frame,
    )
    return np.mean(rmsd)


def find_mean_frame_feature(trajectory, estimator=np.argmin):
    """Find a representative mean frame from feature-space trajectories.

    Parameters
    ----------
    trajectory : array-like of ndarray
        Collection of feature vectors.
    estimator : callable, optional
        Function applied to the distance array to select the representative
        frame index. (default ``numpy.argmin``)

    Returns
    -------
    mean_frame : ndarray
        The selected representative feature vector.
    index : ndarray
        Index of the selected frame within the stacked trajectory.
    """
    trajectory = np.array(trajectory)
    mean_rmsd = np.array(
        [estimate_rmsd_feature(frame, trajectory) for frame in trajectory]
    )
    index_mean_frame = estimator(mean_rmsd)
    mean_frame = trajectory[index_mean_frame]
    return mean_frame, np.array(index_mean_frame)


def estimate_rmsd_feature(frame, trajectory):
    """Compute RMS deviation between a reference feature vector and a trajectory.

    Parameters
    ----------
    frame : ndarray
        Reference feature vector.
    trajectory : ndarray of float, shape (N, M)
        Feature trajectory with N frames and M features.

    Returns
    -------
    float
        Root-mean-square deviation from the reference.
    """
    return np.sqrt(((trajectory - frame) ** 2).sum() / (len(trajectory) - 1))


def align_trajectory_to_reference(trajectory, reference):
    """
    Align each frame in the trajectory to the reference frame using the Kabsch algorithm.

    Parameters
    ----------
    trajectory : ndarray of float, shape (N, M, 3)
        N frames with M atoms to align.
    reference : ndarray of float, shape (1, M, 3)
        Reference frame to align to.

    Returns
    -------
    ndarray of float, shape (N, M, 3)
        Trajectory with each frame aligned to the reference.
    """

    # Extract the reference frame (since reference is of shape (1, 35, 3),
    # we need to squeeze it to (35, 3))
    reference_frame = reference.squeeze()

    # Compute the centroid (mean) of the reference points
    reference_centroid = np.mean(reference_frame, axis=0)

    # Center the reference points by subtracting the centroid
    centered_reference = reference_frame - reference_centroid

    # Initialize the array to store the aligned trajectory
    aligned_trajectory = np.zeros_like(trajectory)

    # Iterate through each frame in the trajectory
    for i in range(trajectory.shape[0]):
        # Extract the current frame
        current_frame = trajectory[i]

        # Compute the centroid of the current frame
        frame_centroid = np.mean(current_frame, axis=0)

        # Center the current frame by subtracting the centroid
        centered_frame = current_frame - frame_centroid

        # Compute the covariance matrix
        H = np.dot(centered_frame.T, centered_reference)

        # Compute the Singular Value Decomposition (SVD)
        U, S, Vt = np.linalg.svd(H)

        # Compute the optimal rotation matrix
        R = np.dot(Vt.T, U.T)

        # Handle special reflection case where the determinant of R is -1
        if np.linalg.det(R) < 0:
            Vt[2, :] *= -1
            R = np.dot(Vt.T, U.T)

        # Apply the rotation to the centered frame
        rotated_frame = np.dot(centered_frame, R)

        # Re-add the reference centroid to align the trajectory in the reference coordinate system
        aligned_trajectory[i] = rotated_frame + reference_centroid

    return aligned_trajectory


def calc_var(
    ref: npt.NDArray[np.floating], trajectory: npt.NDArray[np.floating]
) -> npt.NDArray[np.floating]:
    """Compute per-atom RMSD between a reference frame and a trajectory.

    Aligns the trajectory to the reference using the Kabsch algorithm before
    computing deviations.

    Parameters
    ----------
    ref : ndarray of float, shape (1, M, 3) or (M, 3)
        Reference frame with M atoms.
    trajectory : ndarray of float, shape (N, M, 3)
        Trajectory to compare; aligned to ``ref`` before computing.

    Returns
    -------
    ndarray of float, shape (M,)
        Per-atom root-mean-square deviation averaged over all N frames.
    """
    aligned_trajectory = align_trajectory_to_reference(trajectory, ref)
    d_square = ((aligned_trajectory - ref) ** 2).sum(axis=2)
    return np.sqrt(d_square.mean(axis=0))


def calc_var_feature(
    ref: npt.NDArray[np.floating], trajectory: npt.NDArray[np.floating]
) -> npt.NDArray[np.floating]:
    """Compute per-feature RMS deviation between a reference and a trajectory.

    Parameters
    ----------
    ref : ndarray of float, shape (M,) or (1, M)
        Reference feature vector.
    trajectory : ndarray of float, shape (N, M)
        Feature trajectory with N frames and M features.

    Returns
    -------
    ndarray of float, shape (M,)
        Per-feature root-mean-square deviation averaged over all N frames.
    """
    return np.sqrt(((trajectory - ref) ** 2).mean(axis=0))


def opt_num_batches(n):
    """Compute the optimal number of batches for batched RMSD calculation.

    Parameters
    ----------
    n : int
        Number of frames.

    Returns
    -------
    int
        Optimal batch count, approximately ``n^(2/3) / 2^(1/3)``.
    """
    return int(np.cbrt(n**2 / 2))


def _calc_rmsd_generic(
    lumping: "MPP.Lumping",
    get_traj: Callable[["MPP.Lumping"], md.Trajectory | npt.NDArray],
    find_mean: Callable[
        [npt.NDArray, Callable[[npt.NDArray], float]], (npt.NDArray, npt.NDArray)
    ],
    calc_var_fn: Callable[[npt.NDArray, npt.NDArray], npt.NDArray],
    estimator: Callable[[npt.NDArray], float] = np.argmin,
    quiet: bool = False,
):
    """Calculate the RMSD for different coordinates

    Parameters
    ----------
    lumping : MPP.Lumping
        The lumping to calculate the RMSD for
    get_traj : Callable[MPP.Lumping]
        Loader for the trajectory
    find_mean : Callable[npt.NDArray, Callable[npt.NDArray]]
        A function that determines some mean frame
    calc_var_fn : Callable[npt.NDArray, npt.NDArray]
        A function that calculates the RMSD of a reference to a
        trajectory
    estimator : Callable[npt.NDArray]
        The estimator of the mean frame. Determines a representative
        frame for the given trajectory.
    quiet : bool
        If False: Print the macrostate which is being processed
    """
    t = get_traj(lumping)
    mean_frames = []
    mean_frames_idx = []
    if isinstance(t, np.ndarray):
        n_features = t.shape[1]
    elif isinstance(t, md.Trajectory):
        n_features = t.n_atoms

    rmsd = np.empty([lumping.n_macrostates[lumping.run_index], n_features])

    for j in range(lumping.n_macrostates[lumping.run_index]):
        if not quiet:
            print(f"Process macrostate {j + 1}")
        traj_mask = lumping.macrostate_trajectory[lumping.run_index] == j
        tm = t[traj_mask]
        m_frames = []
        m_frames_idx = []

        # Batched run for speed
        n_batches = opt_num_batches(lumping.macrostate_population[lumping.run_index][j])
        for i in tqdm(range(n_batches)) if not quiet else range(n_batches):
            mean_frame, idx = find_mean(tm[i::n_batches], estimator)
            m_frames.append(mean_frame)
            m_frames_idx.append(idx)

        # Best frame from all batches
        mean_frame, idx_batch = find_mean(m_frames, estimator)
        mean_frames.append(mean_frame)

        # Convert back to full trajectory index
        index_macro = m_frames_idx[idx_batch] * n_batches + idx_batch
        index_mean_frame = np.where(traj_mask)[0][index_macro]
        mean_frames_idx.append(index_mean_frame)

        rmsd[j] = calc_var_fn(mean_frames[j], tm)

    return rmsd, np.array(mean_frames_idx)


# Specializations
def calc_rmsd(lumping, estimator=np.argmin, quiet=True):
    """Calculate per-atom RMSD across macrostates using C-alpha atoms.

    Loads the XTC trajectory with C-alpha atom selection and computes per-atom
    RMSD for each macrostate in the current run.

    Parameters
    ----------
    lumping : MPP.Lumping
        Lumping object with XTC/topology file paths and macrostate assignments.
    estimator : callable, optional
        Function to select the representative frame from RMSD values.
        (default ``numpy.argmin``)
    quiet : bool, optional
        If True, suppress progress output. (default True)

    Returns
    -------
    rmsd : ndarray of float, shape (n_macrostates, n_atoms)
        Per-atom RMSD for each macrostate.
    mean_frames_idx : ndarray of int
        Trajectory indices of the representative frame for each macrostate.
    """
    def get_traj(lumping):
        return load_trajectory(
            lumping.topology_file,
            lumping.xtc_trajectory_file,
            atom_selection="name CA",
            stride=lumping.xtc_stride,
        )

    return _calc_rmsd_generic(
        lumping,
        get_traj,
        find_mean_frame,
        lambda mean, tm: calc_var(mean.xyz, tm.xyz),
        estimator=estimator,
        quiet=quiet,
    )


def calc_rmsd_feature(lumping, estimator=np.argmin, quiet=True):
    """Calculate per-feature RMSD across macrostates using the feature trajectory.

    Parameters
    ----------
    lumping : MPP.Lumping
        Lumping object with ``multi_feature_trajectory`` and macrostate
        assignments.
    estimator : callable, optional
        Function to select the representative frame from RMSD values.
        (default ``numpy.argmin``)
    quiet : bool, optional
        If True, suppress progress output. (default True)

    Returns
    -------
    rmsd : ndarray of float, shape (n_macrostates, n_features)
        Per-feature RMSD for each macrostate.
    mean_frames_idx : ndarray of int
        Trajectory indices of the representative frame for each macrostate.
    """
    return _calc_rmsd_generic(
        lumping,
        lambda l: l.multi_feature_trajectory,
        find_mean_frame_feature,
        calc_var_feature,
        estimator=estimator,
        quiet=quiet,
    )


def find_state_lengths(arr):
    """Compute run-length encoding of a state trajectory.

    Parameters
    ----------
    arr : array-like
        Sequence of state labels.

    Returns
    -------
    unique_states : ndarray
        Ordered sequence of states as they appear in ``arr``.
    lengths : ndarray of int
        Consecutive length of each state run.
    """
    # Lists to store unique states and their consecutive counts
    unique_states = []
    lengths = []

    # Initialize the first state and its count
    current_state = arr[0]
    count = 1

    # Iterate over the array from the second element onward
    for value in arr[1:]:
        if value == current_state:
            # Increment count if the state is the same
            count += 1
        else:
            # Append the state and its count when a new state is encountered
            unique_states.append(current_state)
            lengths.append(count)
            # Update the current state and reset count
            current_state = value
            count = 1

    # Append the last state and its count
    unique_states.append(current_state)
    lengths.append(count)

    return np.array(unique_states), np.array(lengths)


def get_multi_state_trajectory(trajectories: npt.NDArray, limits: npt.NDArray):
    """Split a concatenated trajectory into per-segment sub-trajectories.

    Parameters
    ----------
    trajectories : ndarray
        Concatenated trajectory array.
    limits : ndarray of int or None
        Length of each segment. If None, the full trajectory is returned
        unchanged.

    Returns
    -------
    list of ndarray or ndarray
        List of sub-trajectories, one per segment, or the original array if
        ``limits`` is None.
    """
    if limits is None:
        return trajectories
    trajectory_collection = []
    current_position = 0
    for limit in limits:
        trajectory_collection.append(
            trajectories[current_position : int(current_position + limit)]
        )
        current_position += limit
    return trajectory_collection
