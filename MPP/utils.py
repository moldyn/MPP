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
    trajectory: npt.NDArray[np.int_], map: npt.NDArray[np.int_]
) -> npt.NDArray[np.int_]:
    """
    Transform trajectory to other state names.

    trajectory (NDArray[np.int_]): original state trajectory
    map (NDArray[np.int_]): index is original state, value at that position is
            new value

    returns translated trajectory
    """
    macrostates = np.unique(map)
    if map.max() < 2**8:
        macrostate_trajectory_type = np.uint8
    elif map.max() < 2**16:
        macrostate_trajectory_type = np.uint16
    else:
        macrostate_trajectory_type = np.uint32

    macrostate_trajectory = np.zeros(trajectory.shape, dtype=macrostate_trajectory_type)
    for macrostate in macrostates:
        macrostate_trajectory[np.isin(trajectory, np.where(map == macrostate)[0])] = (
            macrostate
        )
    return macrostate_trajectory


def macrostate_tmat(tmat, macrostate_assignment, pop):
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
    if (x - 1) * y >= n:
        x -= 1
    return x, y


def gmrq(tmat):
    """Generalized matrix Rayleigh quotient."""
    q = np.zeros(len(tmat))
    for i, t in enumerate(tmat):
        val, vec = np.linalg.eig(t)
        val.sort()
        q[i] = val[-4:-1].sum()
    return q


def gmrq2(tmat):
    """Sum of the squares of the first three eigenvalues."""
    q = np.zeros(len(tmat))
    for i, t in enumerate(tmat):
        val, vec = np.linalg.eig(t)
        val.sort()
        q[i] = (val[-4:-1] ** 2).sum()
    return q


def Z_to_linkage(Z):
    linkage = Z[:, :3].copy()
    for i, row in enumerate(linkage):
        mask = np.where(linkage[:, :2] == i + Z.shape[0] + 1)
        linkage[:, :2][mask] = row[1]
    linkage[:, :2] += 1
    return linkage


def linkage_to_Z(linkage, pop):
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
    """Calculate full tmat for a give Z matrix"""
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
    Z (Nx4): Z matrix
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
    """Return similarity of two clusterings"""
    # Similarity matrix
    S = np.zeros((3, ref.n_macrostates[0], sto.n_runs))

    for n_i in range(sto.n_runs):
        ref_ma = ref.macrostate_assignment[0].astype(bool)
        sto_ma = sto.macrostate_assignment[n_i].astype(bool)
        for i in range(ref.n_macrostates[0]):
            for j in range(sto.n_macrostates[n_i]):
                intersect = (
                    np.logical_and(ref_ma[i], sto_ma[j])
                    * ref.full_pop[0, : ref.n_states]
                ).sum()
                union = (
                    np.logical_or(ref_ma[i], sto_ma[j])
                    * ref.full_pop[0, : ref.n_states]
                ).sum()
                # union
                S[0, i, n_i] = max(S[0, i, n_i], intersect / union)
                # reference
                S[1, i, n_i] = max(
                    S[1, i, n_i],
                    intersect / (ref_ma[i] * ref.full_pop[0, : ref.n_states]).sum(),
                )
                # lumping
                S[2, i, n_i] = max(
                    S[2, i, n_i],
                    intersect / (sto_ma[j] * ref.full_pop[0, : ref.n_states]).sum(),
                )
    return S


def shannon_entropy(p):
    p = p / sum(p)
    return -(p * np.log(p)).sum() / np.log(p.shape[0])


def weighting_function(dq):
    if dq.shape[0] == 1:
        return np.exp(-dq)
    # sigma = np.sqrt(np.var(dq))
    sigma2 = np.var(dq)
    return np.exp(-(dq**2) / (2 * sigma2))


### RMSD #####################################################################


def argmedian(x):
    return np.argpartition(x, len(x) // 2)[len(x) // 2]


def load_trajectory(
    topfile,
    trajectoryfile,
    atom_selection: str = "all",
    frames: npt.NDArray = None,
    stride: int = 1,
):
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
    top = md.load_topology(topfile)
    idxs = [int(frame.time[0]) / dt for frame in mean_frames]
    trajectory = md.join(
        [md.load_xtc(trajectoryfile, top=top, frame=frame) for frame in idxs]
    )
    return trajectory


def find_mean_frame(trajectory, estimator=np.argmin):
    trajectory = md.join(trajectory)
    mean_rmsd = np.array([estimate_rmsd(frame, trajectory) for frame in trajectory])
    index_mean_frame = estimator(mean_rmsd)
    mean_frame = trajectory[index_mean_frame]
    return mean_frame, np.array(index_mean_frame)


def estimate_rmsd(frame, trajectory):
    """Calculate mean RMSD"""
    rmsd = md.rmsd(
        trajectory,
        frame,
    )
    return np.mean(rmsd)


def find_mean_frame_feature(trajectory, estimator=np.argmin):
    trajectory = np.array(trajectory)
    mean_rmsd = np.array(
        [estimate_rmsd_feature(frame, trajectory) for frame in trajectory]
    )
    index_mean_frame = estimator(mean_rmsd)
    mean_frame = trajectory[index_mean_frame]
    return mean_frame, np.array(index_mean_frame)


def estimate_rmsd_feature(frame, trajectory):
    # return np.sqrt(((trajectory - frame) ** 2).mean())
    return np.sqrt(((trajectory - frame) ** 2).sum() / (len(trajectory) - 1))


def align_trajectory_to_reference(trajectory, reference):
    """
    Aligns each frame in the trajectory array to the reference frame using the Kabsch algorithm.

    Parameters:
    - trajectory: numpy array of shape (N, 35, 3) where N is the number of frames.
    - reference: numpy array of shape (1, 35, 3) representing the reference points.

    Returns:
    - aligned_trajectory: numpy array of shape (N, 35, 3) where each frame is aligned to the reference.
    """

    # Extract the reference frame (since reference is of shape (1, 35, 3), we need to squeeze it to (35, 3))
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
    """Calculate RMSD

    Parameters
    ----------
    ref, trajectory : ndarray of float, shape (N, M, 3)
        N frames, M atoms

    Returns
    -------
    ndarray of float, shape (M,)
    """
    aligned_trajectory = align_trajectory_to_reference(trajectory, ref)
    # d = ((aligned_trajectory - ref) ** 2).sum(axis=2)
    # Calculate the distance
    d_square = ((aligned_trajectory - ref) ** 2).sum(axis=2)
    # return d.mean(axis=0)
    return np.sqrt(d_square.mean(axis=0))


def calc_var_feature(
    ref: npt.NDArray[np.floating], trajectory: npt.NDArray[np.floating]
) -> npt.NDArray[np.floating]:
    return np.sqrt(((trajectory - ref) ** 2).mean(axis=0))


def opt_num_batches(n):
    return int(np.cbrt(n**2 / 2))


# def calc_rmsd(lumping, quiet=False):
#     t = load_trajectory(
#         lumping.topology_file,
#         lumping.xtc_trajectory_file,
#         atom_selection="name CA",
#         stride=lumping.xtc_stride,
#     )
#     mean_frames = []
#     mean_frames_idx = []
#     rmsd = np.empty([lumping.n_macrostates[lumping.n_i], t.n_atoms])
#     for j in range(lumping.n_macrostates[lumping.n_i]):
#         if not quiet:
#             print(f"Process macrostate {j}")
#         traj_mask = lumping.macrostate_trajectory[lumping.n_i] == j
#         tm = t[traj_mask]
#         m_frames = []
#         m_frames_idx = []
#
#         # Batched run for speed
#         n_batches = opt_num_batches(lumping.macrostate_population[lumping.n_i][j])
#         for i in tqdm(range(n_batches)) if not quiet else range(n_batches):
#             mean_frame, index_mean_frame = find_mean_frame(tm[i::n_batches])
#             m_frames.append(mean_frame)
#             m_frames_idx.append(index_mean_frame)
#
#         # Calculate best frame from all batches
#         mean_frame, index_mean_frame_batch = find_mean_frame(md.join(m_frames))
#         mean_frames.append(mean_frame)
#
#         # Transform index_mean_frame_batch to index in entire trajectory (t)
#         index_mean_frame_macrostate = (
#             m_frames_idx[index_mean_frame_batch] * n_batches + index_mean_frame_batch
#         )
#         index_mean_frame = np.where(traj_mask)[0][index_mean_frame_macrostate]
#         mean_frames_idx.append(index_mean_frame)
#         rmsd[j] = calc_var(mean_frames[j].xyz, tm.xyz)
#     return rmsd, np.array(mean_frames_idx)
#
#
# def calc_rmsd_feature(lumping):
#     t = lumping.multi_feature_trajectory
#     mean_frames = []
#     mean_frames_idx = []
#     rmsd = np.empty([lumping.n_macrostates[lumping.n_i], t.shape[1]])
#     for j in range(lumping.n_macrostates[lumping.n_i]):
#         traj_mask = lumping.macrostate_trajectory[lumping.n_i] == j
#         tm = t[traj_mask]
#         m_frames = []
#         m_frames_idx = []
#
#         # Batched run for speed
#         n_batches = opt_num_batches(lumping.macrostate_population[lumping.n_i][j])
#         for i in range(n_batches):
#             mean_frame, index_mean_frame = find_mean_frame_feature(tm[i::n_batches])
#             m_frames.append(mean_frame)
#             m_frames_idx.append(index_mean_frame)
#
#         # Calculate best frame from all batches
#         mean_frame, index_mean_frame_batch = find_mean_frame_feature(np.array(m_frames))
#         mean_frames.append(mean_frame)
#
#         # Transform index_mean_frame_batch to index in entire trajectory (t)
#         index_mean_frame_macrostate = (
#             m_frames_idx[index_mean_frame_batch] * n_batches + index_mean_frame_batch
#         )
#         index_mean_frame = np.where(traj_mask)[0][index_mean_frame_macrostate]
#         mean_frames_idx.append(index_mean_frame)
#         rmsd[j] = calc_var_feature(mean_frames[j], tm)
#     return rmsd, np.array(mean_frames_idx)


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

    rmsd = np.empty([lumping.n_macrostates[lumping.n_i], n_features])

    for j in range(lumping.n_macrostates[lumping.n_i]):
        if not quiet:
            print(f"Process macrostate {j + 1}")
        traj_mask = lumping.macrostate_trajectory[lumping.n_i] == j
        tm = t[traj_mask]
        m_frames = []
        m_frames_idx = []

        # Batched run for speed
        n_batches = opt_num_batches(lumping.macrostate_population[lumping.n_i][j])
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
    return _calc_rmsd_generic(
        lumping,
        lambda l: l.multi_feature_trajectory,
        find_mean_frame_feature,
        calc_var_feature,
        estimator=estimator,
        quiet=quiet,
    )


def find_state_lengths(arr):
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
    """Load trajectory containing several concatenated trajectories"""
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
