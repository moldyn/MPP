"""
utils.py
========

Utilities for MPT.
"""

import os
import numpy as np
from numba import njit
from itertools import combinations
from typing import List
from numpy.typing import NDArray
import scipy as scy
import msmhelper as mh
import mdtraj as md
from tqdm import tqdm

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
    q = np.zeros(len(tmat))
    for i, t in enumerate(tmat):
        val, vec = np.linalg.eig(t)
        q[i] = val[:3].sum()
    return q

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

def merge_states(tmat, states, new_state, full_pop, reset_states=True):
    full_pop[new_state] = full_pop[states].sum()

    tmat[new_state] = (
        tmat[states] * full_pop[states, np.newaxis]
    ).sum(axis=0) / full_pop[new_state]

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
            # full_tmat[run, n_states+i] = fm[n_states+i]
            # full_tmat[run, :, n_states+i] = fm[:, n_states+i]
    return full_tmat, full_pop

def Z_to_mask(Z):
    """
    Calculate the mask for each lumping step.
    Z (Nx4): Z matric
    """
    # n1 = n-1
    n1 = Z.shape[0]
    n = n1 + 1
    m = np.zeros((n1, 2 * n - 1), dtype=bool)
    m[0, :n] = True
    for k, (i, j) in enumerate(Z[:, :2].astype(int)):
        m[k, [i, j]] = False
        m[k, k + n] = True
        if k < n-2:
            m[k+1] = m[k]
    return m

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

def kullback_leibler(transitions, tmat, epsilon=1e-6):
    """Return Kallback-Leibler probability"""
    kl = scy.stats.entropy(transitions + epsilon, tmat + epsilon, axis=1)
    return scy.special.softmax(-kl)

# def kullback_leibler(transitions, tmat, epsilon=1e-6):
#     """Return Kallback-Leibler probability"""
#     tmat = tmat.copy()
#     tmat += epsilon
#     transitions = transitions.copy()
#     smoothed_tmat = tmat / np.expand_dims(tmat.sum(axis=1), axis=1)
#     kl = scy.stats.entropy(transitions, smoothed_tmat, axis=1)
#     return scy.special.softmax(-kl)

def dq_kl(ref, s, e=1e-6):
    k = scy.stats.entropy(ref+e, s+e, axis=1)
    e_kl = np.exp(-k)
    return 1 - e_kl

def jensen_shannon_div(p, q):
    m = (p + q) / 2
    return (scy.stats.entropy(p, m, axis=1) + scy.stats.entropy(q, m, axis=1)) / 2

def jensen_shannon(p, q):
    if p.ndim == 1:
        p = np.expand_dims(p, axis=0)
    if q.ndim == 1:
        q = np.expand_dims(q, axis=0)
    js = scy.spatial.distance.jensenshannon(p, q, axis=1) ** 2
    return scy.special.softmax(-js)

def shannon_entropy(p):
    p = p / sum(p)
    return -(p * np.log(p)).sum() / np.log(p.shape[0])

def weighting_function(dq):
    if dq.shape[0] == 1:
        return np.exp(-dq)
    sigma = np.sqrt(np.var(dq))
    return np.exp(-dq**2 / (2 * sigma**2))

### Delta function for correlation plot ######################################

def dq_kernel_P(full_tmat, mask=None):
    """Kernel for transition probabilities. For reverse direction submit transposed tmat"""
    if mask is None:
        mask = np.full(full_tmat.shape[0], True)
    n = mask.sum()
    mask_id = np.where(mask)[0]
    idx = np.where(np.tri(n, n, -1).T)
    return full_tmat[mask_id[idx[0]], mask_id[idx[1]]]

def dq_kernel_fnc(full_feature, mask=None):
    """Kernel for difference in fraction of native contacts"""
    if mask is None:
        mask = np.full(full_feature.shape[0], True)
    n = mask.sum()
    c = list(combinations(range(n), 2))
    return abs(np.diff(full_feature[mask][c]).flatten())

def dq_kernel_KLP(full_tmat, mask=None):
    """Kerenl for Kullback-Leibler probabilities"""
    if mask is None:
        mask = np.full(full_tmat.shape[0], True)
    tmat = full_tmat[np.ix_(mask, mask)]
    r = np.roll(np.arange(mask.sum()-1, -1, -1), 1).cumsum()
    start = r[:-1]
    end = r[1:]
    klp = np.empty(r[-1])
    for i, trans_probs in enumerate(tmat[:-1]):
        t = tmat.copy()
        np.fill_diagonal(t, trans_probs)
        t[:, i] = trans_probs[i]
        kl = kullback_leibler(trans_probs, t[i+1:])
        # Renormalize
        if True:
            if kl.shape[0] > 1:
                kl = kl - kl.min()
                kl = kl / kl.sum()
        klp[start[i]:end[i]] = kl
    return klp

def dq_kernel_JSC(full_feature, mask=None):
    """Kernel for Jensen-Shannon contacts"""
    if mask is None:
        mask = np.full(full_feature.shape[0], True)
    feature = full_feature[np.ix_(mask)]
    r = np.roll(np.arange(mask.sum()-1, -1, -1), 1).cumsum()
    start = r[:-1]
    end = r[1:]
    jsc = np.empty(r[-1])
    for i, feature_state in enumerate(feature[:-1]):
        js = jensen_shannon(feature_state, feature[i+1:])
        # Renormalize
        if True:
            if js.shape[0] > 1:
                js = js - js.min()
                js = js / js.sum()
        jsc[start[i]:end[i]] = js
    return jsc

def dq_kernel_pop(full_pop, mask=None):
    """Kernel for sum of population of merged states"""
    if mask is None:
        mask = np.full(full_pop.shape[0], True)
    n = mask.sum()
    c = list(combinations(range(n), 2))
    return full_pop[mask][c].sum(axis=1)

def dq_kernel_origin_pop(full_pop, mask=None):
    """Kernel for sum of population of merged states"""
    if mask is None:
        mask = np.full(full_pop.shape[0], True)
    n = mask.sum()
    c = np.array(list(combinations(range(n), 2))).T[0]
    return full_pop[mask][c]

def dq(full_feature, Z=None, similarity="P"):
    """
    Calculate dq for a given feature.

    feature (np.ndarray): array containing the data
    Z (np.ndarray): Z matrix (2D, only for one run); if None: calculate dq
        once for entire full_feature
    similarity (str):
        P: transition probability (full_tmat)
        fnc: difference in fnc (full_feature)
        KLP: Kullback-Leibler probabilities (full_tmat)
        JSC: Jensen-Shannon contacts (full_feature)
        pop: population (full_pop)
        origin pop: population of origin state (full_pop)

    returns a list of arrays, one array for each stage of the lumping
    """
    if similarity == "P":
        dq_kernel = dq_kernel_P
    elif similarity == "fnc":
        dq_kernel = dq_kernel_fnc
    elif similarity == "KLP":
        dq_kernel = dq_kernel_KLP
    elif similarity == "JSC":
        dq_kernel = dq_kernel_JSC
    elif similarity == "pop":
        dq_kernel = dq_kernel_pop
    elif similarity == "origin pop":
        dq_kernel = dq_kernel_origin_pop

    if Z is None:
        return dq_kernel(full_feature)
    else:
        n_states = Z.shape[0] + 1
        mask = np.full(2 * n_states - 1, False)
        mask[:n_states] = True
        stages = []
        for i, (origin, target) in enumerate(Z[:, :2].astype(int)):
            stages.append(dq_kernel(full_feature, mask))
            mask[n_states + i] = True
            mask[[origin, target]] = False
        return stages


### RMSD #####################################################################

def load_traj(topfile, trajfile, atom_selection="all", frames=None):
    print("Loading trajectory...")
    top = md.load_topology(topfile)
    if frames is None:
        return md.load_xtc(trajfile, top=top, atom_indices=top.select(atom_selection))
    else:
        return md.join([md.load_xtc(
            trajfile,
            top=top,
            atom_indices=top.select(atom_selection),
            frame=frame,
        ) for frame in frames])

def load_mean_frames(topfile, trajfile, mean_frames, dt=0.1):
    top = md.load_topology(topfile)
    idxs = [int(frame.time[0]) / dt for frame in mean_frames]
    traj = md.join([md.load_xtc(trajfile, top=top, frame=frame) for frame in idxs])
    return traj

def find_mean_frame(traj):
    mean_rmsd = np.array([
        estimate_rmsd(frame, traj)
        for frame in traj
    ])
    mean_frame = traj[np.argmin(mean_rmsd)]
    return mean_frame

def estimate_rmsd(frame, traj):
    rmsd = md.rmsd(
        traj,
        frame,
    )
    return np.mean(rmsd)

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

def calc_var(ref, traj):
    """Calculate RMSD"""
    aligned_trajectory = align_trajectory_to_reference(traj, ref)
    d = ((aligned_trajectory - ref) ** 2).sum(axis=2)
    return d.mean(axis=0)

def opt_num_batches(n):
    return int(np.cbrt(n ** 2 / 2))

def calc_rmsd(mpt, n_i=None):
    if n_i is None:
        n_i = mpt.n_i
    t = load_traj(mpt.topology_file, mpt.xtc_trajectory_file, atom_selection="name CA")
    mean_frames = []
    rmsd = np.empty([mpt.n_macrostates[n_i], t.n_atoms])
    for j in range(mpt.n_macrostates[n_i]):
        print(f"Process macrostate {j+1}")
        m = (mpt.macrotraj[:, n_i] == j + 1)
        tm = t[m]
        m_frames = []
        n_batches = opt_num_batches(mpt.macro_pop[n_i][j])
        for i in tqdm(range(n_batches)):
            m_frames.append(find_mean_frame(tm[i::n_batches]))
        mean_frames.append(find_mean_frame(md.join(m_frames)))
        rmsd[j] = calc_var(mean_frames[j].xyz, tm.xyz)
    return rmsd, mean_frames

def write_pdbs(out, vars, top, xtctraj, mean_frames):
    mean_frames_traj = load_mean_frames(top, xtctraj, mean_frames, dt=0.1)
    b_factors = np.zeros((mean_frames_traj.n_frames, mean_frames_traj.n_atoms))
    for frame in range(mean_frames_traj.n_frames):
        atms = 0
        for res in mean_frames_traj.topology.residues:
            new_atms = atms + res.n_atoms
            b_factors[frame, atms:new_atms] = vars[frame, res.resSeq-1]
            atms = new_atms

    for i, frame in enumerate(mean_frames_traj):
        frame.save_pdb(os.path.join(out, f"macrostate_{i+1:02d}.pdb"), bfactors=b_factors[i])
    print(f"PyMol commnand: 'spectrum b, blue_white_red, minimum={vars.min():.3f}, maximum={vars.max():.3f}'")

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

def get_multi_state_traj(trajs: np.ndarray, limits: np.ndarray):
    """Load trajectory containing several concatenated trajectories"""
    if limits is None:
        return trajs
    trajectories = []
    current_position = 0
    for l in limits:
        trajectories.append(trajs[current_position:int(current_position+l)])
        current_position += l
    return trajectories

# def multi_state_traj_to_tmat(multi_state_traj, tlag):
#     """Calculate combined transition matrix from multi state traj"""
#     states = np.unique(multi_state_traj)
#     tmat_tot = np.zeros((states.shape[0], states.shape[0]))
#     for traj in multi_state_traj:
#         tmat, s = mh.msm.estimate_markov_model(traj, tlag)
#         tmat_tot[np.ix_(s-1, s-1)] += tmat * traj.shape[0]
#     return (tmat_tot.T / tmat_tot.sum(axis=1)).T

# def load_stata_traj(traj_file: str, limits_file: str=None) -> List:
#     """Load trajectories from file and return list of np.ndarray"""
#     traj = np.loadtxt(traj_file, dtype=np.uint32)
#     if traj.max() < 2**8:
#         traj = traj.astype(np.uint8)
#     elif traj.max() < 2**16:
#         traj = traj.astype(np.uint16)
#
#     if limits_file is None:
#         return [traj]
#     else:
#         return get_multi_state_traj(traj, np.loadtxt(limits_file, dtype=np.int_))

# def load_feature_traj(traj_file: str, limits_file: str=None) -> List:
#     """Load trajectories from file and return list of np.ndarray"""
#     traj = np.loadtxt(traj_file)
#     if limits_file is None:
#         return [traj]
#     else:
#         return get_multi_state_traj(traj, np.loadtxt(limits_file, dtype=np.int_))


def fnc_from_multi_feature_traj(multi_feature_traj):
    return (multi_feature_traj <= 0.45).mean(axis=1)
