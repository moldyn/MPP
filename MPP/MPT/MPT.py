import os
import datetime
import warnings
import numpy as np
import msmhelper as mh
import matplotlib.pyplot as plt
import mdtraj as md

from pathlib import Path
from tqdm import tqdm
from typing import Callable, List
from numpy.typing import NDArray
from collections.abc import Iterable
from sklearn.metrics import davies_bouldin_score
import pygpcca as gp

from MPT import core

# import MPT.core as core
import MPT.utils as utils
import MPT.kernel as kernel_module
from MPT.graph import draw_knetwork

import MPT.plot as plot

# TODO:
# - change traj and macrotraj to list - add one dimension. First, mark all places that need adaptation.
# - Connect with contacts, check for implications. Float contacts file: /data/PDZ3_Ali/short_ligand/reduction/trans/contacts_analysis/cluster1-7/data/dist_all
# - internally change trajectory to 0-based, still support 1-based, ussue warning; Marcotraj as well


class MPT(object):
    def __init__(
        self,
        # traj: List[NDArray[np.int_]],
        traj: NDArray[np.int_],
        tlag: int,
        feature_traj: NDArray[float] = None,
        contact_threshold=0.45,
        feature_type=np.float64,
        macrostate_thresholds: tuple = (0.005, 0.5),
        limits=None,
        quiet=False,
        frame_length=0.2,
    ):
        self.traj = traj
        self.tlag = tlag
        self.pop_thr, self.q_min = macrostate_thresholds
        self.limits = limits
        tmat, states = mh.msm.estimate_markov_model(
            utils.get_multi_state_traj(self.traj, self.limits),
            self.tlag,
        )
        self.tmat = tmat.astype(np.float64)
        _, self.pop = np.unique(self.traj, return_counts=True)
        self.n_states = len(states)
        self.quiet = quiet
        if feature_traj is not None:
            self.add_feature(
                feature_traj,
                contact_threshold=contact_threshold,
                feature_type=feature_type,
            )
        else:
            self.add_feature(np.ones((traj.shape, 1)))

        self.Z = None
        self._timescales = None
        self._linkage = None
        self._macro_pop = None
        self._tree = None
        self._shannon_entropy = None
        self._davies_bouldin_index = None
        self._gmrq = None
        self._reference = None
        self._topology_file = None
        self._xtc_trajectory_file = None
        self._rmsd = None
        self.n_i = 0
        self.xtc_stride = None
        self.frame_length = frame_length
        self.micro_feature = None

    def mpt(
        self,
        kernel: Callable[
            [NDArray[float], NDArray[np.int_], NDArray[np.bool_]],
            [np.int_, np.int_, NDArray[np.bool_]],
        ] = kernel_module.MPTKernel(),
        feature_kernel=None,
        n: int = 1,
    ) -> (NDArray[float], NDArray[np.int_]):
        """Perform MPT"""
        self.n_runs = n
        self.kernel = kernel
        self.feature_kernel = feature_kernel
        # n: number of macrostates

        self.Z = np.zeros((self.n_runs, self.n_states - 1, 4), dtype=np.float64)
        self.full_pop = np.zeros((self.n_runs, 2 * self.n_states - 1), dtype=np.uint32)
        if self.quiet:
            iter = range(self.n_runs)
        else:
            print("Clustering ...")
            iter = tqdm(range(self.n_runs))
        for i in iter:
            self.Z[i], self.full_pop[i] = core.cluster(
                self.tmat,
                self.pop,
                kernel=self.kernel,
                feature_kernel=self.feature_kernel,
            )
        self.assign_macrostates()

    def add_feature(
        self,
        feature_traj: NDArray[float],
        contact_threshold=0.45,
        feature_type=np.float64,
    ):
        """
        Add feature data to instance

        feature_traj (NDArray(float)): frames x features
        """
        if feature_traj.shape[0] != self.traj.shape[0]:
            raise ValueError(
                "feature_traj must have the same length as the microstate trajectory (mpp.traj)"
            )
        if feature_traj.ndim == 2:
            self.multi_feature_traj = feature_traj.astype(feature_type)
        else:
            raise ValueError("feature_traj must be 2 D")

        self.contact_threshold = contact_threshold
        self.multi_feature_traj_bool = self.multi_feature_traj < self.contact_threshold
        self.feature_traj = self.multi_feature_traj_bool.mean(axis=1)
        self.feature = np.zeros(self.n_states, dtype=feature_type)
        for i in range(self.n_states):
            self.feature[i] = self.feature_traj[self.traj == i + 1].mean()

    def assign_macrostates(self, macrotraj_type=np.uint8):
        """Assign microstates to macrostates and collect associate data"""
        self.macrostate_feature = []
        self.macrostate_multi_feature = []
        self.macrostate_assignment = []
        self.macrostates_map = []
        self.macro_tmat = []
        self.macrotraj = np.zeros(
            (self.traj.shape[0], self.n_runs), dtype=macrotraj_type
        )
        self.n_macrostates = []

        if self.quiet:
            iter = range(self.n_runs)
        else:
            print("Assigning macrostates ...")
            iter = tqdm(range(self.n_runs))
        for n_i in iter:
            self.macrostate_assignment.append(
                utils.get_macrostate_assignment_from_tree(self.tree[n_i])
            )

            # Calculate other macrostate related values
            self.macrostates_map.append(
                np.zeros(self.n_states, dtype=self.traj.dtype.type)
            )
            mas, mis = np.where(self.macrostate_assignment[-1] == 1)
            self.macrostates_map[-1][mis] = mas
            self.macro_tmat.append(
                utils.macro_tmat(self.tmat, self.macrostate_assignment[-1], self.pop)
            )
            self.macrotraj[:, n_i] = utils.translate_traj(
                self.traj, self.macrostates_map[-1]
            )
            self.n_macrostates.append(self.macrostate_assignment[-1].shape[0])
            self.macrostate_feature.append(
                [
                    self.feature_traj[np.where(self.macrotraj[:, n_i] == i)].mean()
                    for i in np.arange(self.n_macrostates[-1]) + 1
                ]
            )
            self.macrostate_multi_feature.append(
                [
                    self.multi_feature_traj_bool[
                        np.where(self.macrotraj[:, n_i] == i)
                    ].mean(axis=0)
                    for i in np.arange(self.n_macrostates[-1], dtype=int) + 1
                ]
            )

    def macro_to_micro_feature(self):
        """Assign macrostate feature values to corresponding microstates"""
        self.micro_feature = np.zeros(
            (self.n_states, self.n_runs), dtype=self.feature_traj.dtype.type
        )
        for i, (ma, mf) in enumerate(
            zip(self.macrostate_assignment, self.macrostate_feature)
        ):
            for j, mb in enumerate(ma.astype(bool)):
                self.micro_feature[mb, i] = mf[j]

    def gpcca(self, n_macrostates, macrotraj_type=np.uint8):
        self.gpcca = gp.GPCCA(self.tmat, method="krylov")
        self.gpcca.optimize(n_macrostates)

        self.n_runs = 1
        self.n_macrostates = [n_macrostates]

        gma = self.gpcca.macrostate_assignment
        gmt = np.empty(self.traj.shape, dtype=self.traj.dtype)
        gmf = np.empty(self.n_macrostates[0])
        for i in range(self.n_macrostates[0]):
            gmt[np.where(np.isin(self.traj, np.where(gma == i)[0] + 1))[0]] = i + 1
            gmf[i] = self.feature_traj[gmt == i + 1].mean()

        order = np.argsort(gmf)[::-1]
        new_states = np.empty(self.n_macrostates[0], dtype=macrotraj_type)
        new_states[order] = np.arange(self.n_macrostates[0], dtype=macrotraj_type)
        self.macrostates_map = [np.empty(gma.shape, dtype=macrotraj_type)]
        for i in range(self.n_macrostates[0]):
            self.macrostates_map[0][np.where(gma == i)] = new_states[i]

        self.macrostate_assignment = [
            np.full((self.n_macrostates[0], self.macrostates_map[0].shape[0]), False)
        ]
        self.macrostate_assignment[0][
            self.macrostates_map[0],
            np.arange(self.macrostates_map[0].shape[0], dtype=int),
        ] = True
        self.macrostate_feature = [gmf[order]]
        self.macrotraj = np.empty(
            (self.traj.shape[0], self.n_runs), dtype=macrotraj_type
        )
        self.macrotraj[:, 0] = utils.translate_traj(self.traj, self.macrostates_map[0])
        self.macro_tmat = [
            utils.macro_tmat(self.tmat, self.macrostate_assignment[0], self.pop)
        ]

        # Create mock Z and mock full_pop for Sankey plot
        # After implementation remove mock Z.npy file in run.py
        self.Z = np.zeros((self.n_runs, self.n_states - 1, 4), dtype=np.float64)
        self.full_pop = np.zeros((self.n_runs, 2 * self.n_states - 1), dtype=np.uint32)
        self.full_pop[0, : self.n_states] = self.pop

        last_merged = self.n_states
        merge = 0
        for macrostate in range(self.n_macrostates[0]):
            microstates = np.where(self.macrostates_map[0] == macrostate)[0]
            origin = microstates[0]
            if microstates.shape[0] > 1:
                for target in microstates[1:]:
                    intermediate_state = self.n_states + merge
                    self.full_pop[0, intermediate_state] = self.full_pop[
                        0, [origin, target]
                    ].sum()
                    self.Z[0, merge] = (
                        origin,
                        target,
                        0.2,
                        self.full_pop[0, intermediate_state],
                    )
                    origin = intermediate_state
                    merge += 1

            if macrostate > 0:
                intermediate_state = self.n_states + merge
                target = last_merged
                self.full_pop[0, intermediate_state] = self.full_pop[
                    0, [origin, target]
                ].sum()
                self.Z[0, merge] = (
                    origin,
                    target,
                    0.9,
                    self.full_pop[0, intermediate_state],
                )
                last_merged = intermediate_state
                merge += 1
            else:
                last_merged = origin

        self.tree
        self.pop_thr = 0
        self.q_min = 0.5

    def set_n_i(self):
        """Sets self.n_i to the lumping with longest first implied timescale."""
        if self.n_runs > 1:
            self.n_i = np.argmax(self.timescales[:, 0])

    def __add__(self, other):
        """'+' operator is used to calculate similarity"""
        if self.n_runs == 1 and other.n_runs >= 1:
            # reference
            ref = self
            # stochastic clustering
            sto = other
        elif other.n_runs == 1 and self.n_runs >= 1:
            ref = other
            sto = self
        else:
            raise ValueError("The reference clustering must have exactly one run.")
        return ref, sto, utils.similarity(ref, sto)

    @property
    def timescales(self):
        """The timescales property."""
        if self._timescales is None:
            self.calc_timescales()
        return self._timescales

    def calc_timescales(self, ntimescales=3, dtype=np.float32):
        """Calculate implied timescales"""
        self._timescales = np.zeros((self.n_runs, ntimescales), dtype=dtype)
        for i, traj in enumerate(self.macrotraj.T):
            self._timescales[i, :] = mh.msm.implied_timescales(
                utils.get_multi_state_traj(traj, self.limits),
                [self.tlag],
                ntimescales=ntimescales,
            )[0]

    def save_macrotraj(self, out):
        header = (
            f"# Created by MPT class\n"
            f"# Time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
            f"# Trajectory contains {self.n_macrostates[self.n_i]} states and {self.macrotraj.shape[0]} frames.\n"
            f"# Trajectory index: {self.n_i}\n"
        )
        np.savetxt(out, self.macrotraj[:, self.n_i], fmt="%.0f", header=header)

    def save_Z(self, out, n_i="all"):
        """Save Z matrix"""
        if not out.endswith(".npy"):
            out += ".npy"

        if n_i == "all":
            np.save(out, self.Z)
        elif isinstance(n_i, Iterable):
            np.save(out, self.Z[n_i])
        elif isinstance(n_i, int):
            np.save(out, self.Z[n_i : n_i + 1])
        else:
            raise ValueError("n_i must be 'all', Iterable or int.")

    def from_Z(self, Z):
        """Load Z matrix"""
        if isinstance(Z, np.ndarray):
            self.Z = Z
        elif os.path.exists(Z):
            self.Z = np.load(Z)
        else:
            raise ValueError("Z must be a numpy array or a .npy file.")

        self.n_runs = self.Z.shape[0]
        # n: number of macrostates
        tmat, states = mh.msm.estimate_markov_model(
            utils.get_multi_state_traj(self.traj, self.limits),
            self.tlag,
        )
        self.tmat = tmat.astype(float)
        _, self.pop = np.unique(self.traj, return_counts=True)
        self.n_states = len(states)
        self.full_pop = np.zeros((self.n_runs, 2 * self.n_states - 1), dtype=np.uint32)
        self.full_pop[:, : self.n_states] = self.pop
        self.full_pop[:, self.n_states :] = self.Z[:, :, 3]

        self.assign_macrostates()

    @property
    def linkage(self):
        """The linkage property."""
        if self._linkage is None:
            self._linkage = utils.Z_to_linkage(self.Z[self.n_i])
        return self._linkage

    @property
    def macro_pop(self):
        """The macro_pop property."""
        if self._macro_pop is None:
            self._macro_pop = []
            for j, ma in enumerate(self.macrostate_assignment):
                self._macro_pop.append(
                    np.zeros(ma.shape[0], dtype=self.full_pop.dtype.type)
                )
                for i, m in enumerate(ma):
                    self._macro_pop[-1][i] = self.full_pop[j, : self.n_states][
                        m.astype(bool)
                    ].sum()
        return self._macro_pop

    @property
    def tree(self):
        """The tree property."""
        if self._tree is None:
            self._tree = []
            for z, pop in zip(self.Z, self.full_pop):
                self._tree.append(self.build_tree(z, pop))
        return self._tree

    def build_tree(self, Z, full_pop):
        """Build tree using BinaryTreeNode and return root"""
        macrostate_thresholds = (self.pop_thr, self.q_min)
        n = Z.shape[0] + 1
        nodes = {}
        for i, (state, target_state, q, pop) in enumerate(Z):
            state = int(state)
            target_state = int(target_state)
            if state not in nodes:
                nodes[state] = core.BinaryTreeNode(
                    state,
                    self.tmat,
                    population=full_pop[state],
                    q=q,
                    macrostate_thresholds=macrostate_thresholds,
                )
            if target_state not in nodes:
                nodes[target_state] = core.BinaryTreeNode(
                    target_state,
                    self.tmat,
                    population=full_pop[target_state],
                    q=q,
                    macrostate_thresholds=macrostate_thresholds,
                )
            nodes[n + i] = core.BinaryTreeNode(
                n + i, self.tmat, q=q, macrostate_thresholds=macrostate_thresholds
            )
            nodes[n + i].left = nodes[state]
            nodes[n + i].right = nodes[target_state]
        for node in nodes[n + i].leaves:
            node.feature = self.feature[node.name]
        return nodes[n + i]

    @property
    def shannon_entropy(self):
        """The shannon_entropy property."""
        if self._shannon_entropy is None:
            self._shannon_entropy = np.zeros(self.n_runs)
            for i, pop in enumerate(self.macro_pop):
                self._shannon_entropy[i] = utils.shannon_entropy(pop)
        return self._shannon_entropy

    @property
    def davies_bouldin_index(self):
        """The davies_bouldin_index property."""
        if self._davies_bouldin_index is None:
            self._davies_bouldin_index = np.zeros(self.n_runs)
            for i in range(self.n_runs):
                self._davies_bouldin_index[i] = davies_bouldin_score(
                    self.multi_feature_traj, self.macrotraj[:, i]
                )
        return self._davies_bouldin_index

    @property
    def gmrq(self):
        """The gmrq property."""
        if self._gmrq is None:
            self._gmrq = utils.gmrq(self.macro_tmat)
        return self._gmrq

    @property
    def reference(self):
        """The reference property."""
        if self._reference is None:
            k = kernel_module.MPTKernel()
            self._reference = MPT(
                self.traj,
                self.tlag,
                self.multi_feature_traj,
                contact_threshold=self.contact_threshold,
                macrostate_thresholds=(self.pop_thr, self.q_min),
                limits=self.limits,
                quiet=True,
            )
            self._reference.mpt(k)
        return self._reference

    @property
    def traj(self):
        """The traj property."""
        return self._traj

    @traj.setter
    def traj(self, value):
        if value.max() < 2**7:
            traj_type = np.uint8
        elif value.max() < 2**15:
            traj_type = np.uint16
        else:
            traj_type = np.uint32

        if value.min() == 1:
            self._traj = value.astype(traj_type)
            # warnings.warn("1-based trajectory was shifted to 0-based.")
        elif value.min() == 0:
            self._traj = value.astype(traj_type) + 1
            warnings.warn(
                "Still 1-based trajectory used, thus, trajectory was shifted to 1-based."
            )
        else:
            raise ValueError("trajectory must be 0 or 1 based")

    def print_rel(self):
        for l, i in [
            (
                "Implied Timescale: ",
                self.timescales[0, 0] / self.reference.timescales[0, 0],
            ),
            ("GMRQ: ", self.gmrq[0] / self.reference.gmrq[0]),
            (
                "DBI: ",
                self.davies_bouldin_index()[0]
                / self.reference.davies_bouldin_index()[0],
            ),
            ("H: ", self.shannon_entropy[0] / self.reference.shannon_entropy[0]),
        ]:
            print(l + f"{i:.2f}")

    @property
    def topology_file(self):
        """The topology_file property."""
        if self._topology_file is None:
            raise ValueError("No topology file set.")
        return self._topology_file

    @topology_file.setter
    def topology_file(self, value):
        if os.path.isfile(value):
            self._topology_file = value
        else:
            raise FileNotFoundError(f"No such file: {value}")

    @property
    def xtc_trajectory_file(self):
        """The xtc_trajectory_file property."""
        if self._xtc_trajectory_file is None:
            raise ValueError("No xtc trajectory file set.")
        return self._xtc_trajectory_file

    @xtc_trajectory_file.setter
    def xtc_trajectory_file(self, value):
        if os.path.isfile(value):
            self._xtc_trajectory_file = value
        else:
            raise FileNotFoundError(f"No such file: {value}")

    @property
    def rmsd(self):
        """The rmsd property."""
        if self._rmsd is None:
            self._rmsd, self.mean_frames = utils.calc_rmsd(self)
        return self._rmsd

    @property
    def frame_length(self):
        """The frame_length property. Frame length in ns."""
        return self._frame_length

    @frame_length.setter
    def frame_length(self, value):
        self._frame_length = value

    def save_rmsd(self, out):
        np.save(out, self.rmsd)

    def load_rmsd(self, f_name):
        self._rmsd = np.load(f_name)

    def write_pdbs(self, out):
        utils.write_pdbs(
            out,
            np.log(self.rmsd),
            self.topology_file,
            self.xtc_trajectory_file,
            self.mean_frames,
        )

    def rmsd_sharpness(self):
        return (
            self.rmsd.mean(axis=1) * self.macro_pop[self.n_i]
        ).sum() / self.macro_pop[self.n_i].sum()

    def draw_random_frames_indices(self, out=None, n=20):
        """
        Draw n random frames for each macrostate

        out (str): Path to directory where to save the .random[n] files
        n (int): number of frames to draw randomly
        """
        drawn_frames = np.empty((self.n_macrostates[self.n_i], n), dtype=int)
        for state in np.arange(self.n_macrostates[self.n_i]):
            frames_in_state = np.where(self.macrotraj[:, self.n_i] == state + 1)[0]
            drawn_frames[state] = np.random.choice(
                frames_in_state, size=n, replace=False
            )
        if self.xtc_stride is not None:
            drawn_frames *= self.xtc_stride
        if out:
            Path(os.path.join(out)).mkdir(parents=True, exist_ok=True)
            for s, i in enumerate(drawn_frames):
                # Path(os.path.join(out, f"{s+1:02d}")).mkdir(parents=True, exist_ok=True)
                # np.savetxt(os.path.join(out, f"{s+1:02d}", f".frames.ndx"), i, fmt="%.0f", header="[frames]")
                np.savetxt(
                    os.path.join(out, f"{s + 1:02d}.ndx"),
                    i,
                    fmt="%.0f",
                    header="[frames]",
                )
        else:
            return drawn_frames

    def draw_random_frames(self, out, n=20):
        """
        Draw n random frames for each macrostate

        out (str): Path to directory where to save the pdb files
        n (int): number of frames to draw randomly
        """
        for state in np.arange(self.n_macrostates[self.n_i]) + 1:
            frames_in_state = np.where(self.macrotraj[:, self.n_i] == state)[0]
            drawn_frames = np.random.choice(frames_in_state, size=n, replace=False)
            for i, frame in enumerate(drawn_frames):
                f = md.load_xtc(
                    self.xtc_trajectory_file,
                    top=self.topology_file,
                    frame=frame,
                )
                f.save_pdb(os.path.join(out, f"S{state}_{i:02d}.pdb"))

    def get_best_defined_contacts(self, n=3):
        """Calculate the variance for each contact in each macrostate."""
        contacts = np.zeros((self.n_macrostates[self.n_i], n), dtype=int)
        for i in range(self.n_macrostates[self.n_i]):
            contacts[i] = np.argsort(
                np.var(
                    self.multi_feature_traj[self.macrotraj[:, self.n_i] == i + 1],
                    axis=0,
                )
            )[:n]
        return contacts

    def get_least_moving_residues(self, contact_index_file, n=3):
        contact_indices = np.loadtxt(contact_index_file, dtype=int)
        contacts = self.get_best_defined_contacts(n)
        least_moving_residues = []
        for c in contacts:
            least_moving_residues.append(np.unique(contact_indices[c].flatten()))
        return least_moving_residues

    def write_least_moving_residues(self, contact_index_file, out, n=3):
        if contact_index_file != "none":
            least_moving_residues = self.get_least_moving_residues(
                contact_index_file, n=n
            )
            with open(out, "w") as f:
                for residues in least_moving_residues:
                    f.write(f"{' '.join(residues.astype(str))}\n")
        else:
            with open(out, "w") as f:
                f.write("")

    ### PLOT METHODS #########################################################

    def plot(self, out: str, scale=1, offset=0):
        """Plot dendrogram"""
        plot.plot_tree(
            self.tree[self.n_i],
            self.macrostate_assignment[self.n_i],
            out,
            scale=scale,
            offset=offset,
        )

    def plot_implied_timescales(self, out, use_ref=True, scale=1):
        """
        out: File to write plot
        use_ref: If it for reference trajectory should be plotted
        scale: scaling factor for plot
        """
        if use_ref:
            ref_traj = self.reference.macrotraj[:, 0]
        else:
            ref_traj = self.traj

        macrotraj = utils.get_multi_state_traj(self.macrotraj[:, self.n_i], self.limits)

        dtlag = max(1, int(1 / self.frame_length))
        plot.plot_implied_timescales(
            [ref_traj, macrotraj],
            # [self.traj, self.macrotraj[:, self.n_i]],
            # np.arange(1, 227, 5),
            np.arange(1, 4.5 * self.tlag + dtlag, dtlag, dtype=int),
            out,
            frame_length=self.frame_length,
            first_ref=True,
            scale=scale,
            use_ref=use_ref,
            ntimescales=self.timescales.shape[1],
        )

    def plot_macro_feature(self, out, ref=None):
        """
        Plot histogram of feature distribution.

        micro_feature (np.ndarray, NxR): N microstates, R runs, holds feature
                values of respective macrostate
        out (str): file to save the plot
        ref (list[tuple]): list of
                - macrostate_assignment
                - macrostate_feature
                - color
                - label
                of the clusterings that should be shown explicitly.
        """
        if self.micro_feature is None:
            self.macro_to_micro_feature()
        plot.plot_macro_feature(
            self.micro_feature, out, self.reference if ref is None else ref
        )

    def plot_rmsd(self, out, helices=None):
        plot.plot_rmsd(self.rmsd, self.macro_pop[self.n_i], helices, out)

    def plot_delta_rmsd(self, out, helices=None):
        plot.plot_delta_rmsd(self.rmsd, self.macro_pop[self.n_i], helices, out)

    def plot_contact_rep(self, cluster_file, out, scale=1):
        plot.contact_rep(
            self.multi_feature_traj,
            cluster_file,
            self.macrotraj[:, self.n_i],
            out,
            utils.get_grid_format(self.n_macrostates[self.n_i]),
            scale=scale,
        )

    def plot_relative_implied_timescales(self, out):
        plot.relative_implied_timescales(self, out)

    def plot_ck_test(self, out):
        plot.chapman_kolmogorov(self, out, self.frame_length)

    def plot_state_network(self, out):
        plot.state_network(self, out)

    def plot_stochastic_state_similarity(self, out):
        plot.evaluate_stochastic_clustering(self, self.reference, out)

    def plot_transition_matrix(self, out):
        plot.transition_matrix(self.macro_tmat[self.n_i], out)

    def plot_transition_time(self, out):
        plot.transition_time(
            self.macro_tmat[self.n_i],
            out,
            tlag=self.tlag,
            frame_length=self.frame_length,
        )

    def plot_graph(self, out, u=0, f=0):
        draw_knetwork(
            self.macrotraj[:, self.n_i], self.tlag, self.feature_traj, out, u=u, f=f
        )

    def plot_sankey(self, out, ax=None, scale=1):
        plot.plot_sankey(self, self.reference, out, ax=ax, scale=scale)

    def plot_macrotraj(self, out, row_length=0.2):
        plot.plot_state_trajectory(
            self.macrotraj[:, self.n_i],
            out,
            row_length=row_length,
            frame_length=self.frame_length,
        )
