import os
import datetime
import warnings
import numpy as np
import msmhelper as mh
import matplotlib.pyplot as plt
import mdtraj as md

from tqdm import tqdm
from typing import Callable, List
from numpy.typing import NDArray
from collections.abc import Iterable
from sklearn.metrics import davies_bouldin_score

import MPT.core as core
import MPT.utils as utils
import MPT.kernel as kernel_module
from MPT.graph import draw_knetwork

import MPT.plot as plot

__all__ = [
    "kernel.MPTKernel",
    "kernel.feature_kernel",
    "MPT",
]

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
            feature_traj: NDArray[np.float_]=None,
            feature_type=np.float64,
            macrostate_thresholds: tuple = (0.005, 0.5),
            limits=None,
            quiet=False
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
            self.add_feature(feature_traj, feature_type)
        else:
            self.add_feature(np.ones(traj.shape), feature_type)

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

    def mpt(
        self,
        kernel: Callable[
            [NDArray[np.float_], NDArray[np.int_], NDArray[np.bool_]],
            [np.int_, np.int_, NDArray[np.bool_]]
        ]=kernel_module.MPTKernel(),
        feature_kernel = 1,
        n: int = 1,
    ) -> (NDArray[np.float_], NDArray[np.int_]):
        """Perform MPT"""
        self.n_runs = n
        self.kernel = kernel
        self.feature_kernel = feature_kernel
        # n: number of macrostates

        self.Z = np.zeros((self.n_runs, self.n_states-1, 4), dtype=np.float64)
        self.full_pop = np.zeros((self.n_runs, 2*self.n_states-1), dtype=np.uint32)
        if not self.quiet:
            print("Clustering ...")
            iter = tqdm(range(self.n_runs))
        else:
            iter = range(self.n_runs)
        for i in iter:
            self.Z[i], self.full_pop[i] = core.cluster(
                self.tmat,
                self.pop,
                kernel=self.kernel,
                feature_kernel=self.feature_kernel
            )
        self.assign_macrostates()

    def add_feature(self, feature_traj: NDArray[np.float_], feature_type=np.float64):
        """Add feature data to instance"""
        self.feature_traj = feature_traj.astype(feature_type)
        self.feature = np.zeros(self.n_states, dtype=feature_type)
        for i in range(self.n_states):
            self.feature[i] = self.feature_traj[self.traj == i+1].mean()

    def assign_macrostates(self, macrotraj_type=np.uint8):
        """Assign microstates to macrostates and collect associate data"""
        self.macrostate_feature = []
        self.macrostate_assignment = []
        self.macrostates_map = []
        self.macro_tmat = []
        self.macrotraj = np.zeros((self.traj.shape[0], self.n_runs), dtype=macrotraj_type)
        self.n_macrostates = []

        if not self.quiet:
            print("Assigning macrostates ...")
            iter = tqdm(range(self.n_runs))
        else:
            iter = range(self.n_runs)
        for n_i in iter:
            self.macrostate_assignment.append(utils.get_macrostate_assignment_from_tree(self.tree[n_i]))

            # Calculate other macrostate related values
            self.macrostates_map.append(np.zeros(self.n_states, dtype=self.traj.dtype.type))
            mas, mis = np.where(self.macrostate_assignment[-1]==1)
            self.macrostates_map[-1][mis] = mas
            self.macro_tmat.append(utils.macro_tmat(self.tmat, self.macrostate_assignment[-1], self.pop))
            self.macrotraj[:, n_i] = utils.translate_traj(self.traj, self.macrostates_map[-1])
            self.n_macrostates.append(self.macrostate_assignment[-1].shape[0])

    def macro_to_micro_feature(self):
        """Assign macrostate feature values to corresponding microstates"""
        self.micro_feature = np.zeros((self.n_states, self.n_runs), dtype=self.feature_traj.dtype.type)
        for i, (ma, mf) in enumerate(zip(self.macrostate_assignment, self.macrostate_feature)):
            for j, mb in enumerate(ma.astype(bool)):
                self.micro_feature[mb, i] = mf[j]

    def plot(self, out: str, scale=1):
        """Plot dendrogram"""
        plot.plot_tree(self.tree[self.n_i], self.macrostate_assignment[self.n_i], out, scale=scale)
    
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
            raise ValueError(
                "The reference clustering must have exactly one run."
            )
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
                ntimescales=ntimescales
            )[0]

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

        plot.plot_implied_timescales(
            [ref_traj, macrotraj],
            # [self.traj, self.macrotraj[:, self.n_i]],
            np.arange(1, 227, 5),
            out,
            first_ref=True,
            scale=scale,
        )

    def plot_timescales(self, out):
        """Plot implied timescales as histogram and save to out"""
        plt.hist(self.timescales[self.n_i][:, 0])
        plt.tight_layout()
        plt.savefig(out)

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
        plot.plot_macro_feature(self.micro_feature, out, ref)

    def save_macrotraj(self, out):
        header = (
            f"# Created by MPT class\n"
            f"# Time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
            f"# Trajectory contains {self.n_macrostates[n_i]} states and {self.macrotraj.shape[0]} frames.\n"
            f"# Trajectory index: {self.n_i}\n"
        )
        np.savetxt(out, self.macrotraj[:, self.n_i], fmt="%.0f", header=header)

    def save_Z(self, out, n_i="all"):
        """Save Z matrix"""
        if not out.endswith(".npy"):
            out += ".npy"
        # if out.endswith(".Z.npy"):
        #     pass
        # elif out.endswith(".Z"):
        #     out += ".npy"
        # elif out.endswith(".npy"):
        #     out = out[:-4] + ".Z.npy"
        # else:
        #     out += ".Z.npy"

        # if os.path.exists(out):
        #     if input("File exists. Overwrite? [y|n] ") == "y":
        #         os.remove(out)
        #     else:
        #         print("Z matrix not saved.")
        #         return None

        if n_i == "all":
            np.save(out, self.Z)
        elif isinstance(n_i, Iterable):
            np.save(out, self.Z[n_i])
        elif isinstance(n_i, int):
            np.save(out, self.Z[n_i:n_i+1])
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
        self.tmat = tmat.astype(np.float_)
        _, self.pop = np.unique(self.traj, return_counts=True)
        self.n_states = len(states)
        self.full_pop = np.zeros((self.n_runs, 2*self.n_states-1), dtype=np.uint32)
        self.full_pop[:, :self.n_states] = self.pop
        self.full_pop[:, self.n_states:] = self.Z[:, :, 3]

        self.assign_macrostates()

    @property
    def linkage(self):
        """The linkage property."""
        if self._linkage == None:
            self._linkage = utils.Z_to_linkage(self.Z[self.n_i])
        return self._linkage

    @property
    def macro_pop(self):
        """The macro_pop property."""
        if self._macro_pop == None:
            self._macro_pop = []
            for j, ma in enumerate(self.macrostate_assignment):
                self._macro_pop.append(np.zeros(ma.shape[0], dtype=self.full_pop.dtype.type))
                for i, m in enumerate(ma):
                    self._macro_pop[-1][i] = self.full_pop[j, :self.n_states][m.astype(bool)].sum()
        return self._macro_pop

    @property
    def tree(self):
        """The tree property."""
        if self._tree == None:
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
                nodes[state] = core.BinaryTreeNode(state, self.tmat, population=full_pop[state], q=q, macrostate_thresholds=macrostate_thresholds)
            if target_state not in nodes:
                nodes[target_state] = core.BinaryTreeNode(target_state, self.tmat, population=full_pop[target_state], q=q, macrostate_thresholds=macrostate_thresholds)
            nodes[n + i] = core.BinaryTreeNode(n + i, self.tmat, q=q, macrostate_thresholds=macrostate_thresholds)
            nodes[n + i].left = nodes[state]
            nodes[n + i].right = nodes[target_state]
        for node in nodes[n + i].leaves:
            node.feature = self.feature[node.name]
        return nodes[n + i]

    def plot_graph(self, out, u=0, f=0):
        draw_knetwork(self.macrotraj[:, self.n_i], self.tlag, self.feature_traj, out, u=u, f=f)

    def plot_tmat(self, out):
        plot.plot_tmat(self.macro_tmat[self.n_i].copy(), out, title="Macrostate Transitiom Matrix")

    def plot_tmat_times(self, out):
        plot.plot_trans_time(self.macro_tmat[self.n_i].copy(), out, title="Macrostate Transitiom Times")

    def plot_sankey(self, out, ax=None, scale=1):
        plot.plot_sankey(self, self.reference, out, ax=ax, scale=scale)

    def plot_macrotraj(self, out, row_length=0.2):
        plot.plot_state_trajectory(self.macrotraj[:, self.n_i], out, row_length=row_length)

    @property
    def shannon_entropy(self):
        """The shannon_entropy property."""
        if self._shannon_entropy is None:
            self._shannon_entropy = np.zeros(self.n_runs)
            for i, pop in enumerate(self.macro_pop):
                self._shannon_entropy[i] = utils.shannon_entropy(pop)
        return self._shannon_entropy

    def davies_bouldin_index(self, multi_feature_traj):
        """The davies_bouldin_index property."""
        if self._davies_bouldin_index is None:
            self._davies_bouldin_index = np.zeros(self.n_runs)
            for i in range(self.n_runs):
                self._davies_bouldin_index[i] = davies_bouldin_score(multi_feature_traj, self.macrotraj[:, i])
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
                self.feature_traj,
                macrostate_thresholds=(self.pop_thr, self.q_min),
                limits=self.limits,
                quiet=True
            )
            self._reference.mpt(k)
        return self._reference

    @property
    def traj(self):
        """The traj property."""
        return self._traj
    @traj.setter
    def traj(self, value):
        if value.max() < 2**8:
            traj_type = np.uint8
        elif value.max() < 2**16:
            traj_type = np.uint16
        else:
            traj_type = np.uint32

        if value.min() == 1:
            self._traj = value.astype(traj_type)
            self._traj_base = 1
            # warnings.warn("1-based trajectory was shifted to 0-based.")
        elif value.min() == 0:
            self._traj = value.astype(traj_type) + 1
            self._traj_base = 0
            warnings.warn("Still 1-based trajectory used, thus, trajectory was shifted to 1-based.")
        else:
            raise ValueError("trajectory must be 0 or 1 based")

    def print_rel(self, multi_feature_traj):
        for l, i in [
            ("Implied Timescale: ", self.timescales[0, 0] / self.reference.timescales[0, 0]),
            ("GMRQ: ", self.gmrq[0] / self.reference.gmrq[0]),
            ("DBI: ", self.davies_bouldin_index(multi_feature_traj)[0] / self.reference.davies_bouldin_index(multi_feature_traj)[0]),
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

    def save_rmsd(self, out):
        np.save(out, self._rmsd)

    def load_rmsd(self, f_name):
        self._rmsd = np.load(f_name)

    def write_pdbs(self, out):
        utils.write_pdbs(
            out,
            np.log(self.rmsd),
            self.topology_file,
            self.xtc_trajectory_file,
            self.mean_frames
        )

    def plot_rmsd(self, out, helices=None):
        plot.plot_rmsd(self.rmsd, self.macro_pop[self.n_i], helices, out)

    def plot_contact_rep(self, multi_feature_traj, cluster_file, out, scale=1):
        plot.contact_rep(
            multi_feature_traj,
            cluster_file,
            self.macrotraj[:, self.n_i],
            out,
            utils.get_grid_format(self.n_macrostates[self.n_i]),
            scale=scale,
        )

    def plot_relative_implied_timescales(self, out):
        plot.plot_relative_implied_timescales(self, out)

    def plot_ck_test(self, out, frame_length=0.2):
        """"frame_length in ns"""
        plot.chapman_kolmogorov(self, out, frame_length)

    def draw_random_frames(self, out, n=20):
        """
        Draw n random frames for each macrostate

        out (str): Path to directory where to save the pdb files
        n (int): number of frames to draw randomly
        """
        for state in np.arange(self.n_macrostates[self.n_i]) + 1:
            frames_in_state = np.where(self.macrotraj[:, self.n_i]==state)[0]
            drawn_frames = np.random.choice(frames_in_state, size=n, replace=False)
            for i, frame in enumerate(drawn_frames):
                f = md.load_xtc(self.xtc_trajectory_file, top=self.topology_file, frame=frame)
                f.save_pdb(os.path.join(out, f"S{state}_{i:02d}.pdb"))

