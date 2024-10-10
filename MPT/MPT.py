import os
import datetime
import numpy as np
import msmhelper as mh
import matplotlib.pyplot as plt

from tqdm import tqdm
from typing import Callable
from numpy.typing import NDArray
from collections.abc import Iterable

import MPT.core as core
import MPT.utils as utils
import MPT.kernel as kernel_module
from graph import draw_knetwork

import MPT.plot as plot

__doc__ = """
MPT - Most Probable Transition algorithm
========================================

**MPT** is a set of tools used to analyze trajectories of molecular dynamics
(MD) simulations. Trajectories are a huge collection of cartesian coordinates
that need to be boiled down to collective variables and frames need to be
assigned a state in order to extract desired information from such a
trajectory.

MPT is based on markov state models from a microstate trajectory.

"""

__all__ = [
    "kernel.MPTKernel",
    "kernel.SMPTKernel",
    "kernel.feature_kernel",
    "MPT",
]

class MPT(object):
    def __init__(self, traj: NDArray[np.int_], tlag: int, macrostate_thresholds: tuple = (0.005, 0.5)):
        if traj.max() < 2**8:
            traj_type = np.uint8
        elif traj.max() < 2**16:
            traj_type = np.uint16
        else:
            traj_type = np.uint32

        self.traj = traj.astype(traj_type)
        self.tlag = tlag
        self.macrostate_thresholds = macrostate_thresholds

        self.timescales = None
        self._linkage = None
        self._macro_pop = None
        self._tree = None

    # TODO:
    # doublecheck annotation
    def mpt(
        self,
        kernel: Callable[
            [NDArray[np.float_], NDArray[np.int_], NDArray[np.bool_]],
            [np.int_, np.int_, NDArray[np.bool_]]
        ]=kernel_module.MPTKernel(),
        feature_kernel = 1,
        n: int = 1,
    ) -> (NDArray[np.float_], NDArray[np.int_]):
        """
        mpt
        ---
        Perform MPT
     
        kernel: kernel object
        n (int): number of runs
     
        defines: self.Z matrix as of cluster,
            self.full_pop population of all states
        """
        self.n_runs = n
        self.kernel = kernel
        self.feature_kernel = feature_kernel
        # n: number of macrostates
        tmat, states = mh.msm.estimate_markov_model(self.traj, self.tlag)
        self.tmat = tmat.astype(np.float64)
        _, pop = np.unique(self.traj, return_counts=True)
        self.n_states = len(states)
        self.Z = np.zeros((self.n_runs, self.n_states-1, 4), dtype=np.float64)
        self.full_pop = np.zeros((self.n_runs, 2*self.n_states-1), dtype=np.uint32)
        print("Clustering ...")
        for i in tqdm(range(self.n_runs)):
            self.Z[i], self.full_pop[i] = core.cluster(
                self.tmat,
                pop,
                kernel=self.kernel,
                feature_kernel=self.feature_kernel
            )

    def add_feature(self, feature_traj: NDArray[np.float_], feature_type=np.float64):
        self.feature_traj = feature_traj.astype(feature_type)
        self.feature = np.zeros(self.n_states, dtype=feature_type)
        for i in range(self.n_states):
            self.feature[i] = self.feature_traj[self.traj == i+1].mean()

    def assign_macrostates(self, pop_thr, q_min, macrotraj_type=np.uint8):
        self.pop_thr = pop_thr
        self.q_min = q_min
        self.macrostate_feature = []
        self.macrostate_assignment = []
        self.macrostates_map = []
        self.macro_tmat = []
        self.macrotraj = np.zeros((self.traj.shape[0], self.n_runs), dtype=macrotraj_type)
        self.n_macrostates = []
        pop = self.full_pop[0, :self.n_states]

        print("Assigning macrostates ...")
        for n_i in tqdm(range(self.n_runs)):
            self.macrostate_assignment.append(utils.get_macrostate_assignment_from_tree(self.tree[n_i]))

            # Calculate other macrostate related values
            self.macrostates_map.append(np.zeros(self.n_states, dtype=self.traj.dtype.type))
            mas, mis = np.where(self.macrostate_assignment[-1]==1)
            self.macrostates_map[-1][mis] = mas
            self.macro_tmat.append(utils.macro_tmat(self.tmat, self.macrostate_assignment[-1], pop))
            self.macrotraj[:, n_i] = utils.translate_traj(self.traj, self.macrostates_map[-1])
            self.n_macrostates.append(self.macrostate_assignment[-1].shape[0])


    def macro_to_micro_feature(self):
        self.micro_feature = np.zeros((self.n_states, self.n_runs), dtype=self.feature_traj.dtype.type)
        for i, (ma, mf) in enumerate(zip(self.macrostate_assignment, self.macrostate_feature)):
            for j, mb in enumerate(ma.astype(bool)):
                self.micro_feature[mb, i] = mf[j]

    def plot(self, out: str, n_i: int = 0):
        plot.plot_tree(self.tree[n_i], self.macrostate_assignment[n_i], out)
    
    def __add__(self, other):
        """
        The '+' operator is used to calculate the similarity between many
        stochastic clusterings and a reference
        """
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

        # Similarity matrix
        S = np.zeros((3, ref.n_macrostates[0], sto.n_runs))
        
        print(f"Calculating similarities for {ref.n_macrostates[0]} macrostates ...")
        for n_i in tqdm(range(sto.n_runs)):
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
        return ref, sto, S

    def __mul__(self, other):
        """
        The '*' operator is used to calculate the similarity between many
        stochastic clusterings and a reference - needs to be the right hand
        argument

        Currently, the number of microstates in common is calculated.
        """
        # Similarity matrix
        S = []
        n = []
        
        for n_i in tqdm(range(self.n_runs-1)):
            for n_j in range(n_i + 1, other.n_runs):
                if n_i == n_j:
                    continue
                # utils.similarity calculates the number of common microstate for each macrostate combination
                S.append(utils.similarity(self.macrostate_assignment[n_i], other.macrostate_assignment[n_j]))
                d1 = self.macrostate_assignment[n_i].shape[0]
                d2 = self.macrostate_assignment[n_j].shape[0]
                n.append(S[-1].mean())
        return S, n

    def calc_timescales(self, ntimescales=3, dtype=np.float32):
        self.timescales = np.zeros((self.n_runs, ntimescales), dtype=dtype)
        for i, traj in enumerate(self.macrotraj.T):
            self.timescales[i] = mh.msm.implied_timescales(traj, [self.tlag], ntimescales=ntimescales)[0]

    def plot_timescales(self, out, n_i=0):
        if self.timescales == None:
            self.calc_timescales()
        plt.hist(self.timescales[n_i][:, 0])
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

    def save_macrotraj(self, out, n_i=0):
        header = (
            f"# Created by MPT class\n"
            f"# Time: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M")}\n"
            f"# Trajectory contains {self.n_macrostates[n_i]} states and {self.macrotraj.shape[0]} frames.\n"
            f"# Trajectory index: {n_i}\n"
        )
        np.savetxt(out, self.macrotraj[:, n_i], fmt="%.0f", header=header)

    def save_Z(self, out, n_i="all"):
        if out.endswith(".Z.npy"):
            pass
        elif out.endswith(".Z"):
            out += ".npy"
        elif out.endswith(".npy"):
            out = out[:-4] + ".Z.npy"
        else:
            out += ".Z.npy"

        if os.path.exists(out):
            if input("File exists. Overwrite? [y|n] ") == "y":
                os.remove(out)
            else:
                print("Z matrix not saved.")
                return None

        if n_i == "all":
            np.save(out, self.Z)
        elif isinstance(n_i, Iterable):
            np.save(out, self.Z[n_i])
        elif isinstance(n_i, int):
            np.save(out, self.Z[n_i:n_i+1])
        else:
            raise ValueError("n_i must be 'all', Iterable or int.")

    def from_Z(self, Z):
        if isinstance(Z, np.ndarray):
            self.Z = Z
        elif os.path.exists(Z):
            self.Z = np.load(Z)
        else:
            raise ValueError("Z must be a numpy array or a .npy file.")
        
        self.n_runs = self.Z.shape[0]
        # n: number of macrostates
        tmat, states = mh.msm.estimate_markov_model(self.traj, self.tlag)
        self.tmat = tmat.astype(np.float_)
        _, pop = np.unique(self.traj, return_counts=True)
        self.n_states = len(states)
        self.full_pop = np.zeros((self.n_runs, 2*self.n_states-1), dtype=np.uint32)
        self.full_pop[:, :self.n_states] = pop
        self.full_pop[:, self.n_states:] = self.Z[:, :, 3]

    @property
    def linkage(self):
        """The linkage property."""
        if self._linkage == None:
            self._linkage = utils.Z_to_linkage(self.Z)
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
        """
        Build tree of BinaryTreeNode from a given Z matrix and the corresponding
        populations.
        """
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

    def plot_graph(self, out, n_i=0, u=0, f=0):
        draw_knetwork(self.macrotraj[:, n_i], self.tlag, self.feature_traj, out, u=u, f=f)

    def plot_tmat(self, out, n_i=0):
        plot.plot_tmat(self.macro_tmat[n_i].copy(), out, title="Macrostate Transitiom Matrix")

    def plot_tmat_times(self, out, n_i=0):
        plot.plot_trans_time(self.macro_tmat[n_i].copy(), out, title="Macrostate Transitiom Times")

