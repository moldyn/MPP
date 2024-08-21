import numpy as np
import msmhelper as mh
import matplotlib.pyplot as plt
import prettypyplot as pplt
import matplotlib.ticker as mtick
from pathos.multiprocessing import ProcessingPool as Pool

from tqdm import tqdm
from typing import Callable
from numpy.typing import NDArray

import core
import utils
import kernel

from plot import plot_dendrogram, plot_macro_feature

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
    def __init__(self, traj: NDArray[np.int_], tlag: int):
        self.traj = traj
        self.tlag = tlag

        self.time_scales = None

    def mpt(
        self,
        kernel: Callable[
            [NDArray[np.float_], NDArray[np.int_], NDArray[np.bool_]],
            [np.int_, np.int_, NDArray[np.bool_]]
        ]=kernel.MPTKernel(),
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
        # n: number of macrostates
        self.tmat, states = mh.msm.estimate_markov_model(self.traj, self.tlag)
        _, pop = np.unique(self.traj, return_counts=True)
        self.n_states = len(states)
        self.Z = np.zeros((self.n_runs, self.n_states-1, 4))
        self.full_pop = np.zeros((n, 2*self.n_states-1))
        for i in tqdm(range(n)):
            self.Z[i], self.full_pop[i] = core.cluster(
                self.tmat,
                pop,
                kernel=kernel,
                feature_kernel=feature_kernel
            )
        # def helper(i):
        #     return core.cluster(
        #         self.tmat,
        #         pop,
        #         kernel=kernel,
        #         feature_kernel=feature_kernel
        #     )
        #
        # with Pool() as pool:
        #     results = list(pool.map(helper, range(self.n_runs)))
        #
        # for i, result in enumerate(results):
        #     self.Z[i], self.full_pop[i] = result

    def add_feature(self, feature_traj: NDArray[np.float_]):
        self.feature_traj = feature_traj
        self.feature = np.zeros(self.n_states)
        for i in range(self.n_states):
            self.feature[i] = self.feature_traj[self.traj == i+1].mean()

    def assign_macrostates(self, pop_thr, q_min):
        self.macrostate_feature = []
        self.macrostate_assignment = []
        self.macrostates_map = []
        self.macro_tmat = []
        self.macrotraj = np.zeros((self.traj.shape[0], self.n_runs))
        self.n_macrostates = []

        print("Assigning macrostates ...")
        for n_i in tqdm(range(self.n_runs)):
            ma = core.assign_macrostates(self.Z[n_i], self.full_pop[n_i], pop_thr, q_min)
            macrostate_feature = np.zeros(ma.shape[0])
            pop = self.full_pop[n_i, :self.n_states]
            # Order macrostates by feature
            for i, ms in enumerate(ma.astype(bool)):
                macrostate_feature[i] = ((self.feature[ms] * pop[ms]) / pop[ms].sum()).sum()
            order = np.argsort(macrostate_feature)[::-1]
            self.macrostate_feature.append(macrostate_feature[order])
            self.macrostate_assignment.append(ma[order])

            # Calculate other macrostate related values
            self.macrostates_map.append(np.zeros(self.n_states, dtype=int))
            mas, mis = np.where(self.macrostate_assignment[-1]==1)
            self.macrostates_map[-1][mis] = mas
            self.macro_tmat.append(utils.macro_tmat(self.tmat, self.macrostate_assignment[-1], pop))
            #self.macrotraj.append(utils.translate_traj(self.traj, self.macrostates_map[-1]))
            self.macrotraj[:, n_i] = utils.translate_traj(self.traj, self.macrostates_map[-1])
            self.n_macrostates.append(self.macrostate_assignment[-1].shape[0])

    def macro_to_micro_feature(self):
        self.micro_feature = np.zeros((self.n_states, self.n_runs))
        for i, (ma, mf) in enumerate(zip(self.macrostate_assignment, self.macrostate_feature)):
            for j, mb in enumerate(ma.astype(bool)):
                self.micro_feature[mb, i] = mf[j]


    def plot(self, out: str, n_i: int = 0):
        plot_dendrogram(
            self.Z[n_i],
            self.full_pop[n_i],
            self.traj,
            self.feature_traj,
            self.macrostate_assignment[n_i],
            out,
        )
    
    def __add__(self, other):
        """
        The '+' operator is used to calculate the similarity between many
        stochastic clusterings and a reference - needs to be the right hand
        argument
        """
        if self.n_runs == 1 and other.n_runs > 1:
            # reference
            ref = self
            # stochastic clustering
            sto = other
        elif other.n_runs == 1 and self.n_runs > 1:
            ref = other
            sto = self
        else:
            raise ValueError(
                "The reference clustering must have exactly one run. The "
                "stochastic clustering must have more than one run."
            )

        # Similarity matrix
        S = np.zeros((3, ref.n_macrostates[0], sto.n_runs))
        
        print(f"Calculating similarities for {ref.n_macrostates[0]} macrostates ...")
        for n_i in tqdm(range(sto.n_runs)):
            ref_ma = ref.macrostate_assignment[0].astype(bool)
            sto_ma = sto.macrostate_assignment[n_i].astype(bool)
            for i in range(ref.n_macrostates[0]):
                for j in range(sto.n_macrostates[n_i]):
                    intersect = np.logical_and(ref_ma[i], sto_ma[j]).sum()
                    union = np.logical_or(ref_ma[i], sto_ma[j]).sum()
                    # union
                    S[0, i, n_i] = max(S[0, i, n_i], intersect / union)
                    # reference
                    S[1, i, n_i] = max(S[1, i, n_i], intersect / ref_ma[i].sum())
                    # clustering
                    S[2, i, n_i] = max(S[2, i, n_i], intersect / sto_ma[j].sum())
        return ref, sto, S

    def __mul__(self, other):
        """
        The '*' operator is used to calculate the similarity between many
        stochastic clusterings and a reference - needs to be the right hand
        argument
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
                # e_intersect = d1 * d2 / self.n_states
                # assuming random distribution:
                # n.append(S[-1].mean() / (e_intersect / (d1 + d2 - e_intersect)))
                n.append(S[-1].mean())
        return S, n

    def calc_time_scales(self, ntimescales=3):
        self.time_scales = np.zeros((self.n_runs, ntimescales))
        for i, traj in enumerate(self.macrotraj):
            self.time_scales[i] = mh.msm.implied_timescales(traj, [self.tlag], ntimescales=ntimescales)[0]

    def plot_time_scales(self, out, n_i=0):
        if self.time_scales == None:
            self.calc_time_scales()
        plt.hist(self.time_scales[n_i][:, 0])
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
        plot_macro_feature(self.micro_feature, out, ref)

