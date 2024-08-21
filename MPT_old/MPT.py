import numpy as np
import msmhelper as mh
from collections import defaultdict, deque
from numba import njit
from numba.typed import Dict, List
from numba.core import types

import core
import utils
#from core import find_all_nodes

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

class MPTBase:
    """
MPTBase - Base class for MPT
-------

Basic functionality for any MPT implementation

    """
    def __init__(self, trajectory: str | np.ndarray, tlag: int, method: str="mpt", params: dict={}):
        """
        trajectory (str|np.ndarray):    path to trajectory file or trajectory
                as np.ndarray.
        tlag (int):                     lag time
        """
        self.method = method
        self.params = params

        # Load trajectory
        if isinstance(trajectory, str):
            self.traj = np.loadtxt(trajectory, dtype=int)
        elif isinstance(trajectory, np.ndarray):
            self.traj = trajectory.astype(int)
        else:
            raise TypeError("trajectory must be of type 'str' or 'numpy.ndarray'.")

        # Lag time
        self.tlag = int(tlag)

        # Transition matrix, state indices, state populations
        self.tmat, self.states = mh.msm.estimate_markov_model(self.traj, tlag)
        _, pop = np.unique(self.traj, return_counts=True)
        self.pop = pop / len(self.traj)
        self.n_states = len(self.states)

        # Dict containing features
        self.features = {}

        # Linkage matrix, shape: Nx2
        # first column: origin state
        # second column: target state
        self.linkage = np.array([])

    def add_feature(self, feature_name, data):
        if not feature_name in self.features:
            self.features[feature_name] = data
        else:
            print(f"Feature '{feature_name}' exists already.")

    def replace_feature(self, feature_name, data):
        self.features[feature_name] = data

    def apply_feature(self, feature_name):
        self.tmat = utils.apply_feature(
            self.tmat,
            self.traj,
            self.features[feature_name]
        )

    def _merge(self, tmat: np.ndarray, origin: int, target: int) -> np.ndarray:
        """
        Update a transition matrix by merging one state to another.

        tmat (np.ndarray(float) MxM): transition matrix, M is the number of
                states
        origin (int): index of origin state
        target (int): index of target state

        returns altered matrix
        """
        # Weigh transition probabilities by population
        p_o = self.merge_pop[origin]
        p_t = self.merge_pop[target]
        p_sum = p_o + p_t
        tmat[target] = (tmat[target] * p_t + tmat[origin] * p_o) / p_sum
        tmat[:, target] += tmat[:, origin]
        tmat[origin] = 0
        tmat[:, origin] = 0
        self.merge_pop[target] = p_sum
        #        self.merge_pop[origin] = 0
        return tmat

    def _map_states(self):#, n_macrostates: int):
        """
        Map microstates to n macrostates using the linkage. The state map and
        the macrostate trajectory are created.

        n_macrostates (int): number of macrostates.
        """
        roots = list(set(range(self.n_states)) - set(self.linkage[:, 0]))
        self.macrostates = defaultdict(list)
        # state_map is zero based (indices)
        for macrostate, root in enumerate(roots):
            self.macrostates[macrostate] += core.find_all_nodes(self.linkage, root)

        # create map of states. dict keys are microstates, values are
        # corresponding macrostates
        state_map = np.array([
            [micro, macro] for macro in self.macrostates for micro in self.macrostates[macro]
        ])
        self.state_map = state_map[np.argsort(state_map[:, 0])]
        # create macrostate trajectory
        self.macro_traj = self.state_map[self.traj-1] + 1

    def _Z(self):
        """
        Generate Z matrix as required by scipy.cluster.hierarchy.dendrogram
        """
        self.Z = np.zeros((self.n_states-1, 4))
        # set Z temporarily 0 based
        self.Z[:, :2] = self.full_linkage-1
        # set self transition probabilities as feature
        self.Z[:, 2] = self.full_stp
        # initiate count for each state at 1
        self.Z[:, 3] = 1
        for i, (origin, target) in enumerate(self.Z[:, :2]):
            # all occurences of target state
            target_in_Z = self.Z[i:, :2] == target
            # increase state count for every occurence by one
            self.Z[i:, 3][np.logical_or(target_in_Z[:, 0], target_in_Z[:, 1])] += 1
            # alter state index. Target states get new index (n + i)
            # -1 because its 0 based at this point
            self.Z[i+1:, :2][target_in_Z[1:]] = self.n_states - 1 + i
        # reset Z matrix to 1 based
        self.Z[:, :2] += 1

    def _mpt(self, method: str="mpt", params: dict={}):
        """
        Most probable transition (MPT) algorithm. Start with the least stable
        (least self transition probability) state. If two states have the same
        stability, chose the lower populated one.

        method (str): method used for clustering. One of
                - "mpt": standard MPT algorithm
                - "smpt": stochastic MPT algorithm
        params (dict): parameters required for clustering methods
                - "smpt":
                  - "%": (float) fraction of states that is at least coverd by
                        considered merging options (e. g. 0.7 -> states with
                        transition probabilities of 0.5 and 0.25 are
                        considered)
                  - "n": (int) number of most probable transitions to consider
                        (e. g. most probable n transitions are considered)
        """
        # array of states that haven't been merged yet
        states_not_merged = np.full(self.n_states, True)
        # prepare linkage matrix, state indices are used.
        self.full_linkage = np.zeros((self.n_states-1, 2), dtype=int)
        # self transition probability at every merging
        self.full_stp = np.zeros((self.n_states-1))
        # Transition matrix to update transition probabilities in the course
        # of merging
        tmat_merging = self.tmat.copy()
        self.merge_pop = self.pop.copy()
        #for i, state in enumerate(merge_order[:-n_macrostates]):
        for i in range(self.n_states-1):
            # self transition probability and population
            stp_pop = np.vstack([np.diag(tmat_merging), self.merge_pop]).T
            # first sorting criterion: self transition probability
            # second sorting criterion: population
            # yields state indices
            state = np.lexsort((stp_pop[:, 1], stp_pop[:, 0]))[i]
            self.full_stp[i] = stp_pop[state, 0]

            states_not_merged[state] = False
            if method == "mpt":
                target_state = np.argmax(tmat_merging[state] * states_not_merged)

            elif method == "smpt":
                transitions = np.argsort(tmat_merging[state] * states_not_merged)[::-1]
                if "%" in params:
                    t_prob_norm = tmat_merging[state] / tmat_merging[state].sum()
                    options = [0]
                    while t_prob_norm[options].sum() < params["%"]:
                        options.append(options[-1]+1)
                    p_options_norm = t_prob_norm[options]
                elif "n" in params:
                    options = list(range(params["n"]))
                else:
                    raise ValueError("Either '%' or 'n' must be given in params.")

                p_options = tmat_merging[state, transitions[options]] * states_not_merged[transitions[options]]
                target_state = np.random.choice(transitions[options], p=p_options / sum(p_options)) #, p=p_options_norm / sum(p_options_norm)) #, p=p_options / sum(p_options))

            else:
                raise ValueError(f"No valid method specified: {method}")
            self.full_linkage[i] = [state, target_state]
            tmat_merging = self._merge(tmat_merging, state, target_state)


    def mpt(self, n_macrostates: int):
        """
        Select a number of macrostates.

        n_macrostates (int): limit for number of macrostates
        """
        # run MPT for one macrostate to get the linkage for the entire merging
        # tree
        self._mpt(self.method, self.params)

        self.linkage = self.full_linkage[:-n_macrostates + 1]
        self.stp = self.full_stp[:-n_macrostates + 1]
        self._Z()

        # State map
        self._map_states()


    def mpp(self):
        pass

