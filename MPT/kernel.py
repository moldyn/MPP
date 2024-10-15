import numpy as np
import scipy as scy
from itertools import combinations
from scipy.stats import entropy # aka Kullback-Leibler divergence

import MPT.utils as utils

__all__ = [
    "MPTKernel",
    "SMPTKernel",
    "KLKernel",
    "FeatureKernel",
    "PCAFeatureKernel",
]


### MERGING KERNEL ###########################################################

class MPTKernel(object):
    """
    smpt_kernel
    -----------
    Stochastic MPT kernel to determine states that are merged next.

    full_tmat (np.ndarray (m, m)): m = n(n+1); Transition matrix for all
            states, including not yet defined states
    states_not_merged (np.ndarray (m, 2)): complete linkage
    mask (np.ndarray (m)): mask for states that are not yet merged
    params (dict): parameters for the kernel
    """
    def __init__(self, method="n", param=1, c=0, kullback_leibler=False):
        self.method = method
        self.param = param
        self.c = c
        self.kullback_leibler = kullback_leibler

    def __call__(self, full_tmat, states_not_merged, mask, feature_kernel=1):
        # Select state with least self transition probability
        mask_state = np.argmin(np.diag(full_tmat)[mask])
        # Get correct state index
        state = states_not_merged[mask][mask_state][0]
        mask[state] = False
    
        if feature_kernel != 1 and feature_kernel.b != 0:
            tmat = feature_kernel.apply(full_tmat)[state][mask]
        else:
            tmat = full_tmat[state][mask]

        # Apply cutoff as suggested by Lukas (report 8)
        # c=0: consider all transitions
        if self.c > 0 and self.c < 1:
            p_max_i = np.where(tmat.max() == tmat)
            p_max = tmat.max()
            cutoff_mask = tmat > p_max * self.c
            tmat = tmat[cutoff_mask]

        # If Kullback-Leibler divergence is used
        if self.kullback_leibler:
            tmat = utils.kullback_leibler_probability(tmat, full_tmat[mask][:, mask])

        # transitions contains indices for masked tmat
        transitions = np.argsort(tmat)[::-1]
        if self.method == "p":
            t_prob_norm = tmat / tmat.sum()
            options = [0]
            while t_prob_norm[options].sum() <= self.param and len(options) < np.count_nonzero(tmat):
                options.append(options[-1]+1)
            # p_options_norm = t_prob_norm[options]
        elif self.method == "n":
            options = list(range(self.param))[:tmat.shape[0]]
        else:
            raise ValueError("Method must be either 'p' or 'n'.")
     
        p_options = tmat[transitions[options]]
        mask_target_state = np.random.choice(transitions[options], p=p_options / sum(p_options))
    
        if self.c > 0 and self.c < 1:
            target_state = states_not_merged[mask][cutoff_mask][mask_target_state][0]
        else:
            target_state = states_not_merged[mask][mask_target_state][0]
        return state, target_state, mask

    def __repr__(self):
        return f"<class MPTKernel>"


### FEATURE KERNEL ###########################################################

class FeatureKernel(object):
    def __init__(self, feature_traj, microstate_traj, sigma=0.13, b=2, feature_type=np.float64, traj_type=np.uint16):
        """
        feature_traj: either N or NxM, N being the number of frames and M the
                number of features
        """
        if len(feature_traj.shape) == 1:
            self.feature_traj = np.expand_dims(feature_traj.astype(feature_type), -1)
        elif len(feature_traj.shape) == 2:
            self.feature_traj = feature_traj.astype(feature_type)
        else:
            raise ValueError("featuretraj must be a 1 D or 2 D array.")

        self.sigma = sigma
        self.b = b

        self._init_feature(microstate_traj.astype(traj_type))

    def __repr__(self):
        return "<class FeatureKernel>"
    
    def _init_feature(self, microstate_traj):
        states, pop = np.unique(microstate_traj, return_counts=True)
        self.n_states = states.shape[0]
        # Mark which states have not yet been merged
        self.states_not_merged = np.full(2*self.n_states-1, False)
        self.states_not_merged[:self.n_states] = True
        # Populations for all states incl intermediate states
        self.full_pop = np.zeros(2*self.n_states-1, dtype=np.uint32)
        self.full_pop[:self.n_states] = pop
        # corresponding feature values
        self.full_feature = np.zeros((2*self.n_states-1, self.feature_traj.shape[1]), dtype=self.feature_traj.dtype.type)
        for i in range(self.n_states):
            self.full_feature[i] = self.feature_traj[
                microstate_traj == i+1
            ].mean(axis=0)

    def reset(self):
        self.states_not_merged[:self.n_states] = True
        self.states_not_merged[self.n_states:] = False
        self.full_pop[self.n_states:] = 0
        self.full_feature[self.n_states:] = 0

    def weighting_function(self, dq):
        # NOTE:
        # Function changed here
        a = 1 / (2 * self.sigma ** 2)
        f = np.exp(-a * np.abs(dq) ** self.b)

        return f

    @property
    def dq(self):
        """The dq property."""
        # get relevant feature values
        feature = self.full_feature[self.states_not_merged]
        return scy.spatial.distance_matrix(feature, feature).astype(self.full_feature.dtype.type)

    def apply(self, tmat, state=None):
        m = self.states_not_merged
        if len(tmat.shape) == 1:
            if tmat.shape[0] > m.sum():
                tmat[m] = tmat[m] * self.weighting_function(self.dq[state-(~m[:state]).sum()])
                return tmat
            else:
                return tmat * self.weighting_function(self.dq[state])
        elif tmat.shape[0] == m.sum():
            return tmat * self.weighting_function(self.dq)
        elif tmat.shape[0] == m.shape[0]:
            t = tmat.copy()
            t[np.ix_(m, m)] = t[m][:, m] \
                    * self.weighting_function(self.dq)
            return t
        else:
            raise ValueError(
                "Mismatch in tmat shape. Did you update this kernel?"
            )

    def update(self, origin, target, new_state):
        self.states_not_merged[[origin, target]] = False
        self.states_not_merged[new_state] = True
        self.full_pop[new_state] = self.full_pop[[origin, target]].sum()
        self.full_feature[new_state] = (
            self.full_feature[origin] * self.full_pop[origin] \
            + self.full_feature[target] * self.full_pop[target]
        ) / self.full_pop[new_state]
    
        feature = self.full_feature[self.states_not_merged]
        sparse = np.array(list(combinations(range(feature.shape[0]), 2))).T
        kl = entropy(feature[sparse[0]], feature[sparse[1]], axis=1)

        return utils.sparse_to_matrix(kl)

    def apply(self, tmat):
        m = self.states_not_merged
        if tmat.shape[0] == m.sum():
            return tmat * self.weighting_function(self.dq)
        elif tmat.shape[0] == m.shape[0]:
            tmat[m][:, m] = tmat[m][:, m] \
                    * self.weighting_function(self.dq)
            return tmat
        else:
            raise ValueError(
                "Mismatch in tmat shape. Did you update this kernel?"
            )

    def __mul__(self, other):
        if isinstance(other, np.ndarray):
            return self.apply(other)
        return NotImplemented

    def update(self, origin, target, new_state):
        self.states_not_merged[[origin, target]] = False
        self.states_not_merged[new_state] = True
        self.full_pop[new_state] = self.full_pop[[origin, target]].sum()
        self.full_feature[new_state] = self.full_feature[
            [origin, target]
        ].sum(axis=0)

