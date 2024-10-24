import numpy as np
import scipy as scy
from itertools import combinations
from sklearn.decomposition import PCA

import MPT.utils as utils

__all__ = [
    "MPTKernel",
    "SMPTKernel",
    "KLKernel",
    "FeatureKernel",
    "PCAFeatureKernel",
]


### MERGING KERNEL ###########################################################

# TEST:
# deletion of a, d, e, f
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
    def __init__(self, method="n", param=1, cutoff=0, similarity="P", b=1, c=1):
        """
        similarity:
            - P: probability
            - KL: Kullback-Leibler
            - JS: Jensen-Shannon
        """
        self.method = method
        self.param = param
        self.cutoff = cutoff
        self.similarity = similarity
        self.b = b
        self.c = c

    def __call__(self, full_tmat, states_not_merged, mask, feature_kernel=1):
        # Select state with least self transition probability
        mask_state = np.argmin(np.diag(full_tmat)[mask])
        # Get correct state index
        state = states_not_merged[mask][mask_state][0]
        mask[state] = False
    
        # if feature_kernel != 1 and feature_kernel.b != 0:
        #     trans_probs = feature_kernel.apply(full_tmat[state], state, mask)[mask]
        # else:
        trans_probs = full_tmat[state][mask]

        # Apply cutoff as suggested by Lukas (report 8)
        # c=0: consider all transitions
        if self.cutoff > 0 and self.cutoff < 1:
            p_max_i = np.where(trans_probs.max() == trans_probs)
            p_max = trans_probs.max()
            cutoff_mask = trans_probs > p_max * self.cutoff
            trans_probs = trans_probs[cutoff_mask]

        trans_probs /= trans_probs.sum()

        # If Kullback-Leibler divergence is used
        if self.similarity == "KL":
            t = full_tmat[mask][:, mask].copy()
            np.fill_diagonal(t, trans_probs)
            f1 = utils.kullback_leibler(trans_probs, t)
            if feature_kernel != 1:
                f2 = feature_kernel.kl(state, mask)
            else:
                f2 = 0
        elif self.similarity == "JS":
            t = full_tmat[mask][:, mask].copy()
            np.fill_diagonal(t, trans_probs)
            f1 = utils.jensen_shannon(trans_probs, t)
            if feature_kernel != 1:
                f2 = feature_kernel.js(state, mask)
            else:
                f2 = 0
        else:
            f1 = 0
            if feature_kernel != 1 and feature_kernel.b != 0:
                f2 = feature_kernel.apply(full_tmat[state], state, mask)[mask]
            else:
                f2 = 0

        if isinstance(f1, np.ndarray):
            if f1.shape[0] > 1:
                df1 = f1 - f1.min()
                f1 = df1 / df1.sum()

        if isinstance(f2, np.ndarray):
            if f2.shape[0] > 1:
                df2 = f2 - f2.min()
                f2 = df2 / df2.sum()

        tr_prob = trans_probs / trans_probs.sum()
        if isinstance(f1, np.ndarray):
            f1 /= f1.sum()
        if isinstance(f2, np.ndarray):
            f2 /= f2.sum()

        trans_probs = tr_prob + self.b * f1 + self.c * f2

        # transitions contains indices for masked tmat
        transitions = np.argsort(trans_probs)[::-1]
        if self.method == "p":
            t_prob_norm = trans_probs / trans_probs.sum()
            options = [0]
            while t_prob_norm[options].sum() <= self.param and len(options) < np.count_nonzero(trans_probs):
                options.append(options[-1]+1)
            # p_options_norm = t_prob_norm[options]
        elif self.method == "n":
            options = list(range(self.param))[:trans_probs.shape[0]]
        else:
            raise ValueError("Method must be either 'p' or 'n'.")
     
        p_options = trans_probs[transitions[options]]
        mask_target_state = np.random.choice(transitions[options], p=p_options / sum(p_options))
    
        if self.cutoff > 0 and self.cutoff < 1:
            target_state = states_not_merged[mask][cutoff_mask][mask_target_state][0]
        else:
            target_state = states_not_merged[mask][mask_target_state][0]
        return state, target_state, mask

    def __repr__(self):
        return f"<class MPTKernel>"


### FEATURE KERNEL ###########################################################

# TODO:
# return only KL or JS. P: calculate 1D fnc - nothing done yet, except similarity property
class FeatureKernel(object):
    def __init__(self, feature_traj, microstate_traj, sigma=0.13, b=2, feature_type=np.float64, traj_type=np.uint16, similarity="P"):
        """
        feature_traj: either N or NxM, N being the number of frames and M the
                number of features
        """
        if len(feature_traj.shape) == 1:
            self.feature_traj = np.expand_dims(feature_traj.astype(feature_type), -1)
            self.multidim_feature = False
        elif len(feature_traj.shape) == 2:
            self.feature_traj = feature_traj.astype(feature_type)
            self.multidim_feature = True
        else:
            raise ValueError("featuretraj must be a 1 D or 2 D array.")

        self.similarity = similarity
        self.sigma = sigma
        self.b = b

        self._init_feature(microstate_traj.astype(traj_type))

    def __repr__(self):
        return "<class FeatureKernel>"
    
    def _init_feature(self, microstate_traj):
        states, pop = np.unique(microstate_traj, return_counts=True)
        self.n_states = states.shape[0]
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
        self.full_pop[self.n_states:] = 0
        self.full_feature[self.n_states:] = 0

    def weighting_function(self, dq):
        a = 1 / (2 * self.sigma ** 2)
        return np.exp(-a * np.abs(dq) ** self.b)

    def apply(self, trans_probs, state, mask=None):
        # m = self.states_not_merged
        if self.multidim_feature:
            return trans_probs * self.weighting_function(
                utils.dq_kl(self.full_feature[state], self.full_feature)
            )
        else:
            return trans_probs * self.weighting_function(
                np.abs(self.full_feature - self.full_feature[state])[:, 0]
            )

    def update(self, origin, target, new_state):
        self.full_pop[new_state] = self.full_pop[[origin, target]].sum()
        self.full_feature[new_state] = (
            self.full_feature[origin] * self.full_pop[origin] \
            + self.full_feature[target] * self.full_pop[target]
        ) / self.full_pop[new_state]
    
    def kl(self, state, mask):
        return utils.kullback_leibler(
            self.full_feature[state],
            self.full_feature[mask]
        )

    def js(self, state, mask):
        return utils.jensen_shannon(
            self.full_feature[state],
            self.full_feature[mask]
        )
