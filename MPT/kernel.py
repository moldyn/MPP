import numpy as np
import scipy as scy
from itertools import combinations
from sklearn.decomposition import PCA
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
    mpt_kernel
    ----------
    MPT kernel to determine states that are merged next.

    full_tmat (np.ndarray (m, m)): m = n(n+1); Transition matrix for all
            states, including not yet defined states
    states_not_merged (np.ndarray (m, 2)): complete linkage
    mask (np.ndarray (m)): mask for states that are not yet merged
    params (dict): parameters for the kernel
    """
    def __call__(self, full_tmat, states_not_merged, mask):
        # Select state with least self transition probability
        mask_state = np.argsort(np.diag(full_tmat)[mask])[0]
        # Get correct state index
        state = states_not_merged[mask][mask_state][0]
        mask[state] = False
        # Select state with highest transition probability as target state.
        mask_target_state = np.argsort(full_tmat[state][mask])[-1]
        target_state = states_not_merged[mask][mask_target_state][0]
        return state, target_state, mask

    def __repr__(self):
        return "<class MPTKernel>"

class SMPTKernel(object):
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
    def __init__(self, method="n", param=2, c=0.1):
        self.method = method
        self.param = param
        self.c = c

    def __call__(self, full_tmat, states_not_merged, mask):
        # Select state with least self transition probability
        mask_state = np.argsort(np.diag(full_tmat)[mask])[0]
        # Get correct state index
        state = states_not_merged[mask][mask_state][0]
        mask[state] = False
     
        tmat = full_tmat[state][mask]

        # Apply cutoff as suggested by Lukas (report 8)
        if self.c > 0 and self.c < 1:
            p_max_i = np.where(tmat.max() == tmat)
            p_max = tmat.max()
            cutoff_mask = tmat > p_max * self.c
            # print(f"{mask.sum()}: {cutoff_mask.sum()}")
            tmat = tmat[cutoff_mask]
            # print(tmat)
     
        # transitions contains indices for masked tmat
        transitions = np.argsort(tmat)[::-1]
        if self.method == "p":
            t_prob_norm = tmat / tmat.sum()
            options = [0]
            while t_prob_norm[options].sum() <= self.param and len(options) < np.count_nonzero(tmat):
                options.append(options[-1]+1)
            p_options_norm = t_prob_norm[options]
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
        return f"<class SMPTKernel>"

class KLKernel(object):
    """
    KLKernel
    --------
    Kullback-Leibler kernel to determine states that are merged next.

    full_tmat (np.ndarray (m, m)): m = n(n+1); Transition matrix for all
            states, including not yet defined states
    states_not_merged (np.ndarray (m, 2)): complete linkage
    mask (np.ndarray (m)): mask for states that are not yet merged
    params (dict): parameters for the kernel
    """
    def __call__(self, full_tmat, states_not_merged, mask):
        # Select state with least self transition probability
        mask_state = np.argsort(np.diag(full_tmat)[mask])[0]
        # Get correct state index
        state = states_not_merged[mask][mask_state][0]
        mask[state] = False

        state_transitions = full_tmat[state][mask]

        # Don't modify original tmat, add 1 because KL can't treat 0 properly
        tmp_tmat = full_tmat.copy() + 1
        # Get trans probs to state
        to_state = tmp_tmat[:, state]
        # fill diagonal wtih trans prob to state to compare trans probs
        # between states themselves.
        np.fill_diagonal(tmp_tmat, to_state)

        kl = scy.stats.entropy(state_transitions, tmp_tmat[mask][:, mask], axis=1)
        # Select state with least KL divergence
        mask_target_state = np.argsort(kl)[0]
        target_state = states_not_merged[mask][mask_target_state][0]
        return state, target_state, mask

    def __repr__(self):
        return "<class KLKernel>"


### FEATURE KERNEL ###########################################################

class FeatureKernel(object):
    def __init__(self, feature_traj, microstate_traj, sigma=0.13, b=2, feature_type=np.float32, traj_type=np.uint16):
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
        n_states = states.shape[0]
        # Mark which states have not yet been merged
        self.states_not_merged = np.full(2*n_states-1, False)
        self.states_not_merged[:n_states] = True
        # Populations for all states incl intermediate states
        self.full_pop = np.zeros(2*n_states-1, dtype=np.uint32)
        self.full_pop[:n_states] = pop
        # corresponding feature values
        self.full_feature = np.zeros((2*n_states-1, self.feature_traj.shape[1]), dtype=self.feature_traj.dtype.type)
        for i in range(n_states):
            self.full_feature[i] = self.feature_traj[
                microstate_traj == i+1
            ].mean(axis=0)

    def weighting_function(self, dq):
        # NOTE:
        # Function changed here
        a = 1 / (2 * self.sigma ** 2)
        # return np.exp(-a * np.abs(dq) ** self.b)
        f = np.exp(-a * np.abs(dq) ** self.b)
        # if self.states_not_merged.sum() <= 5:
        #     print(dq)
        #     print(f)
        return f
        # return np.exp(-a * dq-1 ** self.b)
        # return np.random.uniform(0.1, 0.9, dq.shape)
        # return -dq

    @property
    def dq(self):
        """The dq property."""
        # get relevant feature values
        feature = self.full_feature[self.states_not_merged]
        return scy.spatial.distance_matrix(feature, feature).astype(self.full_feature.dtype.type)

    def apply(self, tmat):
        # NOTE:
        # Here are still some commented lines
        m = self.states_not_merged
        if tmat.shape[0] == m.sum():
            return tmat * self.weighting_function(self.dq)
            # return tmat * np.random.uniform(0, 1, self.dq.shape)
        elif tmat.shape[0] == m.shape[0]:
            # if self.states_not_merged.sum()%100 == 0:
            #     least_stable = np.argsort(np.diag(tmat[m][:, m]))[0]
            #     lt = tmat[m][:, m][least_stable]
            #     w = self.weighting_function(self.dq[least_stable])
            #     order1 = np.argsort(lt)[::-1][:5]
            #     order2 = np.argsort(lt * w)[::-1][:5]
            #     print(f"w min: {w.min()}")
            #     print(order1)
            #     print(order2)

            tmat[m][:, m] = tmat[m][:, m] \
                    * self.weighting_function(self.dq)
            # tmat[m][:, m] = tmat[m][:, m] \
            #         * np.random.uniform(0, 1, self.dq.shape)
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
        # NOTE:
        # Here, things have been adapted
        # Here's needs adaption
        self.full_feature[new_state] = (
            self.full_feature[origin] * self.full_pop[origin] \
            + self.full_feature[target] * self.full_pop[target]
        ) / self.full_pop[new_state]
        # self.full_feature[self.new_state] = self.full_feature[
        #     [origin, target]
        # ].sum(axis=0)
    
    def _update_bak(self, origin, target):
        # Doesn't work for more than one run since new_state simply counts on
        self.states_not_merged[[origin, target]] = False
        self.states_not_merged[self.new_state] = True
        self.full_pop[self.new_state] = self.full_pop[[origin, target]].sum()
        # NOTE:
        # Here, things have been adapted
        # Here's needs adaption
        self.full_feature[self.new_state] = (
            self.full_feature[origin] * self.full_pop[origin] \
            + self.full_feature[target] * self.full_pop[target]
        ) / self.full_pop[self.new_state]
        # self.full_feature[self.new_state] = self.full_feature[
        #     [origin, target]
        # ].sum(axis=0)
        self.new_state += 1

class PCAFeatureKernel(object):
    def __init__(self, feature_traj, microstate_traj, sigma=0.13, b=2, n_PCs=3):
        """
        feature_traj: either N or NxM, N being the number of frames and M the
                number of features
        """
        if len(feature_traj.shape) == 1:
            self.feature_traj = np.expand_dims(feature_traj, -1)
        elif len(feature_traj.shape) == 2:
            self.feature_traj = feature_traj
        else:
            raise ValueError("featuretraj must be a 1 D or 2 D array.")

        self.sigma = sigma
        self.b = b
        self.n_PCs = n_PCs

        self._init_feature(microstate_traj)

    def __repr__(self):
        return f"<class PCAFeatureKernel>"
    
    def _init_feature(self, microstate_traj):
        states, pop = np.unique(microstate_traj, return_counts=True)
        n_states = states.shape[0]
        # Mark which states have not yet been merged
        self.states_not_merged = np.full(2*n_states-1, False)
        self.states_not_merged[:n_states] = True
        # Index of next state
        self.new_state = n_states
        # Populations for all states incl intermediate states
        self.full_pop = np.zeros(2*n_states-1)
        self.full_pop[:n_states] = pop
        # Perform PCA
        pca = PCA(n_components=self.n_PCs)
        pca_feature = pca.fit_transform(self.feature_traj)
        # corresponding feature values
        self.full_feature = np.zeros((2*n_states-1, pca_feature.shape[1]))
        for i in range(n_states):
            self.full_feature[i] = pca_feature[
                microstate_traj == i+1
            ].mean(axis=0)

    def weighting_function(self, dq):
        a = 1 / (2 * self.sigma ** 2)
        return np.exp(-a * np.abs(dq) ** self.b)

    @property
    def dq(self):
        """The dq property."""
        # get relevant feature values
        feature = self.full_feature[self.states_not_merged]
        sparse = np.array(list(combinations(range(feature.shape[0]), 2))).T
        kl = entropy(feature[sparse[0]], feature[sparse[1]], axis=1)

#        return scy.spatial.distance_matrix(feature, feature)
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

class MultiFeatureKullbackLeiblerKernel(object):
    def __init__(self, feature_traj, microstate_traj, sigma=0.13, b=2, features="all", contact_distance=0.45):
        """
        feature_traj: either N or NxM, N being the number of frames and M the
                number of features
        features: list of indices of features to use
        contact_distance: below this, contacts are considered established, in nm
        """
        if len(feature_traj.shape) == 2:
            if features == "all":
                self.feature_traj = feature_traj
            else:
                self.feature_traj = feature_traj[:, features]
        else:
            raise ValueError("featuretraj must be a 2 D array.")

        self.features = features
        self.sigma = sigma
        self.b = b
        self.contact_distance = contact_distance

        self._init_feature(microstate_traj)

    def __repr__(self):
        return "<class MultiFeatureKullbackLeiblerKernel>"
   
    def _init_mat(self):
        probs = np.zeros((n_states, self.feature_traj.shape[1]))
        for state in range(n_states):
            probs[state] = (dist[:, contacts][traj==state+1] < thr).mean(axis=0)
        p_mat = np.zeros((probs.shape[0], probs.shape[0]))
        for i, pi in enumerate(probs[:-1]):
            for j, pj in enumerate(probs[i+1:], start=i+1):
                p_mat[i, j] = p_mat[j, i] = entropy(pi, pj)


    def _init_feature(self, microstate_traj):
        states, pop = np.unique(microstate_traj, return_counts=True)
        n_states = states.shape[0]
        # Mark which states have not yet been merged
        self.states_not_merged = np.full(2*n_states-1, False)
        self.states_not_merged[:n_states] = True
        # Index of next state
        self.new_state = n_states
        # Populations for all states incl intermediate states
        self.full_pop = np.zeros(2*n_states-1)
        self.full_pop[:n_states] = pop
        # corresponding feature values
        self.full_feature = np.zeros((2*n_states-1, self.feature_traj.shape[1]))
        for i in range(n_states):
            self.full_feature[i] = (
                self.feature_traj[microstate_traj==i+1] < self.contact_distance
            ).mean(axis=0) + 1

    def weighting_function(self, dq):
        a = 1 / (2 * self.sigma ** 2)
        return np.exp(-a * np.abs(dq) ** self.b)

    @property
    def dq(self):
        """
        The distance between two samples. 0 is closest, greater values mean further away.
        
        Returns NxN np.ndarray: N is the number of states not merged
        """
#         n_states_to_merge = self.states_not_merged.sum()
#         p_mat = np.zeros((n_states_to_merge, n_states_to_merge))
#         for i, pi in enumerate(self.full_feature[self.states_not_merged][:-1]):
#             for j, pj in enumerate(self.full_feature[self.states_not_merged][i+1:], start=i+1):
#                 p_mat[i, j] = p_mat[j, i] = entropy(pi, pj)
#         # Transform matrix to reperesent a fitness between 0 and 1
# #        return (p_mat.max() - p_mat) / (p_mat.max() - p_mat.min())
#         return p_mat
        # get relevant feature values
        feature = self.full_feature[self.states_not_merged]
        sparse = np.array(list(combinations(range(feature.shape[0]), 2))).T
        kl = entropy(feature[sparse[0]], feature[sparse[1]], axis=1)

#        return scy.spatial.distance_matrix(feature, feature)
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


### WEIGHTING FUNCTIONS ######################################################

class Gaussian:
    def __init__(self, sigma=0.13, b=2):
        self.sigma = sigma
        self.b = b

    def __call__(self, dq):
        a = 1 / (2 * self.sigma ** 2)
        return np.exp(-a * np.abs(dq) ** self.b)

class Linear:
    def __init__(self, m=1.0, offset=0.0):
        self.m = m
        self.offset = offset

    def __call__(self, dq):
        return np.clip(1 - np.abs(dq) * self.m + self.offset)
