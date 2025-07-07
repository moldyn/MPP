import numpy as np
import scipy as scy

import MPT.utils as utils

__all__ = [
    "MPTKernel",
    "FeatureKernel",
]


### MERGING KERNEL ###########################################################


class MPTKernel(object):
    def __init__(self, method="n", param=1, cutoff=0, similarity="T", term="*"):
        """
        similarity:
            - T: transition probability
            - KL: Kullback-Leibler
            - JS: Jensen-Shannon
            - F: Use only feature kernel
        term:
            - +: sum
            - *: product
        """
        # , a=1, b=0, c=0
        self.method = method
        self.param = param
        self.cutoff = cutoff
        self.similarity = similarity
        if self.similarity == "T":
            self.a = 1
            self.b = 0
        elif self.similarity == "KL" or self.similarity == "JS":
            self.a = 0
            self.b = 1
        elif self.similarity is None or self.similarity == "none":
            self.a = 0
            self.b = 0
        # self.a = a
        # self.b = b
        self.c = 0
        self.term = term

    def __call__(self, full_tmat, states_not_merged, mask, feature_kernel=None):
        if feature_kernel:
            self.c = 1
        else:
            self.c = 0
        # Select state with least self transition probability
        mask_state = np.argmin(np.diag(full_tmat)[mask])
        # Get correct state index
        state = states_not_merged[mask][mask_state][0]

        mask[state] = False

        trans_probs = full_tmat[state][mask]

        # Apply cutoff as suggested by Lukas (report 8)
        # c=0: consider all transitions
        if self.cutoff > 0 and self.cutoff < 1:
            p_max = trans_probs.max()
            cutoff_mask = trans_probs > p_max * self.cutoff
            trans_probs = trans_probs[cutoff_mask]

        trans_probs /= trans_probs.sum()

        # If Kullback-Leibler divergence is used
        if self.similarity == "KL":
            t = full_tmat[mask][:, mask].copy()
            np.fill_diagonal(t, trans_probs)
            epsilon = 1e-6
            dkl = scy.stats.entropy(
                trans_probs + epsilon,
                t + epsilon,
                axis=1,
            )
            f1 = utils.weighting_function(dkl)
            # f1 = utils.kullback_leibler(trans_probs, t)
        elif self.similarity == "JS":
            t = full_tmat[mask][:, mask].copy()
            np.fill_diagonal(t, trans_probs)
            p = trans_probs.copy()
            q = t
            if p.ndim == 1:
                p = np.expand_dims(p, axis=0)
            if q.ndim == 1:
                q = np.expand_dims(q, axis=0)
            djs = scy.spatial.distance.jensenshannon(p, q, axis=1) ** 2
            f1 = utils.weighting_function(djs)
            # f1 = utils.jensen_shannon(trans_probs, t)
        else:
            f1 = 0

        if isinstance(f1, np.ndarray):
            if f1.shape[0] > 1:
                df1 = f1 - f1.min()
                f1 = df1 / df1.sum()

        tr_prob = trans_probs / trans_probs.sum()

        if feature_kernel:
            f2 = feature_kernel.apply(full_tmat[state], state, mask)
        else:
            f2 = 0

        if self.term == "+":
            trans_probs = self.a * tr_prob + self.b * f1 + self.c * f2
            if not isinstance(trans_probs, np.ndarray) and trans_probs == 0:
                trans_probs = np.array([1.0])
        elif self.term == "*":
            trans_probs = 1
            if self.a != 0:
                trans_probs *= tr_prob
            if self.b != 0:
                trans_probs *= f1
            if self.c != 0:
                if not isinstance(f2, np.ndarray) and f2 == 0:
                    f2 = np.array([1.0])
                trans_probs *= f2
        trans_probs = np.nan_to_num(trans_probs, copy=False, nan=1e-6)
        if trans_probs.sum() == 0:
            trans_probs = f1

        # transitions contains indices for masked tmat
        transitions = np.argsort(trans_probs)[::-1]
        # print(mask.sum())
        if self.method == "p":
            t_prob_norm = trans_probs / trans_probs.sum()
            options = [0]
            while t_prob_norm[options].sum() <= self.param and len(
                options
            ) < np.count_nonzero(trans_probs):
                options.append(options[-1] + 1)
            # p_options_norm = t_prob_norm[options]
        elif self.method == "n":
            # print(trans_probs)
            options = list(range(self.param))[: trans_probs.shape[0]]
        else:
            raise ValueError("Method must be either 'p' or 'n'.")

        p_options = trans_probs[transitions[options]]

        mask_target_state = np.random.choice(
            transitions[options], p=p_options / sum(p_options)
        )

        if self.cutoff > 0 and self.cutoff < 1:
            target_state = states_not_merged[mask][cutoff_mask][mask_target_state][0]
        else:
            target_state = states_not_merged[mask][mask_target_state][0]
        return state, target_state, mask

    def __repr__(self):
        return "<class MPTKernel>"


### FEATURE KERNEL ###########################################################


class FeatureKernel(object):
    def __init__(
        self,
        feature_traj,
        microstate_traj,
        sigma=0.05,
        b=2,
        feature_type=np.float64,
        traj_type=np.uint16,
    ):
        """
        feature_traj: either N, N being the number of frames and M the
                number of features
        """
        if len(feature_traj.shape) == 1:
            self.feature_traj = feature_traj.astype(feature_type)
        else:
            raise ValueError("featuretraj must be a 1 D array.")

        self.sigma = sigma
        self.b = b

        self._init_feature(microstate_traj.astype(traj_type))

    def __repr__(self):
        return "<class FeatureKernel>"

    def _init_feature(self, microstate_traj):
        states, pop = np.unique(microstate_traj, return_counts=True)
        self.n_states = states.shape[0]
        # Populations for all states incl intermediate states
        self.full_pop = np.zeros(2 * self.n_states - 1, dtype=np.uint32)
        self.full_pop[: self.n_states] = pop
        # corresponding feature values
        self.full_feature = np.zeros(
            2 * self.n_states - 1, dtype=self.feature_traj.dtype.type
        )
        for i in range(self.n_states):
            self.full_feature[i] = self.feature_traj[microstate_traj == i + 1].mean()

    def reset(self):
        self.full_pop[self.n_states :] = 0
        self.full_feature[self.n_states :] = 0

    def apply(self, trans_probs, state, mask):
        f = (
            trans_probs
            * utils.weighting_function(
                np.abs(self.full_feature - self.full_feature[state])
            )
        )[mask]
        f -= f.min()
        if f.sum() != 0:
            return f / f.sum()
        else:
            return 0

    def update(self, origin, target, new_state):
        self.full_pop[new_state] = self.full_pop[[origin, target]].sum()
        self.full_feature[new_state] = (
            self.full_feature[origin] * self.full_pop[origin]
            + self.full_feature[target] * self.full_pop[target]
        ) / self.full_pop[new_state]

    def full_feature_from_Z(self, Z):
        # Ensure that Z is 3D
        if Z.ndim == 2:
            Z = Z.reshape((1, *Z.shape))

        full_dim = 2 * self.n_states - 1

        self.n_full_feature = np.empty((Z.shape[0], full_dim))
        self.n_full_feature[:, : self.n_states] = self.full_feature[: self.n_states]
        for run, z in enumerate(Z):
            self.reset()
            for i, (origin, target) in enumerate(z[:, :2].astype(int)):
                self.update(origin, target, self.n_states + i)
            self.n_full_feature[run, self.n_states :] = self.full_feature[
                self.n_states :
            ]
        return self.n_full_feature


class MultiFeatureKernel(object):
    def __init__(
        self,
        feature_traj,
        microstate_traj,
        feature_type=np.float64,
        traj_type=np.uint16,
        similarity="JS",
    ):
        """
        feature_traj: either N or NxM, N being the number of frames and M the
                number of features
        """
        if len(feature_traj.shape) == 2:
            self.feature_traj = feature_traj.astype(feature_type)
        else:
            raise ValueError("featuretraj must be a 2 D array.")

        self.similarity = similarity

        self._init_feature(microstate_traj.astype(traj_type))

    def __repr__(self):
        return "<class MultiFeatureKernel>"

    def _init_feature(self, microstate_traj):
        states, pop = np.unique(microstate_traj, return_counts=True)
        self.n_states = states.shape[0]
        # Populations for all states incl intermediate states
        self.full_pop = np.zeros(2 * self.n_states - 1, dtype=np.uint32)
        self.full_pop[: self.n_states] = pop
        # corresponding feature values
        self.full_feature = np.zeros(
            (2 * self.n_states - 1, self.feature_traj.shape[1]),
            dtype=self.feature_traj.dtype.type,
        )
        for i in range(self.n_states):
            self.full_feature[i] = self.feature_traj[microstate_traj == i + 1].mean(
                axis=0
            )

    def reset(self):
        self.full_pop[self.n_states :] = 0
        self.full_feature[self.n_states :] = 0

    def apply(self, trans_prob, state, mask):
        if self.similarity == "KL":
            f = self.kl(state, mask)
        elif self.similarity == "JS":
            f = self.js(state, mask)
        else:
            raise ValueError(f"Invalid similarity: {self.similarity}")
        f -= f.min()
        if f.sum() != 0:
            return f / f.sum()
        else:
            return 0

    def update(self, origin, target, new_state):
        self.full_pop[new_state] = self.full_pop[[origin, target]].sum()
        self.full_feature[new_state] = (
            self.full_feature[origin] * self.full_pop[origin]
            + self.full_feature[target] * self.full_pop[target]
        ) / self.full_pop[new_state]

    def kl(self, state, mask, epsilon=1e-6):
        dkl = scy.stats.entropy(
            self.full_feature[state] + epsilon,
            self.full_feature[mask] + epsilon,
            axis=1,
        )
        return utils.weighting_function(dkl)

    def js(self, state, mask):
        p = self.full_feature[state]
        q = self.full_feature[mask]
        if p.ndim == 1:
            p = np.expand_dims(p, axis=0)
        if q.ndim == 1:
            q = np.expand_dims(q, axis=0)
        djs = scy.spatial.distance.jensenshannon(p, q, axis=1) ** 2
        return utils.weighting_function(djs)

    def full_feature_from_Z(self, Z):
        # Ensure that Z is 3D
        if Z.ndim == 2:
            Z = Z.reshape((1, *Z.shape))

        full_dim = 2 * self.n_states - 1

        self.n_full_feature = np.empty(
            (Z.shape[0], full_dim, self.feature_traj.shape[1])
        )
        self.n_full_feature[:, : self.n_states] = self.full_feature[: self.n_states]
        for run, z in enumerate(Z):
            self.reset()
            for i, (origin, target) in enumerate(z[:, :2].astype(int)):
                self.update(origin, target, self.n_states + i)
            self.n_full_feature[run, self.n_states :] = self.full_feature[
                self.n_states :
            ]
        return self.n_full_feature
