import numpy as np
import scipy as scy

from . import utils

__all__ = [
    "LumpingKernel",
    "FeatureKernel",
]


### MERGING KERNEL ###########################################################


class LumpingKernel(object):
    """Kernel for the most probable path (MPP) algorithm.

    This object holds the parameters of the lumping and analyzes the
    full transition matrix of the lumping based on a mask of not yet
    merged states, upon calling.

    Notes
    -----
    The similarity between two states may be composed of a dynamic
    similarity (defined in this object, c.f. parameter
    <similarity>) and / or a geometric similarity, which is
    determined by the feature kernel (passed at call).
    """

    def __init__(self, method="n", param=1, similarity="T"):
        """Initialize LumpingKernel

        Parameters
        ----------
        method : str
            'n' : Consider <param> most similar options. (default)
            'p' : Consider as many most similar options as needed to
                represent <param> similarity. For similarity 'T',
                <param>=0.5 means that at least 50% of the transitions
                to other states must be considered.
        param : int|float
            for 'n' : Number of most similar options to consider
                (1 deterministic lumping). (default 1)
            for 'p' : Accumulated similarity threshold for most similar
                states to consider.
        similarity:
            - T: Utilize the transition probabilities as dynamic
                metric. (default)
            - KL: Utilize the Kullback-Leibler divergence between the
                transition probabilities of the options.
            - none: Utilize only the feature as similarity measure.
        """
        self.method = method
        self.param = param
        self.similarity = similarity

    def __call__(self, full_tmat, states_not_merged, mask, feature_kernel=None):
        """
        Finds the states to be lumped together next.

        The least metastable state is selected and the most similar
        other state is determined. The similarity of two states is
        determined by the parameters of the object and the feature
        kernel.
        """
        # Select state with least self transition probability
        mask_state = np.argmin(np.diag(full_tmat)[mask])
        # Get correct state index
        state = states_not_merged[mask][mask_state][0]

        mask[state] = False
        trans_probs = full_tmat[state][mask]
        trans_probs /= trans_probs.sum()

        # If Kullback-Leibler divergence is used
        if self.similarity == "KL":
            # Mask self transition probabilities
            t = full_tmat[mask][:, mask].copy()
            np.fill_diagonal(t, trans_probs)

            # Regularization parameter
            epsilon = 1e-6
            kl = scy.stats.entropy(
                trans_probs + epsilon,
                t + epsilon,
                axis=1,
            )
            dkl = utils.weighting_function(kl)
            if dkl.shape[0] > 1:
                dkl -= dkl.min()
                dkl /= dkl.sum()
            trans_probs = dkl
        elif self.similarity == "none":
            trans_probs = 1

        # Apply feature kernel, if there is one
        if feature_kernel:
            feature = feature_kernel.apply(state, mask)
            if not isinstance(feature, np.ndarray):  # and feature == 0:
                feature = np.array([1.0])
            trans_probs *= feature

        trans_probs = np.nan_to_num(trans_probs, copy=False, nan=1e-6)

        # transitions contains indices for masked tmat
        transitions = np.argsort(trans_probs)[::-1]
        # consider n most similar options
        if self.method == "n":
            options = list(range(self.param))[: trans_probs.shape[0]]
        # consider as many most probable options until they sum up to param
        elif self.method == "p":
            t_prob_norm = trans_probs / trans_probs.sum()
            options = [0]
            while t_prob_norm[options].sum() <= self.param and len(
                options
            ) < np.count_nonzero(trans_probs):
                options.append(options[-1] + 1)
        else:
            raise ValueError("Method must be either 'p' or 'n'")

        # Get similarities of the options
        p_options = (
            trans_probs[transitions[options]] if len(options) > 1 else np.ones(1)
        )
        if np.isnan(p_options).any():
            raise ValueError(f"p_options contains NaN: {p_options}")
        if sum(p_options) == 0:
            raise ValueError(f"sum of p_options is 0: {p_options}")

        # Select the target state
        mask_target_state = np.random.choice(
            transitions[options], p=p_options / sum(p_options)
        )

        target_state = states_not_merged[mask][mask_target_state][0]
        return state, target_state, mask

    def __repr__(self):
        return "<class LumpingKernel>"


class FeatureKernel(object):
    def __init__(
        self,
        feature_trajectory,
        microstate_trajectory,
        feature_type=np.float64,
        trajectory_type=np.uint16,
    ):
        """
        feature_trajectory: either N or NxM, N being the number of frames and M the
                number of features
        """
        if feature_trajectory.ndim == 2:
            self.feature_trajectory = feature_trajectory.astype(feature_type)
        else:
            raise ValueError("featuretrajectory must be a 2 D array.")

        self._init_feature(microstate_trajectory.astype(trajectory_type))

    def __repr__(self):
        return "<class FeatureKernel>"

    def _init_feature(self, microstate_trajectory):
        if microstate_trajectory.min() == 1:
            microstate_trajectory -= 1
        states, pop = np.unique(microstate_trajectory, return_counts=True)
        self.n_states = states.shape[0]
        # Populations for all states incl intermediate states
        self.full_pop = np.zeros(2 * self.n_states - 1, dtype=np.uint32)
        self.full_pop[: self.n_states] = pop
        # corresponding feature values
        self.full_feature = np.zeros(
            (2 * self.n_states - 1, self.feature_trajectory.shape[1]),
            dtype=self.feature_trajectory.dtype.type,
        )
        for i in range(self.n_states):
            self.full_feature[i] = self.feature_trajectory[
                microstate_trajectory == i
            ].mean(axis=0)

    def reset(self):
        self.full_pop[self.n_states :] = 0
        self.full_feature[self.n_states :] = 0

    def apply(self, state, mask):
        f = self.js(state, mask)
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
            (Z.shape[0], full_dim, self.feature_trajectory.shape[1])
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
