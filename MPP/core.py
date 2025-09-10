"""
core.py
=======

Core functions for MPT class
"""

__all__ = [
    "cluster",
]

import sys
import numpy as np
from typing import Callable
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

from anytree import NodeMixin
from anytree.iterators import PreOrderIter

from . import utils
from . import kernel as kernel_module

sys.setrecursionlimit(2020)


class BinaryTreeNode(NodeMixin):
    def __init__(
        self,
        name,
        tmat,
        population=0,
        q=0,
        feature=0,
        pop_thr=0.005,
        q_min=0.5,
        parent=None,
        left=None,
        right=None,
    ):
        """
        This class is used to plot dendrograms.

        prameters:
        ----------

        name (str): name of the node
        population (float): population of the node
        q (float): value at which the node is merged
        feature (float): some feature used for coloring
        parent: parent node
        left: left node
        right: right node
        """
        self._left = None
        self._right = None
        self._is_macrostate = None
        self._macrostates = None
        self._all_macrostates = None
        self._parent_macrostate = None
        self._assigned_macrostate = None

        self.name = name
        self.tmat = tmat
        self.n_states = int((self.tmat.shape[0] + 1) / 2)
        self.population = population  # Base population, used if the node is a leaf
        self.q = q
        self.feature = feature
        self.pop_thr = pop_thr
        self.q_min = q_min
        self.parent = parent
        self.left = left
        self.right = right

        self._x_origin = None
        self._x_target = None
        self._y_origin = None

        self._bins = None
        self._feature_norm = None
        self._colors = None

    def __repr__(self):
        return f"<Node of state {self.name}>"

    @property
    def population(self):
        """Population of state."""
        if self.is_leaf:
            return self._population
        else:
            return (self.left.population if self.left else 0) + (
                self.right.population if self.right else 0
            )

    @population.setter
    def population(self, value):
        if self.is_leaf:
            self._population = value
        else:
            return ValueError("population can only be set for microstates (leaves)")

    @property
    def q(self):
        """Q, e. g. self transition probability at which states were merged."""
        return self._q

    @q.setter
    def q(self, value):
        if 0 <= value <= 1:
            self._q = value
        else:
            raise ValueError("q must be 0 <= q <= 1")

    @property
    def feature(self):
        """
        Feature for states (e. g. fraction of native contacts). Is forwarded
        weighted by population
        """
        if self.is_leaf:
            return self._feature
        else:
            return (
                (self.left.feature * self.left.population if self.left else 0)
                + (self.right.feature * self.right.population if self.right else 0)
            ) / self.population

    @feature.setter
    def feature(self, value):
        if 0 <= value <= 1:
            self._feature = value
        else:
            raise ValueError("feature must be 0 <= feature <= 1")

    @property
    def left(self):
        return self._left

    @left.setter
    def left(self, node):
        if node is not None and node.parent is not None:
            raise ValueError("Node already has a parent")
        if self._left is not None:
            self._left.parent = None
        self._left = node
        if node is not None:
            node.parent = self

    @property
    def right(self):
        return self._right

    @right.setter
    def right(self, node):
        if node is not None and node.parent is not None:
            raise ValueError("Node already has a parent")
        if self._right is not None:
            self._right.parent = None
        self._right = node
        if node is not None:
            node.parent = self

    @property
    def children(self):
        """Return the two child nodes."""
        children = []
        if self.left is not None:
            children.append(self.left)
        if self.right is not None:
            children.append(self.right)
        return children

    @property
    def is_leaf(self):
        """Check if this node is leaf node."""
        return not (self.left or self.right)

    @property
    def is_macrostate(self):
        """Mark macrostates using this flag."""
        if self._is_macrostate is None:
            if (
                self.parent is not None
                and self.parent.q >= self.q_min
                and self.population >= self.root.population * self.pop_thr
                and self.siblings[0].population >= self.root.population * self.pop_thr
            ):
                self._is_macrostate = True
                self.siblings[0].is_macrostate = True
            elif self.parent is None:
                self._is_macrostate = True
            else:
                self.is_macrostate = False
        return self._is_macrostate

    @is_macrostate.setter
    def is_macrostate(self, value):
        if isinstance(value, bool):
            self._is_macrostate = value
        else:
            raise ValueError("is_macrostate must be boolean")

    @property
    def macrostates(self):
        """Returns all macrostate nodes."""
        if self._macrostates is None:
            true_macrostates = []
            for macrostate in self.all_macrostates:
                if len(macrostate.all_macrostates) == 1:
                    true_macrostates.append(macrostate)
            self._macrostates = tuple(true_macrostates)
        return self._macrostates

    @property
    def all_macrostates(self):
        """Returns all macrostate nodes."""
        if self._all_macrostates is None:
            self._all_macrostates = tuple(
                PreOrderIter(self, filter_=lambda node: node.is_macrostate)
            )
        return self._all_macrostates

    @property
    def parent_macrostate(self):
        """The parent_macrostate property."""
        if self._parent_macrostate is None:
            parent = self.parent
            while parent is not None and not parent.is_macrostate:
                parent = parent.parent
            self._parent_macrostate = parent
        return self._parent_macrostate

    @property
    def assigned_macrostate(self):
        """The assigned_macrostate property."""
        if self._assigned_macrostate is None:
            if self.is_leaf:
                if self.is_macrostate:
                    self._assigned_macrostate = self
                else:
                    if len(self.parent_macrostate.macrostates) == 1:
                        self._assigned_macrostate = self.parent_macrostate
                    else:
                        trans_probs = []
                        for m in self.parent_macrostate.macrostates:
                            macrostate = np.array(
                                [(s.name, s.population) for s in m.leaves]
                            )
                            indices = list(macrostate[:, 0])
                            indices.append(self.name)
                            indices.append(0)
                            tmp_tmat = self.tmat[np.ix_(indices, indices)].copy()
                            pops = list(macrostate[:, 1])
                            pops.append(self.population)
                            pops.append(0)
                            tmp_tmat, pops = utils.merge_states(
                                tmp_tmat,
                                list(range(macrostate.shape[0])),
                                -1,
                                np.array(pops),
                            )
                            trans_probs.append(tmp_tmat[-2, -1])
                        self._assigned_macrostate = self.parent_macrostate.macrostates[
                            np.argmax(trans_probs)
                        ]
            else:
                self._assigned_macrostate = None
        return self._assigned_macrostate

    @property
    def bins(self):
        """The bins property."""
        if self.is_root and self._bins is None:
            leaf_features = [leaf.feature for leaf in self.leaves]
            min_feature = min(leaf_features)
            max_feature = max(leaf_features)
            self._bins = np.linspace(min_feature, max_feature, 11)
        if self.is_root:
            return self._bins
        else:
            return self.root.bins

    @property
    def feature_norm(self):
        """The feature_norm property."""
        if self.is_root and self._feature_norm is None:
            self._feature_norm = Normalize(self.bins[0], self.bins[-1])
        if self.is_root:
            return self._feature_norm
        else:
            return self.root.feature_norm

    @property
    def colors(self):
        """The colors property."""
        if self.is_root and self._colors is None:
            cmap = plt.get_cmap("plasma_r", 10)
            self._colors = [cmap(idx) for idx in range(cmap.N)]
        if self.is_root:
            return self._colors
        else:
            return self.root.colors

    @property
    def color(self):
        """Color according to feature."""
        for color, rlower, rhigher in zip(
            self.colors, np.arange(0, 1, 0.1), np.arange(0.1, 1.1, 0.1)
        ):
            if rlower <= self.feature_norm(self.feature) <= rhigher:
                return color
        return "k"

    @property
    def edge_width(self):
        """Edge width from population."""
        return 6 * self.population / self.root.population

    @property
    def macrostate(self):
        """
        Macrostate this state belongs to. None if no macrostates are found
        above in tree.
        """
        node = self
        while not node.is_macrostate and node.parent:
            node = node.parent
        if node.is_macrostate:
            return node
        else:
            return None

    @property
    def x(self):
        """X coordinates for dandrogram for this node"""
        return np.array([self.x_origin, self.x_origin, self.x_target]) + 0.5

    @property
    def x_origin(self):
        """The x_origin property."""
        if not self.is_leaf:
            if not self._x_origin:
                self.x_origin = self.children[0].x_target
        return self._x_origin

    @x_origin.setter
    def x_origin(self, value):
        self._x_origin = value

    @property
    def x_target(self):
        """The x_target property."""
        if not self._x_target:
            if self.is_root:
                self.x_target = self.x_origin
            else:
                self.x_target = (self.x_origin + self.siblings[0].x_origin) / 2
        return self._x_target

    @x_target.setter
    def x_target(self, value):
        self._x_target = value

    @property
    def y(self):
        """Y coordinates for dandrogram for this node"""
        return np.array([self.y_origin, self.y_target, self.y_target])

    @property
    def y_origin(self):
        """The y_origin property."""
        if self.is_leaf:
            return 0
        else:
            if not self._y_origin:
                self.y_origin = self.children[0].y_target
            return self._y_origin

    @y_origin.setter
    def y_origin(self, value):
        self._y_origin = value

    @property
    def y_target(self):
        """The y_target property."""
        if self.parent:
            return self.parent.q
        else:
            return 1

    def plot(self, ax):
        for c in self.children:
            ax = c.plot(ax)
        # Remove this condition if root should be plotted as well.
        if not self.is_root:
            ax.plot(
                self.x,
                self.y,
                color=self.color,
                linewidth=self.edge_width if self.edge_width > 0.15 else 0.15,
            )
        return ax

    def plot_tree(self, ax):
        for i, leaf in enumerate(self.leaves):
            leaf.x_origin = i
        return self.plot(ax)


def cluster(
    tmat: NDArray[float],
    pop: NDArray[np.int_],
    kernel: Callable[
        [NDArray[float], NDArray[np.int_], NDArray[np.bool_]],
        [np.int_, np.int_, NDArray[np.bool_]],
    ] = kernel_module.LumpingKernel(),
    feature_kernel=None,
) -> (NDArray[float], NDArray[np.int_]):
    """
    cluster
    -------
    Perform full clustering for a transition matrix, given populations and a
    kernel.

    tmat (NDArray[float]): transition matrix, e. g. from
            mh.msm.estimate_markov_model
    pop (NDArray[float]): populations of microstates
    kernel: kernel object that determines the next merge

    returns Z (np.ndarray), full_pop (np.ndarray):
        The Z matrix holds the full merging of microstates:
            0: origin state
            1: target state
            2: distance between origin and target
            3: joint population
            i: Z[i, 0] and Z[i, 1] are combined to cluster n + i
            reference: scipy.cluster.hierarchy.linkage
        full_pop holds all state populations from state 0 to n + i
    """
    n = tmat.shape[0]

    full_tmat = np.zeros((2 * n - 1, 2 * n - 1), dtype=tmat.dtype.type)
    full_tmat[:n, :n] = tmat

    full_pop = np.zeros(2 * n - 1, dtype=pop.dtype.type)
    full_pop[:n] = pop

    if tmat.shape[0] < 2**7:
        states_type = np.uint8
    elif tmat.shape[0] < 2**15:
        states_type = np.uint16
    else:
        states_type = np.uint32

    # complete linkage
    full_states = np.zeros((2 * n - 1, 2), dtype=states_type)
    full_states[:n, 0] = np.arange(0, n)

    mask = np.full(2 * n - 1, False)
    mask[:n] = True

    # 0: state a
    # 1: state b
    # 2: distance between a and b
    # 3: population
    # i: Z[i, 0] and Z[i, 1] are combined to cluster n + i
    Z = np.zeros((n - 1, 4), dtype=np.float32)

    if feature_kernel:
        feature_kernel.reset()
    for i in range(n - 1):
        # Index of new state
        new_state = n + i

        # Use feature only for determination of target state
        if feature_kernel:
            # state, target_state, mask = kernel(feature_kernel * full_tmat, full_states, mask)
            state, target_state, mask = kernel(
                full_tmat, full_states, mask, feature_kernel
            )
            feature_kernel.update(state, target_state, new_state)
        else:
            state, target_state, mask = kernel(full_tmat, full_states, mask)

        metastability = full_tmat[state, state]
        # Merge states in transition matrix
        full_tmat, full_pop = utils.merge_states(
            full_tmat, [state, target_state], new_state, full_pop
        )

        # Update state linkage
        full_states[state, 1] = new_state
        full_states[target_state, 1] = new_state
        full_states[new_state:, 0] = new_state

        Z[i] = [state, target_state, metastability, full_pop[new_state]]

        # Update mask
        mask[new_state] = True
        mask[target_state] = False

    return Z, full_pop
