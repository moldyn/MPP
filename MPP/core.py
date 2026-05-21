"""
core.py
=======

Core functions for MPP
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
    """
    A node in the MPP lumping tree.

    Each node represents either a microstate (leaf) or a merged cluster
    (internal node) produced during the MPP lumping procedure. The tree is
    built bottom-up by the ``cluster`` function and parsed top-down to
    identify macrostates.

    Inherits from ``anytree.NodeMixin``, which provides tree-traversal
    utilities such as ``leaves``, ``root``, ``siblings``, ``is_root``, and
    ``PreOrderIter``.

    Parameters
    ----------
    name : int or str
        Identifier for this node (microstate index for leaves, composite
        index for internal nodes).
    tmat : NDArray[float], shape (2n-1, 2n-1)
        Full transition matrix extended to hold merged states. ``n`` is
        the number of original microstates. Shared across nodes of the
        same tree.
    population : float, optional
        Population of this microstate. For leaf nodes this value is stored
        directly; for internal nodes it is derived from children. Default 0.
    q : float, optional
        Metastability (self-transition probability) at which this node was
        merged into its parent. Must satisfy ``0 <= q <= 1``. Default 0.
    feature : float, optional
        Feature value for this microstate (e.g. fraction of native contacts),
        in ``[0, 1]``. For leaf nodes this value is stored directly; for
        internal nodes the population-weighted mean of children is returned.
        Default 0.
    pop_thr : float, optional
        Minimum population fraction required for a node to be classified as
        a macrostate. Default 0.005.
    q_min : float, optional
        Minimum metastability of the parent merge required for a node to be
        classified as a macrostate. Default 0.5.
    parent : BinaryTreeNode, optional
        Parent node in the tree. Default None (root).
    left : BinaryTreeNode, optional
        Left child node. Default None (leaf).
    right : BinaryTreeNode, optional
        Right child node. Default None (leaf).
    """

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
        """Return a string representation identifying the node by state name."""
        return f"<Node of state {self.name}>"

    @property
    def population(self):
        """
        Population of this node.

        For leaf nodes (microstates), returns the stored population value.
        For internal nodes (merged clusters), returns the sum of the left
        and right children's populations.

        Returns
        -------
        float
            Population of this node.
        """
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
            raise ValueError("population can only be set for microstates (leaves)")

    @property
    def q(self):
        """
        Metastability at which this node was merged into its parent.

        Corresponds to the self-transition probability of this node at the
        lumping step that produced its parent. Must satisfy ``0 <= q <= 1``.

        Returns
        -------
        float
            Metastability value in ``[0, 1]``.
        """
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
        Feature value for this node, weighted by population.

        For leaf nodes (microstates), returns the stored feature value.
        For internal nodes (merged clusters), returns the population-weighted
        mean of the left and right children's feature values.

        Returns
        -------
        float
            Feature value in ``[0, 1]``.
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
        """
        Left child node, or ``None`` if this node is a leaf.

        Setting this property updates the ``parent`` reference of the child
        node accordingly. Raises ``ValueError`` if the assigned node already
        has a parent.

        Returns
        -------
        BinaryTreeNode or None
            Left child node.
        """
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
        """
        Right child node, or ``None`` if this node is a leaf.

        Setting this property updates the ``parent`` reference of the child
        node accordingly. Raises ``ValueError`` if the assigned node already
        has a parent.

        Returns
        -------
        BinaryTreeNode or None
            Right child node.
        """
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
        """
        List of child nodes, in left-then-right order.

        Returns a list containing the left child, the right child, or both,
        omitting any that are ``None``.

        Returns
        -------
        list of BinaryTreeNode
            Non-None children of this node.
        """
        children = []
        if self.left is not None:
            children.append(self.left)
        if self.right is not None:
            children.append(self.right)
        return children

    @property
    def is_leaf(self):
        """
        Whether this node is a leaf (microstate) in the lumping tree.

        Returns
        -------
        bool
            ``True`` if this node has no children, ``False`` otherwise.
        """
        return not (self.left or self.right)

    @property
    def is_macrostate(self):
        """
        Whether this node qualifies as a macrostate.

        A node is classified as a macrostate when all of the following hold:

        * Its parent exists and has ``q >= q_min``.
        * Its own population is at least ``pop_thr`` times the root population.
        * Its sibling's population is at least ``pop_thr`` times the root
          population.

        The root node is always classified as a macrostate. Setting
        ``is_macrostate`` on a node also sets it on the sibling node.

        Returns
        -------
        bool
            ``True`` if this node is a macrostate, ``False`` otherwise.
        """
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
        """
        Terminal macrostate nodes in the subtree rooted at this node.

        Returns only those macrostate nodes that contain no further
        macrostates in their own subtrees — i.e., the leaf macrostates of
        the macrostate hierarchy. These represent the final macrostate
        assignment targets.

        See also ``all_macrostates`` for the full set of macrostate nodes
        including intermediate ones.

        Returns
        -------
        tuple of BinaryTreeNode
            Terminal macrostate nodes.
        """
        if self._macrostates is None:
            true_macrostates = []
            for macrostate in self.all_macrostates:
                if len(macrostate.all_macrostates) == 1:
                    true_macrostates.append(macrostate)
            self._macrostates = tuple(true_macrostates)
        return self._macrostates

    @property
    def all_macrostates(self):
        """
        All macrostate nodes in the subtree rooted at this node.

        Traverses the subtree in pre-order and collects every node for which
        ``is_macrostate`` is ``True``, including intermediate macrostate nodes.

        See also ``macrostates`` for only the terminal (leaf) macrostates.

        Returns
        -------
        tuple of BinaryTreeNode
            All macrostate nodes in pre-order.
        """
        if self._all_macrostates is None:
            self._all_macrostates = tuple(
                PreOrderIter(self, filter_=lambda node: node.is_macrostate)
            )
        return self._all_macrostates

    @property
    def parent_macrostate(self):
        """
        Nearest ancestor node that is classified as a macrostate.

        Walks up the tree from this node's parent until a macrostate node
        is found, or returns ``None`` if no such ancestor exists.

        Returns
        -------
        BinaryTreeNode or None
            Nearest macrostate ancestor, or ``None`` if none exists.
        """
        if self._parent_macrostate is None:
            parent = self.parent
            while parent is not None and not parent.is_macrostate:
                parent = parent.parent
            self._parent_macrostate = parent
        return self._parent_macrostate

    @property
    def assigned_macrostate(self):
        """
        Macrostate assignment for this leaf node.

        Only defined for leaf nodes (microstates); returns ``None`` for
        internal nodes.

        Assignment logic for leaf nodes:

        * If the leaf is itself a macrostate, returns ``self``.
        * If the nearest ancestor macrostate (``parent_macrostate``) contains
          only one terminal macrostate, assigns to that macrostate.
        * Otherwise, assigns to the terminal macrostate with the highest
          transition probability from this leaf, computed by temporarily
          merging each macrostate's microstates with this leaf state.

        Returns
        -------
        BinaryTreeNode or None
            The assigned terminal macrostate node for leaf nodes, or ``None``
            for internal nodes.
        """
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
        """
        Feature bin edges used for color mapping in dendrogram plots.

        Computed once at the root as 11 equally spaced values spanning the
        range of leaf feature values, producing 10 bins. Non-root nodes
        delegate to the root.

        Returns
        -------
        NDArray[float], shape (11,)
            Bin edges from the minimum to maximum leaf feature value.
        """
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
        """
        Normalization object for mapping feature values to ``[0, 1]``.

        Computed once at the root using the minimum and maximum bin edges
        from ``bins``. Non-root nodes delegate to the root.

        Returns
        -------
        matplotlib.colors.Normalize
            Normalizer mapping feature values to ``[0, 1]``.
        """
        if self.is_root and self._feature_norm is None:
            self._feature_norm = Normalize(self.bins[0], self.bins[-1])
        if self.is_root:
            return self._feature_norm
        else:
            return self.root.feature_norm

    @property
    def colors(self):
        """
        List of 10 colors from the ``plasma_r`` colormap for feature bins.

        Computed once at the root. Non-root nodes delegate to the root.

        Returns
        -------
        list of tuple
            Ten RGBA color tuples, one per feature bin.
        """
        if self.is_root and self._colors is None:
            cmap = plt.get_cmap("plasma_r", 10)
            self._colors = [cmap(idx) for idx in range(cmap.N)]
        if self.is_root:
            return self._colors
        else:
            return self.root.colors

    @property
    def color(self):
        """
        RGBA color for this node based on its feature value.

        Looks up the bin that contains the normalized feature value and
        returns the corresponding color from ``colors``. Returns black
        (``"k"``) if the feature value falls outside all bins.

        Returns
        -------
        tuple or str
            RGBA color tuple, or ``"k"`` if outside the feature range.
        """
        for color, rlower, rhigher in zip(
            self.colors, np.arange(0, 1, 0.1), np.arange(0.1, 1.1, 0.1)
        ):
            if rlower <= self.feature_norm(self.feature) <= rhigher:
                return color
        return "k"

    @property
    def edge_width(self):
        """
        Line width for dendrogram edges, scaled by population.

        Proportional to this node's population relative to the root's
        total population, scaled to a maximum of 6.

        Returns
        -------
        float
            Edge width for use in ``matplotlib`` plot calls.
        """
        return 6 * self.population / self.root.population

    @property
    def macrostate(self):
        """
        Nearest macrostate ancestor at or above this node.

        Walks up the tree from this node until a macrostate node is found.
        Returns ``None`` if no macrostate exists at or above this node.

        Returns
        -------
        BinaryTreeNode or None
            The nearest macrostate node, or ``None`` if none is found.
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
        """
        X coordinates for plotting this node's dendrogram segment.

        Returns an array of three x values: ``[x_origin, x_origin, x_target]``,
        offset by 0.5 for visual centering.

        Returns
        -------
        NDArray[float], shape (3,)
            X coordinates for the dendrogram line segment.
        """
        return np.array([self.x_origin, self.x_origin, self.x_target]) + 0.5

    @property
    def x_origin(self):
        """
        X position of this node in the dendrogram (vertical axis position).

        For internal nodes, derived from the left child's ``x_target``.
        For leaf nodes, set externally by ``plot_tree``.

        Returns
        -------
        float or None
            X origin coordinate, or ``None`` if not yet assigned.
        """
        if not self.is_leaf:
            if not self._x_origin:
                self.x_origin = self.children[0].x_target
        return self._x_origin

    @x_origin.setter
    def x_origin(self, value):
        self._x_origin = value

    @property
    def x_target(self):
        """
        X position of the merge point connecting this node to its sibling.

        For the root, equals ``x_origin``. For other nodes, computed as the
        midpoint between this node's ``x_origin`` and its sibling's
        ``x_origin``.

        Returns
        -------
        float or None
            X target coordinate, or ``None`` if not yet computed.
        """
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
        """
        Y coordinates for plotting this node's dendrogram segment.

        Returns an array of three y values: ``[y_origin, y_target, y_target]``,
        representing the vertical rise from this node to its parent merge.

        Returns
        -------
        NDArray[float], shape (3,)
            Y coordinates for the dendrogram line segment.
        """
        return np.array([self.y_origin, self.y_target, self.y_target])

    @property
    def y_origin(self):
        """
        Y position of this node in the dendrogram (metastability axis).

        For leaf nodes (microstates), always 0. For internal nodes,
        derived from the left child's ``y_target``.

        Returns
        -------
        float
            Y origin coordinate.
        """
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
        """
        Y position of the merge point connecting this node to its parent.

        Equals the parent's metastability ``q``. For the root (no parent),
        returns 1.

        Returns
        -------
        float
            Y target coordinate (parent's ``q``, or 1.0 for the root).
        """
        if self.parent:
            return self.parent.q
        else:
            return 1

    def plot(self, ax):
        """
        Recursively plot dendrogram line segments for this subtree.

        Plots the line segment for each non-root node using its ``x``, ``y``,
        ``color``, and ``edge_width`` properties. Children are plotted before
        this node (post-order traversal).

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Axes object on which to draw the dendrogram segments.

        Returns
        -------
        matplotlib.axes.Axes
            The same ``ax`` passed in, with segments added.
        """
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
        """
        Assign leaf x positions and plot the full dendrogram.

        Assigns sequential x positions (0, 1, 2, ...) to all leaf nodes
        in left-to-right order, then delegates to ``plot`` to draw the
        full dendrogram recursively.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Axes object on which to draw the dendrogram.

        Returns
        -------
        matplotlib.axes.Axes
            The same ``ax`` passed in, with the full dendrogram drawn.
        """
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
    Perform full lumping for a transition matrix, given populations and a kernel.

    Iteratively merges microstates according to the lumping kernel until a
    single cluster remains. At each step, the kernel selects the microstate
    to merge and its target, the transition matrix is updated via
    ``utils.merge_states``, and the merge is recorded in the Z matrix.

    Parameters
    ----------
    tmat : NDArray[float], shape (n, n)
        Transition matrix of the microstate model.
    pop : NDArray[int], shape (n,)
        Populations of the ``n`` microstates.
    kernel : callable, optional
        Lumping kernel that selects the next merge. Must accept
        ``(full_tmat, full_states, mask)`` (and optionally a
        ``feature_kernel``) and return ``(state, target_state, mask)``.
        Defaults to ``LumpingKernel()``.
    feature_kernel : FeatureKernel, optional
        Optional feature kernel for geometric similarity. When provided,
        it is passed to the lumping kernel and updated after each merge.
        Default ``None``.

    Returns
    -------
    Z : NDArray[float], shape (n-1, 4)
        Z matrix recording all merges in scipy linkage format. Each row
        contains ``[state_a, state_b, metastability_a, joint_population]``.
        The merged state for row ``i`` is assigned index ``n + i``.
        See also ``scipy.cluster.hierarchy.linkage``.
    full_pop : NDArray[int], shape (2n-1,)
        Populations of all states from index 0 to ``2n-2``, including
        original microstates and all intermediate merged states.
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
