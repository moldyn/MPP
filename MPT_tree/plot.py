#!/usr/bin/env python3
"""
plot.py
==================

Plot dendrogram from linkage and qmin. Most of the code originates from
procss_mpp.py from Daniel Nagel.
"""

import numpy as np
import prettypyplot as pplt
from matplotlib import pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import to_hex, Normalize, LinearSegmentedColormap
from anytree import NodeMixin
from anytree.iterators import PreOrderIter
import core
import utils


__all__ = [
    "plot_dendrogram",
    "evaluate_stochastic_clustering",
    "plot_implied_timescales",
    "plot_heatmap",
    "plot_macro_feature",
]


### DENDROGRAM ###############################################################

class BinaryTreeNode(NodeMixin):
    def __init__(self, name, population=0, q=0, feature=0, parent=None, left=None, right=None, is_macrostate=False):
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
        self.name = name
        self._population = population  # Base population, used if the node is a leaf
        self._q = q
        self._feature = feature
        self._parent = parent
        self._left = left
        self._right = right
        self._is_macrostate = is_macrostate

        self._x_origin = None
        self._x_target = None
        self._y_origin = None

        if left:
            self.add_left(left)
        if right:
            self.add_right(right)

    @property
    def population(self):
        """Population of state."""
        if self.is_leaf:
            return self._population
        else:
            return (self.left.population if self.left else 0) \
                + (self.right.population if self.right else 0)
    @population.setter
    def population(self, value):
        if not self.is_leaf:
            pass
        elif 0 <= value <= 1:
            self._population = value
        else:
            # NOTE:
            # population currently is the count of frames in that state
            raise ValueError("population must be 0 <= population <= 1")

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
            return ((self.left.feature * self.left.population if self.left else 0) \
                + (self.right.feature * self.right.population if self.right else 0)) / self.population
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

    def add_left(self, node):
        self.left = node

    def add_right(self, node):
        self.right = node

    def add_node(self, node):
        if not self.left:
            self.left = node
        elif not self.right:
            self.right = node
        else:
            raise ValueError(f"{self.name} has already two nodes")

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
        return tuple(PreOrderIter(self, filter_=lambda node: node.is_macrostate))

    @property
    def color(self):
        """Color according to feature."""
        steps = 10
        cmap = plt.get_cmap('plasma_r', steps)
        colors = [cmap(idx) for idx in range(cmap.N)]

        bins = np.linspace(0, 1, steps + 1)
        for color, rlower, rhigher in zip(colors, bins[:-1], bins[1:]):
            if rlower <= self.feature <= rhigher:
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
        return np.array([self.x_origin, self.x_origin, self.x_target])

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
        if self._x_target:
            return self._x_target
        else:
            if not self.is_root:
                self.x_target = (self.x_origin + self.siblings[0].x_origin) / 2
            else:
                self.x_target = self.x_origin
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
            ax.plot(self.x, self.y, color=self.color, linewidth=self.edge_width if self.edge_width > 0.15 else 0.15)
        return ax

    def plot_tree(self, ax):
        for i, leaf in enumerate(self.leaves):
            leaf.x_origin = i
        return self.plot(ax)

def plot_tree(root, macrostate_assignment, output_file):
    """
    Plot the dendrogram from a given state tree of BinaryTreeNode.
    """
    n_states = len(root.leaves)

    # setup matplotlib
    pplt.use_style(figsize=2.6, figratio='golden', true_black=True)

    fig, (ax, ax_mat) = plt.subplots(
        2,
        1,
        gridspec_kw={
            'hspace': 0.05,
            'height_ratios': [9, 1],
        },
    )
    for key, spine in ax_mat.spines.items():
        spine.set_visible(False)

    ax = root.plot_tree(ax)

    ax.set_ylabel(r'metastability $Q_\text{min}$')
    ax.set_xlabel('microstates')
    ax.set_xlim(-0.005 * n_states, 1.005 * n_states)
    ax.set_ylim(0, 1.05)

    # plot legend
    cmap = plt.get_cmap('plasma_r', 10)
    bins = np.linspace(0, 1, 11)
    norm = Normalize(bins[0], bins[-1])
    label = r'$\langle Q \rangle_\text{state} $'

    cmappable = ScalarMappable(norm, cmap)
    plt.sca(ax)
    pplt.colorbar(cmappable, width='5%', label=label, position='top')

    # bring microstates in the right order
    macrostate_assignment = macrostate_assignment[:, [l.name for l in root.leaves]]

    #yticks = np.arange(0.5, 1.5 + len(root.macrostates))
    yticks = np.arange(0.5, 1.5 + macrostate_assignment.shape[0])
    xticks = np.arange(0, n_states + 1)
    cmap = LinearSegmentedColormap.from_list(
        'binary', [(0, 0, 0, 0), (0, 0, 0, 1)],
    )

    xvals = 0.5 * (xticks[:-1] + xticks[1:])
    for idx, assignment in enumerate(macrostate_assignment):
        xmean = np.median(xvals[assignment == 1])

        pplt.text(
            xmean,
            yticks[idx] - (yticks[1] - yticks[0]),
            f'{idx + 1:.0f}',
            ax=ax_mat,
            va='top',
            contour=True,
            size='small',
        )

    # Plot macrostate assignments
    ax_mat.pcolormesh(
        xticks,
        yticks,
        macrostate_assignment,
        snap=True,
        cmap=cmap,
        vmin=0,
        vmax=1,
    )
    # set x-labels
    ax_mat.set_yticks(yticks)
    ax_mat.set_yticklabels([])
    ax_mat.grid(visible=True, axis='y', ls='-', lw=0.5)
    ax_mat.tick_params(axis='y', length=0, width=0)
    ax_mat.set_xlim(ax.get_xlim())
    ax.set_xlabel('')
    ax_mat.set_xlabel('macrostates')
    ax_mat.set_ylabel('')
    fig.align_ylabels([ax, ax_mat])

    ax_mat.set_xticks(np.arange(0.5, 0.5 + n_states))

    # Hide microstate labels
    if True:
        for axes in (ax, ax_mat):
            axes.set_xticks([])
            axes.set_xticks([], minor=True)
            axes.set_xticklabels([])
            axes.set_xticklabels([], minor=True)

    pplt.savefig(output_file)

def build_tree(Z, full_pop):
    """
    Build tree of BinaryTreeNode from a given Z matrix and the corresponding
    populations.
    """
    n = Z.shape[0]  + 1
    nodes = {}
    for i, (state, target_state, q, pop) in enumerate(Z):
        state = int(state)
        target_state = int(target_state)
        if state not in nodes:
            nodes[state] = BinaryTreeNode(state, population=full_pop[state], q=q)
        if target_state not in nodes:
            nodes[target_state] = BinaryTreeNode(target_state, population=full_pop[target_state], q=q)
        # if n + i not in nodes:
        nodes[n + i] = BinaryTreeNode(n + i, q=q)
        nodes[n + i].left = nodes[state]
        nodes[n + i].right = nodes[target_state]
    return nodes[n + i]

# def define_macrostates(root, n):
#     macrostates = {root}
#     bit_vector = 0
#     for i in range(n-1):
#         macrostates_list = list(macrostates)
#         j = 0
#         node_to_split = macrostates_list[j]
#         while node_to_split.is_leaf:
#             j += 1
#             node_to_split = macrostates_list[j]
#         for node in macrostates_list[1:]:
#             if node.q > node_to_split.q and not node.is_leaf:
#                 node_to_split = node
#         macrostates.update({node_to_split.left, node_to_split.right})
#         macrostates -= {node_to_split}
#
#     for node in list(macrostates):
#         node.is_macrostate = True
#
#     return root

def add_feature(traj, feature_traj, root):
    """
    Add a feature (e. g. fraction of native contacts) to the leaves of a tree.
    """
    for leave in root.leaves:
        leave.feature = feature_traj[traj == leave.name+1].mean()
    return root

# def plot_mpt(mpt, out, n_i = 0):
#     """
#     Plot a clustering from MPT object.
#     """
#     root = build_tree(mpt.Z[n_i], mpt.full_pop[n_i])
#     add_feature(mpt.traj, mpt.feature_traj, root)
#     plot_tree(root, mpt.macrostate_assignment, out)
 
def plot_dendrogram(Z, full_pop, traj, feature_traj, ma, out):
    """
    Plot dendrogram from Z matrix, full_pop, trajectory, feature_traj and
    macrostate_assignment.
    """
    root = build_tree(Z, full_pop)
    add_feature(traj, feature_traj, root)
    pop_thr = 0.005
    q_min = 0.5
#    macrostate_assignment = core.assign_macrostates(Z, full_pop, pop_thr, q_min)
    plot_tree(root, ma, out)


### SIMILARITY ###############################################################

def evaluate_stochastic_clustering(mpt1, mpt2, out):
    """
    Plot similarity values for a reference and a stochastic clustering.
    """
    ref, sto, S = mpt1 + mpt2
    s1, s2, s3 = S
    n_states = S.shape[1]
    x, y = utils.get_grid_format(n_states)
    fig, axs = plt.subplots(y, x, figsize=(2*x, 2*y))
    for state, ax in enumerate(axs.flatten()[:n_states]):
        m = 0
        # Set left limit to minimum instead of 0
        if True:
            m = min([min(s1[state]), min(s2[state]), min(s3[state])]) - 0.02

        ax.hist(s1[state], bins=np.linspace(m, 1, 21), color="g", alpha=0.7)
        ax.hist(s2[state], bins=np.linspace(m, 1, 21), color="b", alpha=0.7)
        ax.hist(s3[state], bins=np.linspace(m, 1, 21), color="r", alpha=0.7)
        ax.set_title(f"state {state+1} ({ref.macrostate_assignment[0][state].sum()})")
    fig.supxlabel("Macrostate similarity")
    fig.supylabel(f"Count of clusterings ({sto.n_runs} clusterings)")
    leg = plt.figlegend(["union", "reference", "clustering"], ncols=3, loc='lower center', bbox_to_anchor=(0.5, 0.05))
    plt.tight_layout(rect=(0, 0.04, 1, 1))
    plt.savefig(out)


### IMPLIED TIMESCALES #######################################################

def plot_implied_timescales(trajs, lagtimes, out, titles="", frame_length=0.2, first_ref=False):
    """
    frame_length in ns
    """
    if first_ref:
        ref_traj = trajs.pop(0)
    x, y = utils.get_grid_format(len(trajs))
    pplt.use_style(figsize=(3*x, 2*y), latex=False, colors='pastel_autumn')
    fig, axs = plt.subplots(y, x, sharex=True, sharey=True)
    if not isinstance(axs, np.ndarray):
        axs = np.array([axs])

    if titles != "":
        titles = titles
    else:
        titles = [""] * len(trajs)

    min_it = None
    max_it = None
    
    if first_ref:
        it_ref = mh.msm.implied_timescales(ref_traj, lagtimes, ntimescales=3)
        # change from frames to ns
        it_ref *= frame_length
        min_it = it_ref.min()
        max_it = it_ref.max()

    lagtimes_ns = lagtimes * frame_length
    for ax, traj, title in zip(axs.flatten(), trajs, titles):
        #ax.yaxis.set_major_formatter(mtick.ScalarFormatter(useMathText=True))
        ax.yaxis.set_major_formatter(mtick.LogFormatterSciNotation)
        it = mh.msm.implied_timescales(traj, lagtimes, ntimescales=3)
        # change from frames to ns
        it *= frame_length
        if min_it == None:
            min_it = it.min()
        else:
            min_it = min(it.min(), min_it)
        if max_it == None:
            max_it = it.max()
        else:
            max_it = max(it.max(), max_it)

        if first_ref:
            _plot_impl_times(it_ref, lagtimes_ns, ax, ls=":")
        _plot_impl_times(it, lagtimes_ns, ax)
        ax.set_yscale("log")
        ax.set_title(title)

    for ax in axs.flatten():
        ax.set_ylim(min(min_it * 0.9, int(lagtimes_ns.shape[0] / 4)), max_it * 1.1)

    if len(axs.shape) == 2:
        for ax in axs[-1]:
            ax.set_xlabel(r'lag time $\tau$ / ns')
        for axx in axs:
            for ax in axx[1:]:
                plt.setp(ax.get_yticklabels(), visible=False)
        for ax in axs[:, 0]:
            ax.set_ylabel('time scale / ns')
    elif len(axs.shape) == 1:
        axs[0].set_ylabel('time scale / ns')
        for ax in axs:
            ax.set_xlabel(r'lag time $\tau$ / ns')
        for ax in axs[1:]:
            plt.setp(ax.get_yticklabels(), visible=False)

    plt.tight_layout()
    plt.savefig(out)

def _plot_impl_times(impl_times, lagtimes, ax, ls="-"):
    """Plot the implied timescales"""
    #colors = ['pplt:blue', 'pplt:red', 'pplt:green']
    colors = ['#264653', '#2A9D8F', '#E9C46A']
    for idx, impl_time in enumerate(impl_times.T):
        if ls == ":":
            label = f'$t_{{ref,{idx + 1}}}$'
        else:
            label = f'$t_{idx + 1}$'
        ax.plot(lagtimes, impl_time, label=label, color=colors[idx], ls=ls)

    xlim = lagtimes[0], lagtimes[-1]
    ref_low = int(lagtimes.shape[0] / 4)
    ax.set_xlim(xlim)
    # highlight diagonal
    #x_i = np.arange(max(xlim[0], 1), xlim[1])
    x_i = np.arange(ref_low, xlim[1])
    ax.fill_between(x_i, x_i, color='pplt:grid')
    pplt.legend(outside='right', frameon=False)


### SIMILARITY MATRIX ########################################################

def plot_heatmap(a, out, title=""):
    """
    Plot heatmap from a matrix. This is supposed for a similarity matrix as
    returned from the multiplication of two MPT objects.
    """
    fig, ax = plt.subplots()
    ax.imshow(a, norm="log")
    ax.set_aspect('equal', 'box')

    ax.set_xticks(np.arange(a.shape[1]))
    ax.set_yticks(np.arange(a.shape[0]))
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            #text = ax.text(j, i, f"{a[i, j]*100:.1f}",
            text = ax.text(j, i, f"{a[i, j]:.0f}",
                       ha="center", va="center", color="w")
    if title:
        ax.set_title(title)
    ax.set_xlabel("Macrostates A")
    ax.set_ylabel("Macrostates B")
    plt.tight_layout()
    plt.savefig(out)


### MACROSTATE FEATURES ######################################################

def plot_macro_feature(micro_feature, out, ref=None, pop=None):
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
    min_feature = micro_feature.min() * 0.95
    max_feature = micro_feature.max() * 1.05
    counts, bins = np.histogram(
        micro_feature,
        bins=np.linspace(min_feature, max_feature, 101),
        weights=pop,
        density=True,
    )
    norm_counts = counts / micro_feature.shape[1]
    y_min = norm_counts[norm_counts > 0].min() * 0.7

    fig, ax = plt.subplots()
    ax.hist(bins[:-1], bins=bins, weights=norm_counts, label="Stochastic Clustering")
    if ref != None:
        for mas, mfs, c, l, w in ref:
            add_ref(mas, mfs, ax, color=c, label=l, weights=w)
    ax.set_xlabel("Fraction of native contacts")
    ax.set_ylabel("Population")
    ax.set_title(f"Macrostate Features, {micro_feature.shape[1]} runs")
    ax.set_yscale("log")
    ylim = ax.get_ylim()
    ax.set_ylim((y_min, ylim[1]))
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(out)

def add_ref(macrostate_assignment, macrostate_feature, ax, color="r", label="Reference", weights=None):
    """
    Add a clustering to the histogram.

    macrostate_assignment (np.ndarray, MxN): macrostate assignement, M: number
            of macrostates, N: number of microstates.
    macrostate_feature (np.ndarray, M): mean feature for every macrostate.
    """
    b = True
    for i, (ma, mf) in enumerate(zip(macrostate_assignment, macrostate_feature)):
        x = [mf, mf]
        if weights is None:
            weights = 1
        else:
            weights = weights / weights.sum()
        y = [1e-9, (ma * weights).sum() / weights.sum() * 1e-3]
        if b:
            ax.plot(x, y, c=color, label=label + " / 1000")
            b = False
        else:
            ax.plot(x, y, c=color)
#        ax.text(mf + 0.005, ma.sum() * 0.9, f"{i:.0f}", c=color, backgroundcolor="w")
        pplt.text(
            mf + 0.015,
            y[1] * 0.82,
            f"{i + 1:.0f}",
            c=color,
            ax=ax,
            contour=True,
            size='small',
        )
