#!/usr/bin/env python3
"""
plot.py
==================

Plot dendrogram from linkage and qmin. Most of the code originates from
procss_mpp.py from Daniel Nagel.
"""

import os
from os.path import splitext
import itertools

import numpy as np
import pandas as pd
import prettypyplot as pplt
from matplotlib import pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import to_hex, Normalize, LinearSegmentedColormap, LogNorm, ListedColormap
from matplotlib import colors
from matplotlib.cbook import boxplot_stats
import matplotlib.patches as patches
from anytree import NodeMixin
from anytree.iterators import PreOrderIter
import networkx as nx
import msmhelper as mh
from msmhelper._cli.contact_rep import load_clusters
import MPT.core as core
import MPT.utils as utils


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
        feature_norm = np.linalg.norm(self.feature)

        bins = np.linspace(0, 1, steps + 1)
        for color, rlower, rhigher in zip(colors, bins[:-1], bins[1:]):
            if rlower <= feature_norm <= rhigher:
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
        nodes[n + i] = BinaryTreeNode(n + i, q=q)
        nodes[n + i].left = nodes[state]
        nodes[n + i].right = nodes[target_state]
    return nodes[n + i]

def add_feature(traj, feature_traj, root):
    """
    Add a feature (e. g. fraction of native contacts) to the leaves of a tree.
    """
    if len(feature_traj.shape) == 2:
        state_features = np.array([feature_traj[traj == i+1].mean(axis=0) for i in range(len(root.leaves))])
        norm_features = np.linalg.norm(state_features, axis=1)
        sn = norm_features - norm_features.min()
        norm_feature = 1 - sn / sn.max()
        for leave in root.leaves:
            leave.feature = norm_feature[leave.name]
        return root
    elif len(feature_traj.shape) == 1:
        for leave in root.leaves:
            leave.feature = feature_traj[traj == leave.name+1].mean()
        return root
    else:
        raise ValueError("feature_traj must be 1 D or 2 D.")
 
def plot_dendrogram(Z, full_pop, traj, feature_traj, ma, out):
    """
    Plot dendrogram from Z matrix, full_pop, trajectory, feature_traj and
    macrostate_assignment.
    """
    root = build_tree(Z, full_pop)
    add_feature(traj, feature_traj, root)
    pop_thr = 0.005
    q_min = 0.5
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


def plot_relative_implied_timescales(cl, ref, out):
    if cl.timescales == None:
        cl.calc_timescales()
    if ref.timescales == None:
        ref.calc_timescales()
    
    its = cl.timescales / ref.timescales

    fig, axs = plt.subplots(1, 4, figsize=(8, 2.5), sharey=True)
    for i, ax in enumerate(axs[:-1]):
        ax.hist(its[:, i], bins=20)
        ax.set_title(f'its {i+1}')
    axs[-1].hist(its.mean(axis=1), bins=20)
    axs[-1].set_title(f'Mean its {1}-{i+1}')

    # fig.supxlabel(r"Implied Timescale Similarity $\left(\frac{t_\mathrm{stoch}}{t_\mathrm{det}}\right)$")
    fig.supxlabel(r"Relative Implied Timescale $\left(\frac{t_\mathrm{stoch}}{t_\mathrm{det}}\right)$")
    fig.supylabel('Count of Clusterings')
#    fig.suptitle(r"Relative Implied Timescale Similarity $\left(\frac{t_\mathrm{stoch}}{t_\mathrm{det}}\right)$")
    plt.tight_layout()
    plt.savefig(out)


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
    ax.set_xlabel("Macrostate")
    ax.set_ylabel("Macrostate")
    plt.tight_layout()
    plt.savefig(out)

def plot_tmat(a, out, title="Transition Matrix", color_thr=0.01):
    """
    Plot heatmap from a matrix. This is supposed for a similarity matrix as
    returned from the multiplication of two MPT objects.
    """
    # Scale a to percent
    a = a * 100

    # Define the colormap for the diagonal elements (logarithmic Reds)
    diagonal_values = np.diag(a)
    diag_norm = LogNorm(vmin=diagonal_values.min(), vmax=diagonal_values.max())
    diag_cmap = plt.cm.Reds

    # Adjust the Reds colormap to make the lower bound closer to red
    reds_custom = diag_cmap(np.linspace(0.2, 1, 256))
    diag_cmap_custom = ListedColormap(reds_custom)

    # Define the colormap for the off-diagonal elements (logarithmic viridis)
    off_diag_mask = ~np.eye(a.shape[0], dtype=bool)
    off_diag_values = a[off_diag_mask]
    
    # Threshold for light gray
    threshold = color_thr * off_diag_values.max()
    print(f"Threshold for probabilities: {threshold:.3f} %")

    off_diag_norm = LogNorm(vmin=threshold*(1-color_thr), vmax=off_diag_values.max())
    off_diag_cmap = plt.cm.viridis

    # Create a custom colormap for off-diagonal values including light gray
    colors_list = plt.cm.viridis(np.linspace(0, 1, 256))
    gray = np.array([0.9, 0.9, 0.9, 1.0])
    #colors_list[:int(threshold / off_diag_values.max() * 256)] = gray
    colors_list[:int(color_thr * 256)] = gray
    custom_off_diag_cmap = colors.ListedColormap(colors_list)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect('equal', 'box')
    ax.grid(False)
    
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            value = a[i, j]
            if value == 0:
                color = (1, 1, 1, 1)  # Zero probabilities are white
            elif i == j:
                color = diag_cmap_custom(diag_norm(value))
            else:
                color = gray if value < threshold else custom_off_diag_cmap(off_diag_norm(value))
        
            ax.add_patch(patches.Rectangle((j - 0.5, i - 0.5), 1, 1, color=color))
        
            # Add text with transition probabilities
            # print(np.array(color[:3]))
            if value != 0:
                grayscale = np.sum(np.array(color[:3]) * np.array([0.299, 0.587, 0.114]))
                text_color = 'white' if grayscale < 0.5 else 'black'
                ax.text(j, i, f"{value:.2f}%", ha='center', va='center', color=text_color, fontsize=10)

    ax.set_xticks(np.arange(a.shape[1]))
    ax.set_yticks(np.arange(a.shape[0]))
    ax.set_xticklabels(np.arange(1, a.shape[1] + 1))
    ax.set_yticklabels(np.arange(1, a.shape[0] + 1))
    ax.set_xlim(-0.5, a.shape[1]-0.5)
    ax.set_ylim(-0.5, a.shape[0]-0.5)

    # Add a colorbar for diagonal values
    cbar_diag = fig.colorbar(plt.cm.ScalarMappable(norm=diag_norm, cmap=diag_cmap), ax=ax, shrink=0.5)
    cbar_diag.set_label('Self Transition Probabilities / \\%')

    # Add a colorbar for off-diagonal values
    cbar_off_diag = fig.colorbar(plt.cm.ScalarMappable(norm=off_diag_norm, cmap=custom_off_diag_cmap), ax=ax, shrink=0.5)
    cbar_off_diag.set_label('Transitiion Probabilities / \\%')

    if title:
        ax.set_title(title)

    ax.set_xlabel("From Macrostate")
    ax.set_ylabel("To Macrostate")
    plt.tight_layout()
    plt.savefig(out)

def plot_trans_time(
        a,
        out,
        tlag=50.0,
        frame_length=0.2,
        title=r"Transition Times $\frac{t_\mathrm{lag}}{P}$",
        color_thr=0.01,
    ):
    """
    Plot heatmap from a matrix. This is supposed for a similarity matrix as
    returned from the multiplication of two MPT objects.
    frame_length in ns
    """
    with np.errstate(divide='ignore'):
        a = tlag / a * frame_length
    #a = (np.full(a.shape, tlag) / a) * frame_length

    # Define the colormap for the diagonal elements (logarithmic Reds)
    diagonal_values = np.diag(a)
    diag_norm = LogNorm(vmin=diagonal_values.min(), vmax=diagonal_values.max())
    diag_cmap = plt.cm.Reds_r

    # Adjust the Reds colormap to make the lower bound closer to red
    reds_custom = diag_cmap(np.linspace(0, 0.8, 256))
    diag_cmap_custom = ListedColormap(reds_custom)

    # Define the colormap for the off-diagonal elements (logarithmic viridis)
    off_diag_mask = ~np.eye(a.shape[0], dtype=bool)
    off_diag_values = a[off_diag_mask]
    off_diag_values_non_inf = off_diag_values[off_diag_values < np.inf]

    # Threshold for light gray
    threshold = off_diag_values.min() / color_thr
    print(f"Threshold for probabilities: {threshold:.2f} ns")

    #off_diag_norm = LogNorm(vmin=off_diag_values.min(), vmax=off_diag_values_non_inf.max())
    off_diag_norm = LogNorm(vmin=off_diag_values.min(), vmax=threshold/(1-color_thr))
    off_diag_cmap = plt.cm.viridis_r

    # Create a custom colormap for off-diagonal values including light gray
    colors_list = plt.cm.viridis_r(np.linspace(0, 1, 256))
    gray = np.array([0.9, 0.9, 0.9, 1.0])
#    colors_list[:int(threshold / off_diag_values.max() * 256)] = gray
#    colors_list[int(threshold / off_diag_values_non_inf.min() * 256):] = gray
    colors_list[int((1-color_thr) * 256):] = gray
    custom_off_diag_cmap = colors.ListedColormap(colors_list)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect('equal', 'box')
    ax.grid(False)
    
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            value = a[i, j]
            if value == np.inf:
                color = (1, 1, 1, 1) # Zero probabilities are white
            elif i == j:
                color = diag_cmap_custom(diag_norm(value))
            else:
                color = gray if value > threshold else custom_off_diag_cmap(off_diag_norm(value))
        
            ax.add_patch(patches.Rectangle((j - 0.5, i - 0.5), 1, 1, color=color))
        
            # Add text with transition probabilities
            # print(np.array(color[:3]))
            if value != np.inf:
                grayscale = np.sum(np.array(color[:3]) * np.array([0.299, 0.587, 0.114]))
                text_color = 'white' if grayscale < 0.5 else 'black'
                if value >= threshold:
                    pre_text = f"{value:.1g}"
                    text = pre_text[:2] + pre_text[-1]
                else:
                    if value >= 100:
                        text = f"{value:.0f}"
                    else:
                        text = f"{value:#.3g}"
                ax.text(j, i, text, ha='center', va='center', color=text_color, fontsize=10)

    ax.set_xticks(np.arange(a.shape[1]))
    ax.set_yticks(np.arange(a.shape[0]))
    ax.set_xticklabels(np.arange(1, a.shape[1] + 1))
    ax.set_yticklabels(np.arange(1, a.shape[0] + 1))
    ax.set_xlim(-0.5, a.shape[1]-0.5)
    ax.set_ylim(-0.5, a.shape[0]-0.5)

    # Add a colorbar for diagonal values
    cbar_diag = fig.colorbar(plt.cm.ScalarMappable(norm=diag_norm, cmap=diag_cmap), ax=ax, shrink=0.5)
    cbar_diag.set_label('Self Transition Times / ns')

    # Add a colorbar for off-diagonal values
    cbar_off_diag = fig.colorbar(plt.cm.ScalarMappable(norm=off_diag_norm, cmap=custom_off_diag_cmap), ax=ax, shrink=0.5)
    cbar_off_diag.set_label('Transitiion Times / ns')

    if title:
        ax.set_title(title)

    ax.set_xlabel("From Macrostate")
    ax.set_ylabel("To Macrostate")
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
    ax.set_title(f"Macrostate Features, {micro_feature.shape[1]} clusterings")
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
        #y = [1e-9, (ma * weights).sum() / weights.sum() * 1e-3]
        y = [1e-9, (ma * weights).sum() / weights.sum() * 1e-3]
        if b:
            #ax.plot(x, y, c=color, label=label + " / 1000")
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


### CONTACT REPRESENTATION ###################################################

def contact_rep(contacts, cluster_file, state_traj, output, grid):
    """
    Adapted from msmhelper.

    Contact representation of states.

    This script creates a contact representation of states. Were the states
    are obtained by [MoSAIC](https://github.com/moldyn/MoSAIC) and the contact
    representation was introduced in Nagel et al.[^1].

    [^1]: Nagel et al., **Selecting Features for Markov Modeling: A Case Study
          on HP35.**, *J. Chem. Theory Comput.*, submitted,

    """
    # setup matplotlib
    pplt.use_style(
        figsize=0.8, colors='pastel_autumn', true_black=True, latex=False,
    )

    # load files
    #contacts = mh.opentxt(contact_file, dtype=np.float64)
    states = np.unique(state_traj)
    clusters = load_clusters(cluster_file)

    contact_idxs = np.hstack(clusters)
    n_idxs = len(contact_idxs)
    n_frames = len(contacts)

    xtickpos = np.cumsum([
        0,
        *[
            len(clust) for clust in clusters[:-1]
        ],
    ]) - 0.5
    nrows, ncols = grid
    hspace, wspace = 0, 0
    ylims = 0, np.quantile(contacts, 0.999)

    counter = 0
    for chunk in mh.plot._ck_test._split_array(states, nrows * ncols):
        fig, axs = plt.subplots(
            int(np.ceil(len(chunk) / ncols)),
            ncols,
            sharex=True,
            sharey=True,
            squeeze=False,
            gridspec_kw={'wspace': wspace, 'hspace': hspace},
        )

        # ignore outliers
        for state, ax in zip(chunk, axs.flatten()):
            contacts_state = contacts[state_traj == state]
            pop_state = len(contacts_state) / n_frames

            # get colormap
            c1, c2, c3 = pplt.categorical_color(3, 'C0')

            stats = {
                idx: boxplot_stats(contacts_state[:, idx])[0]
                for idx in contact_idxs
            }

            for color, (key_low, key_high), label in (
                (c3, ('whislo', 'whishi'), r'$Q_{1/3} \pm 1.5\mathrm{IQR}$'),
                (c2, ('q1', 'q3'), r'$\mathrm{IQR} = Q_3 - Q_1$'),
            ):
                ymax = [stats[idx][key_high] for idx in contact_idxs]
                ymin = [stats[idx][key_low] for idx in contact_idxs]
                ax.stairs(
                    ymax,
                    np.arange(n_idxs + 1) - 0.5,
                    baseline=ymin,
                    color=color,
                    lw=0,
                    fill=True,
                    label=label,
                )

            ax.hlines(
                [stats[idx]['med'] for idx in contact_idxs],
                xmin=np.arange(n_idxs) - 0.5,
                xmax=np.arange(n_idxs) + 0.5,
                label='median',
                color=c1,
            )

            pplt.text(
                0.5,
                0.95,
                fr'S{state} {pop_state:.1%}',
                ha='center',
                va='top',
                ax=ax,
                transform=ax.transAxes,
                contour=True,
            )

            ax.set_xlim([-0.5, n_idxs - 0.5])
            ax.set_ylim(*ylims)
            ax.set_xticks(xtickpos)
            ax.set_xticklabels(np.arange(len(xtickpos)) + 1)

            ax.grid(False)
            for pos in xtickpos:
                ax.axvline(pos, color='pplt:grid', lw=1.0)

        pplt.hide_empty_axes()
        pplt.legend(
            ax=axs[0, 0],
            outside='top',
            bbox_to_anchor=(
                0,
                1.0,
                axs.shape[1] + wspace * (axs.shape[1] - 1),
                0.01,
            ),
            frameon=False,
            ncol=2,
        )
        pplt.subplot_labels(
            xlabel='contact clusters',
            ylabel='distances [nm]',
        )

        # save figure and continue
        if output is None:
            output = f'{state_file}.contactRep.pdf'
        # insert state_str between pathname and extension
        path, ext = splitext(output)
        #pplt.savefig(f'{path}.state{chunk[0]:.0f}-{chunk[-1]:.0f}{ext}')
        if counter == 0:
            pplt.savefig(output)
        else:
            pplt.savefig(f'{path}.state{chunk[0]:.0f}-{chunk[-1]:.0f}{ext}')
        counter += 1
            


### REPORT ###################################################################

def report_stochastic(cl, ref, multi_feature, cluster_file, out, frame_length=0.2):
    """
    frame_length in ns
    """
    if not os.path.isdir(out):
        os.makedirs(out)

    tex = os.path.join(out, os.path.basename(out)) + ".tex"
    timescales_plot = os.path.join(out, "timescales.pdf")

    plot_relative_implied_timescales(cl, ref, timescales_plot)

    dendrogram_min = os.path.join(out, "dendro_min.pdf")
    dendrogram_max = os.path.join(out, "dendro_max.pdf")
    
    min_ts = np.where(cl.timescales[:, 0].min() == cl.timescales[:, 0])[0][0]
    max_ts = np.where(cl.timescales[:, 0].max() == cl.timescales[:, 0])[0][0]

    cl.plot(dendrogram_min, min_ts)
    cl.plot(dendrogram_max, max_ts)

    contact_rep_min = os.path.join(out, "contact_rep_min.pdf")
    contact_rep_max = os.path.join(out, "contact_rep_max.pdf")
    contact_rep(multi_feature, cluster_file, cl.macrotraj[:, min_ts], contact_rep_min, (4, 3))
    contact_rep(multi_feature, cluster_file, cl.macrotraj[:, max_ts], contact_rep_max, (4, 3))

    similarity_file = os.path.join(out, "similarity.pdf")
    evaluate_stochastic_clustering(ref, cl, similarity_file)

    # header
    # - Title 
    label = f"ana:{os.path.basename(out)}"

    lagtime = f"Lagtime: \\SI{{{cl.tlag*frame_length}}}{{\\nano\\second}}"
    traj_length = f"Traj langth: \\SI{{{cl.traj.shape[0]*frame_length*1e-3:.0f}}}{{\\micro\\second}}"

    title = f"Stochastic Clustering"
    kernel = f"\\verb|{cl.kernel}|"
    thr = f"$\\mathrm{{pop}}_\\mathrm{{min}}={cl.pop_thr}$, $q_\\mathrm{{min}}={cl.q_min}$"
    if cl.kernel.method == "n":
        mode = f"{thr}, n={cl.kernel.param}, c=\\SI{{{cl.kernel.c*100:.0f}}}{{\\percent}}"
    elif cl.kernel.method == "p":
        mode = f"{thr}, p=\\SI{{{cl.kernel.param*100:.0f}}}{{\\percent}}, c=\\SI{{{cl.kernel.c*100:.0f}}}{{\\percent}}"
    else:
        mode = thr
    runs = f"{cl.n_runs} clusterings"
    thresholds = f"pop: \\SI{{{cl.pop_thr*100:.2f}}}{{\\percent}} $q_\\mathrm{{min}}$={cl.q_min}"

    if cl.feature_kernel != 1:
        feature_kernel = f"\\verb|{cl.feature_kernel}|"
        feature_params = f"$\\sigma$={cl.feature_kernel.sigma}, b={cl.feature_kernel.b}"
    else:
        feature_kernel = "No feature"
        feature_params = ""

    header = f"""
\\newpage
\\begin{{analysis}}
\\label{{{label}}}
\\begin{{table}}[H]
\\centering
\\begin{{tabular}}{{lll}}
    General & Clustering & Feature \\\\\\midrule
    {lagtime} & {kernel} & {feature_kernel} \\\\
    {traj_length} & {runs} & {feature_params} \\\\
    & {mode} &
\\end{{tabular}}
\\end{{table}}

\\begin{{figure}}[H]
\\centering
\\includegraphics[width=1.0\\textwidth]{{{os.path.abspath(timescales_plot)}}}
\\includegraphics[width=0.48\\textwidth]{{{os.path.abspath(dendrogram_min)}}}
\\includegraphics[width=0.48\\textwidth]{{{os.path.abspath(dendrogram_max)}}}
\\includegraphics[width=0.48\\textwidth]{{{os.path.abspath(contact_rep_min)}}}
\\includegraphics[width=0.48\\textwidth]{{{os.path.abspath(contact_rep_max)}}}
\\end{{figure}}

\\begin{{figure}}[H]
\\centering
\\includegraphics[width=1.0\\textwidth]{{{os.path.abspath(similarity_file)}}}
\\end{{figure}}

\\subsubsection*{{Previous page}}

Top: relative implied timescales. 
Centre-left/bottom: dendrogram and contact representation of stochastic 
clustering with lowest implied timescale 
(${cl.timescales[min_ts, 0] / ref.timescales[0, 0]:.2f}\\cdot t_\\mathrm{{det}}$) 
Centre-right/bottom: dendrogram and contact representation of stochastic 
clustering with lowest implied timescale 
(${cl.timescales[max_ts, 0] / ref.timescales[0, 0]:.2f}\\cdot t_\\mathrm{{det}}$)

\\subsubsection*{{This page}}

Similarity of macrostates. The numbers in brakets are the number of microstate in the reference macrostate. The similarities are defined as follows:

\\begin{{align*}}
	S_1\\left(s_i|s\\right) &= \\max_j\\left(\\frac{{s_i \\cap s_j}}{{s_i \\cup s_j}}\\right) \\quad \\mathrm{{union}} \\\\
	S_2\\left(s_i|s\\right) &= \\max_j\\left(\\frac{{s_i \\cap s_j}}{{s_i}}\\right) \\quad \\mathrm{{reference}} \\\\
	S_3\\left(s_i|s\\right) &= \\max_j\\left(\\frac{{s_i \\cap s_j}}{{s_j}}\\right) \\quad \\mathrm{{clustering}}
\\end{{align*}}
\\end{{analysis}}
"""

    with open(tex, "w") as f:
        f.write(header)

def report_1v1(cl, ref, multi_feature, cluster_file, out, frame_length=0.2):
    """
    frame_length in ns
    """
    if not os.path.isdir(out):
        os.makedirs(out)
    
    if cl.timescales == None:
        cl.calc_timescales()
    if ref.timescales == None:
        ref.calc_timescales()
    
    its = cl.timescales / ref.timescales

    tex = os.path.join(out, os.path.basename(out)) + ".tex"
    # timescales_plot = os.path.join(out, "timescales.pdf")
    #
    # plot_relative_implied_timescales(cl, ref, timescales_plot)

    dendrogram_cl = os.path.join(out, "dendro_cl.pdf")
    # dendrogram_ref = os.path.join(out, "dendro_ref.pdf")
    
    cl.plot(dendrogram_cl, 0)
    # ref.plot(dendrogram_ref, 0)

    contact_rep_cl = os.path.join(out, "contact_rep_cl.pdf")
    # contact_rep_max = os.path.join(out, "contact_rep_max.pdf")
    contact_rep(multi_feature, cluster_file, cl.macrotraj[:, 0], contact_rep_cl, (4, 3))
    # contact_rep(multi_feature, cluster_file, cl.macrotraj[:, max_ts], contact_rep_max, (4, 3))

    similarity_file = os.path.join(out, "similarity.pdf")
    evaluate_stochastic_clustering(ref, cl, similarity_file)

    # header
    # - Title 
    label = f"ana:{os.path.basename(out)}"

    lagtime = f"Lagtime: \\SI{{{cl.tlag*frame_length}}}{{\\nano\\second}}"
    traj_length = f"Traj langth: \\SI{{{cl.traj.shape[0]*frame_length*1e-3:.0f}}}{{\\micro\\second}}"

    title = f"1v1 comparison"
    kernel = f"\\verb|{cl.kernel}|"
    thr = f"$\\mathrm{{pop}}_\\mathrm{{min}}={cl.pop_thr}$, $q_\\mathrm{{min}}={cl.q_min}$"
    try:
        if cl.kernel.method == "n":
            mode = f"n={cl.kernel.param}, c=\\SI{{{cl.kernel.c*100:.0f}}}{{\\percent}}"
        elif cl.kernel.method == "p":
            mode = f"p=\\SI{{{cl.kernel.param*100:.0f}}}{{\\percent}}, c=\\SI{{{cl.kernel.c*100:.0f}}}{{\\percent}}"
        else:
            mode = ""
    except AttributeError:
        mode = ""
    thresholds = f"pop: \\SI{{{cl.pop_thr*100:.2f}}}{{\\percent}} $q_\\mathrm{{min}}$={cl.q_min}"

    if cl.feature_kernel != 1:
        feature_kernel = f"\\verb|{cl.feature_kernel}|"
        feature_params = f"$\\sigma$={cl.feature_kernel.sigma}, b={cl.feature_kernel.b}"
    else:
        feature_kernel = "No feature"
        feature_params = ""

    header = f"""
\\newpage
\\begin{{analysis}}
\\label{{{label}}}
\\begin{{table}}[H]
\\centering
\\begin{{tabular}}{{lll}}
    General & Clustering & Feature \\\\\\midrule
    {lagtime} & {kernel} & {feature_kernel} \\\\
    {traj_length} & {thr} & {feature_params} \\\\
    & {mode} & \\\\
\\end{{tabular}}
\\end{{table}}

\\begin{{figure}}[H]
\\centering
\\includegraphics[width=0.53\\textwidth]{{{os.path.abspath(dendrogram_cl)}}}
\\includegraphics[width=0.46\\textwidth]{{{os.path.abspath(contact_rep_cl)}}}
\\end{{figure}}

\\begin{{figure}}[H]
\\centering
\\includegraphics[width=0.7\\textwidth]{{{os.path.abspath(similarity_file)}}}
\\end{{figure}}

\\textbf{{Left}} Dendrogram and macrostate assignment. \\textbf{{Right}}
Contact representation. Implied timescales are 
$t_1={its[0, 0]:.2f}\\cdot t_\\mathrm{{det}}$, 
$t_2={its[0, 1]:.2f}\\cdot t_\\mathrm{{det}}$, 
$t_3={its[0, 2]:.2f}\\cdot t_\\mathrm{{det}}$
\\textbf{{Bottom}} Similarity of macrostates. The numbers in brakets are the number of microstate in the reference macrostate. The similarities are defined as follows:

\\begin{{align*}}
	S_1\\left(s_i|s\\right) &= \\max_j\\left(\\frac{{s_i \\cap s_j}}{{s_i \\cup s_j}}\\right) \\quad \\mathrm{{union}} \\\\
	S_2\\left(s_i|s\\right) &= \\max_j\\left(\\frac{{s_i \\cap s_j}}{{s_i}}\\right) \\quad \\mathrm{{reference}} \\\\
	S_3\\left(s_i|s\\right) &= \\max_j\\left(\\frac{{s_i \\cap s_j}}{{s_j}}\\right) \\quad \\mathrm{{clustering}}
\\end{{align*}}
\\end{{analysis}}
"""

    with open(tex, "w") as f:
        f.write(header)

