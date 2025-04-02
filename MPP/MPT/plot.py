#!/usr/bin/env python3
"""
plot.py
==================

Plot dendrogram from linkage and qmin. Most of the code originates from
procss_mpp.py from Daniel Nagel.
"""

import os
from os.path import splitext

import numpy as np
import prettypyplot as pplt
from matplotlib import pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize, LinearSegmentedColormap, LogNorm, ListedColormap
from matplotlib import colors
from matplotlib.cbook import boxplot_stats
import matplotlib.animation as animation
import matplotlib.patches as patches
from matplotlib.ticker import MultipleLocator
import msmhelper as mh
from msmhelper._cli.contact_rep import load_clusters
from scipy.stats import pearsonr
import MPT.utils as utils
from MPT.sankey_gap import sankey
import MPT.kernel as krnl

plt.rcParams['font.family'] = 'sans-serif'

### DENDROGRAM ###############################################################


def plot_tree(root, macrostate_assignment, output_file, scale=1):
    """
    Plot the dendrogram from a given state tree of BinaryTreeNode.
    """
    n_states = len(root.leaves)

    # setup matplotlib
    pplt.use_style(figsize=3.2*scale, figratio='golden', true_black=True)
    plt.rcParams['font.family'] = 'sans-serif'

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

    ax.set_ylabel(r'metastability $Q_\mathrm{min}$')
    ax.set_xlabel('microstates')
    ax.set_xlim(-0.005 * n_states, 1.005 * n_states)
    ax.set_ylim(0, 1.05)

    # plot legend
    cmap = plt.get_cmap('plasma_r', 10)
    bins = np.linspace(0, 1, 11)
    norm = Normalize(bins[0], bins[-1])
    # label = r'$\langle Q \rangle_\text{state} $'
    label = 'fraction of native contacts'

    cmappable = ScalarMappable(norm, cmap)
    plt.sca(ax)
    pplt.colorbar(cmappable, width='5%', label=label, position='top')

    # bring microstates in the right order
    macrostate_assignment = macrostate_assignment[:, [l.name for l in root.leaves]]

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
    plt.close()


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
        ax.set_title(f"state {state+1}")
    fig.supxlabel("Macrostate similarity")
    fig.supylabel(f"Count of clusterings ({sto.n_runs} clusterings)")
    leg = plt.figlegend(["union", "reference", "clustering"], ncols=3, loc='lower center', bbox_to_anchor=(0.5, 0.05))
    plt.tight_layout(rect=(0, 0.04, 1, 1))
    plt.savefig(out)
    plt.close()


### IMPLIED TIMESCALES #######################################################

def plot_implied_timescales(trajs, lagtimes, out, titles="", frame_length=0.2, first_ref=False, scale=1):
    """
    frame_length in ns / frame
    """
    if first_ref:
        ref_traj = trajs.pop(0)
    x, y = utils.get_grid_format(len(trajs))
    # pplt.use_style(figsize=(2*x, 2*y), latex=False, colors='pastel_autumn')
    pplt.use_style(figsize=(2.8*scale, 3.2*scale), latex=False, colors='pastel_autumn')
    fig, axs = plt.subplots(y, x, sharex=True, sharey=True)
    plt.grid(False)
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
        ax.axvline(10, color='pplt:grid')
        # ax.yaxis.set_major_formatter(mtick.LogFormatterSciNotation)
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

    # Get handles and labels
    handles, labels = plt.gca().get_legend_handles_labels()

    # Reorder the handles and labels manually to achieve column-major ordering
    # desired_order = [0, 3, 1, 4, 2, 5]  # Indices in column-major order
    desired_order = [3, 0, 4, 1, 5, 2]  # Indices in column-major order
    handles = [handles[i] for i in desired_order]
    labels = [labels[i] for i in desired_order]

    pplt.legend(handles=handles, labels=labels, outside='top', frameon=False, ncols=3)

    plt.tight_layout()
    plt.savefig(out)
    plt.close()

def _plot_impl_times(impl_times, lagtimes, ax, ls="-"):
    """Plot the implied timescales"""
    colors = ['#264653', '#2A9D8F', '#E9C46A']
    for idx, impl_time in enumerate(impl_times.T):
        if ls == ":":
            label = f'$t_{{\\mathrm{{ref}},{idx + 1}}}$'
        else:
            label = f'$t_{idx + 1}$'
        ax.plot(lagtimes, impl_time, label=label, color=colors[idx], ls=ls)

    xlim = lagtimes[0], lagtimes[-1]
    ref_low = int(lagtimes.shape[0] / 4)
    ax.set_xlim(xlim)
    # highlight diagonal
    x_i = np.arange(ref_low, xlim[1])
    ax.fill_between(x_i, x_i, color='pplt:grid')
    # pplt.legend(outside='right', frameon=False)


def plot_relative_implied_timescales_(cl, ref, out):
    its = cl.timescales / ref.timescales

    fig, axs = plt.subplots(1, 4, figsize=(8, 2.5), sharey=True)
    for i, ax in enumerate(axs[:-1]):
        ax.hist(its[:, i], bins=20)
        ax.set_title(f'its {i+1}')
    axs[-1].hist(its.mean(axis=1), bins=20)
    axs[-1].set_title(f'Mean its {1}-{i+1}')

    fig.supxlabel(r"Relative Implied Timescale $\left(\frac{t_\mathrm{stoch}}{t_\mathrm{det}}\right)$")
    fig.supylabel('Count of Clusterings')
    plt.tight_layout()
    plt.savefig(out)
    plt.close()


def plot_relative_implied_timescales(cl, out):
    ref = cl.reference
    its = cl.timescales / ref.timescales

    fig = plt.figure(figsize=(8, 2.5))
    ax1 = fig.add_subplot(1, 3, 1)
    ax2 = fig.add_subplot(1, 3, 2, sharey=ax1)
    ax3 = fig.add_subplot(1, 3, 3)

    for ax in (ax1, ax2, ax3):
        ax.grid(False)

    ax1.hist(its[:, 0], bins=20)
    ax1.set_title("its 1")
    ax1.set_xlabel(r"Relative Implied Timescale $\left(\frac{t_\mathrm{stoch}}{t_\mathrm{ref}}\right)$")
    ax1.set_ylabel('Count of Clusterings')
    ax2.hist(its.mean(axis=1), bins=20)
    ax2.set_title(f"Mean its {1}-{3}")
    ax2.set_xlabel(r"Relative Implied Timescale $\left(\frac{t_\mathrm{stoch}}{t_\mathrm{ref}}\right)$")

    bins = np.array(range(min(cl.n_macrostates)-1, max(cl.n_macrostates)+1)) + 0.5

    ax3.hist(cl.n_macrostates, bins=bins)
    ax3.set_title("n macrostates")
    ax3.set_xlabel("macrostate count")

    plt.tight_layout()
    plt.savefig(out)
    plt.close()


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
    plt.close()

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
    plt.close()

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

    off_diag_norm = LogNorm(vmin=off_diag_values.min(), vmax=threshold/(1-color_thr))
    off_diag_cmap = plt.cm.viridis_r

    # Create a custom colormap for off-diagonal values including light gray
    colors_list = plt.cm.viridis_r(np.linspace(0, 1, 256))
    gray = np.array([0.9, 0.9, 0.9, 1.0])
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
    plt.close()


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
    plt.close()

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

def contact_rep(contacts, cluster_file, state_traj, output, grid, scale=1):
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
        figsize=1.2*scale, colors='pastel_autumn', true_black=True, latex=False,
    )

    # load files
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
        if counter == 0:
            pplt.savefig(output)
            plt.close()
        else:
            pplt.savefig(f'{path}.state{chunk[0]:.0f}-{chunk[-1]:.0f}{ext}')
            plt.close()
        counter += 1
            

### SANKEY ###################################################################

def plot_sankey(cl, ref, out, ax=None, scale=1):
    features = []
    for macrostate in cl.tree[cl.n_i].macrostates:
        features.append(macrostate.feature)
    ma_order = np.argsort(features)[::-1]
    colorDict = {}
    for i, o in enumerate(ma_order):
        colorDict[str(i+1)] = cl.tree[cl.n_i].macrostates[o].color
    if ax is None:
        pplt.use_style(figsize=(1.7*scale, 3.6*scale), true_black=True)
    sankey(
        left=(cl.macrostates_map[cl.n_i] + 1).astype(str),
        right=(ref.macrostates_map[0] + 1).astype(str),
        leftWeight=ref.pop,
        rightWeight=ref.pop,
        leftLabels=np.arange(1, cl.n_macrostates[cl.n_i] + 1).astype(str).tolist(),
        rightLabels=np.arange(1, ref.n_macrostates[0] + 1).astype(str).tolist(),
        colorDict=colorDict,
        ax=ax,
    )
    if ax is None:
        pplt.savefig(out)
        plt.close()


### RMSD HEATMAP #############################################################

def plot_rmsd_(vars, row_heights, helices=None, filename=None, num_x_labels=8):
    """
    Plots a 2D NumPy array as a heatmap with a logarithmic color scale and variable row heights.

    Parameters:
    - vars (np.ndarray): The 2D NumPy array to plot. Values must be positive for logarithmic scaling.
    - row_heights (np.ndarray): 1D array defining the height of each row.
    - helices (np.ndarray): Array with start and end points for blocks to be indicated in the bottom row.
    - filename (str, optional): If provided, saves the heatmap to this file.
    """
    # Ensure all values are positive for logarithmic scaling
    if np.any(vars <= 0):
        raise ValueError("All values in `vars` must be positive for logarithmic scaling.")
    
    if vars.shape[0] != len(row_heights):
        raise ValueError("Length of `row_heights` must match the number of rows in `vars`.")

    # Calculate y-axis boundaries using cumulative sum of row heights
    y_boundaries = np.insert(np.cumsum(row_heights), 0, 0)

    # Generate x-axis boundaries (evenly spaced)
    x_boundaries = np.arange(vars.shape[1] + 1)

    # Create the heatmap with a logarithmic color scale
    # plt.figure(figsize=(4, 3))
    fig, ax = plt.subplots(figsize=(4, 3))
    plt.pcolormesh(x_boundaries, y_boundaries, vars, cmap="viridis", norm=LogNorm(), shading='flat') # , edgecolors='black', linewidth=0.5
        
    # Draw horizontal lines at each y-boundary
    for y in y_boundaries:
        plt.axhline(y=y, color='black', linewidth=0.5)

    if helices is not None:
        # Add the additional row at the bottom for block indicators
        plot_height = y_boundaries[-1]  # Total height of the heatmap
        indicator_row_height = plot_height * 0.05  # 5% of plot height
        indicator_y = -indicator_row_height * 1.2  # Position for the indicator row

        # Draw the white background for the indicator row
        plt.fill_between(x_boundaries, indicator_y, 0, color='white')

        # Draw a horizontal line across the indicator row
        indicator_line = indicator_y + 0.5 * indicator_row_height
        plt.plot([0, vars.shape[1]], [indicator_line] * 2, color='black', linewidth=0.8)

        # Add black boxes for each block in 'helices'
        block_height = indicator_row_height * 0.9  # Block height as 90% of the row height
        for start, end in helices:
            start -= 1
            rect = patches.Rectangle(
                (start, indicator_y + 0.05 * indicator_row_height),  # Position of the block
                end - start, block_height, color='black'
            )
            ax.add_patch(rect)
    
        displayed_y_ticks = [indicator_line]
        displayed_y_labels = ["H"]

    else:
        displayed_y_ticks = []
        displayed_y_labels = []

    # Set x-axis labels with 10 evenly spaced labels along the x-axis, ensuring equal intervals
    num_x_ticks = vars.shape[1]
    interval = max(1, num_x_ticks // (num_x_labels - 1))
    x_ticks = np.arange(0, num_x_ticks, interval)
    if x_ticks[-1] != num_x_ticks - 1:
        x_ticks = np.append(x_ticks, num_x_ticks - 1)  # Ensure the last label aligns with the array's end

    # Set y-axis labels (start from 1)
    y_ticks = y_boundaries[:-1] + np.diff(y_boundaries) / 2
    y_labels = np.arange(1, len(y_ticks) + 1)
    
    # Determine spacing threshold based on figure size and row height differences
    min_spacing = (y_boundaries[-1] - y_boundaries[0]) / len(y_ticks) * 1.0  # Minimum spacing between labels
    
    last_displayed_y = -np.inf
    for y, label in zip(y_ticks, y_labels):
        if y - last_displayed_y >= min_spacing:  # Only display label if it's far enough from the last one
            displayed_y_ticks.append(y)
            displayed_y_labels.append(label)
            last_displayed_y = y

    plt.xticks(ticks=x_ticks + 0.5, labels=x_ticks + 1)
    plt.yticks(ticks=displayed_y_ticks, labels=displayed_y_labels)

    if helices is not None:
        plt.ylim(indicator_y - indicator_row_height * 0.22, y_boundaries[-1])
    
    plt.ylabel("Macrostate")
    plt.xlabel("Residue")

    # Hide grid lines
    # plt.grid(True, axis="y", mec="k")
    plt.grid(False)

    # Display the colorbar
    plt.colorbar(label="RMSD Variance / nm")

    # Save to file if filename is provided
    if filename:
        plt.savefig(filename, bbox_inches="tight", dpi=100)
    else:
        plt.show()
    plt.close()


### RMSD LINES ###############################################################

def plot_rmsd(rmsds, pops, helices=None, filename=None, num_x_labels=8):
    """
    Plots a 2D NumPy array as a heatmap with a logarithmic color scale and variable row heights.

    Parameters:
    - vars (np.ndarray): The 2D NumPy array to plot. Values must be positive for logarithmic scaling.
    - row_heights (np.ndarray): 1D array defining the height of each row.
    - helices (np.ndarray): Array with start and end points for blocks to be indicated in the bottom row.
    - filename (str, optional): If provided, saves the heatmap to this file.
    """
    # Ensure all values are positive for logarithmic scaling
    if np.any(rmsds <= 0):
        raise ValueError("All values in `rmsds` must be positive for logarithmic scaling.")
    
    if rmsds.shape[0] != len(pops):
        raise ValueError("Length of `pops` must match the number of rows in `rmsds`.")

    if helices is not None:
        n_plots = rmsds.shape[0] + 1
    else:
        n_plots = rmsds.shape[0]

    pplt.use_style(
        figsize=(8, 6) , colors="pastel_autumn", true_black=True, latex=False,
    )
    fig, axs = plt.subplots(
        n_plots,
        2,
        sharex="col",
        figsize=(8, 6),
        width_ratios=[6, 1],
        gridspec_kw={'wspace':0, 'hspace':0},
    )

    ylim = 0.5 * rmsds.min(), 2 * rmsds.max()
    pops = pops / pops.sum()
    ylim_hist = 0, 1.05 * pops.max()

    for i, ((ax, hist_ax), rmsd, pop) in enumerate(zip(axs[:-1] if helices is not None else axs, rmsds, pops)):
        rect = patches.Rectangle(
            (0, 0.3),  # Position of the block
            pop, 0.4, # color='black'
        )
        hist_ax.add_patch(rect)
        hist_ax.set_xlim(ylim_hist)
        hist_ax.set_yticks([], [])
        hist_ax.grid(False)

        ax.plot(np.arange(rmsd.shape[0])+1, rmsd)
        ax.fill_between(
            np.arange(rmsd.shape[0])+1,
            [ylim[0]]*rmsd.shape[0],
            rmsd,
            alpha=0.5,
            # facecolor="none",
            # hatch="/",
        )

        ax.set_yscale("log")
        ax.set_ylabel(f"{i+1}")
        ax.set_xlim((0.5, rmsd.shape[0] + 0.5))
        ax.set_ylim(ylim)
        ax.grid(True)

    if helices is not None:
        helices_ax = axs[-1, 0]
        helices_ax.plot([1, rmsds.shape[1]], [0.5, 0.5]) #, c="k")
        for start, end in helices:
            start -= 0.5
            end += 0.5
            rect = patches.Rectangle(
                (start, 0.3),  # Position of the block
                end - start, 0.4, # color='black'
            )
            helices_ax.add_patch(rect)
       
        helices_ax.set_ylim((0, 1))
        helices_ax.set_ylabel("H")
        helices_ax.set_yticks([], [])
        helices_ax.grid(False)

        axs[-1, 1].grid(False)
        axs[-1, 1].set_yticks([], [])

    hist_ticks = axs[-1, 1].get_xticks()
    hist_labels = axs[-1, 1].get_xticklabels()
    # hist_labels = [f"{float(i._text)/1000:.0f}k" for i in hist_labels]
    hist_labels[0] = ""
    axs[-1, 1].set_xticks(hist_ticks, hist_labels)

    axs[-1, 0].xaxis.set_major_locator(MultipleLocator(5))
    axs[-1, 0].xaxis.set_minor_locator(MultipleLocator(1))
    axs[-1, 0].set_xlabel("Residue")
    axs[-1, 1].set_xlabel("Population")
    fig.supylabel("Macrostate; RMSD Variance / nm")

    # Save to file if filename is provided
    plt.tight_layout()
    if filename:
        plt.savefig(filename, bbox_inches="tight", dpi=100)
    else:
        plt.show()
    plt.close()


### TRAJECTORY ###############################################################

def plot_state_trajectory(trajectory, filename, row_length=0.2):
    """
    Plot state trajectory

    trajectory (np.ndarray): state trajectory
    filename (str): file name to save the plot to
    row_length (int|float):
        row_length > 1: number of frames in each row
        0 < row_length <= 1: fraction of total frames per row (1/n_rows)
    """
    if row_length > 1:
        x_max = int(row_length)
    elif row_length > 0:
        x_max = int(np.ceil(trajectory.shape[0] * row_length))
    else:
        raise ValueError("row_lengthg must be > 0")

    # Calculate unique states and their lengths
    unique_states, lengths = utils.find_state_lengths(trajectory)
    n_rows = int(np.ceil(trajectory.shape[0] / x_max))
    
    # Set up figure size proportional to data
    width = max(6, x_max * 0.0001)  # Minimum width of 6 inches
    height = max(2, (unique_states.max() - unique_states.min() + 1) * 0.05 * n_rows + 0.6)  # Minimum height of 4 inches
    
    
    # Use a logarithmic color scale for lengths
    norm = colors.LogNorm(vmin=lengths.min(), vmax=lengths.max())
    cmap = plt.cm.viridis
   
    # plt.figure(figsize=(width, height))
    # a4 = True
    a4 = False
    if a4:
        figsize = (11.7, 8.3)
    else:
        figsize = (width, height)

    fig, axs = plt.subplots(n_rows, 1, sharex=True, figsize=figsize, gridspec_kw={'wspace':0, 'hspace':0})
    axi = 0

    # Plot each state occurrence as a line segment
    x_start = 0  # Initial x-coordinate for the first segment
    for state, length in zip(unique_states, lengths):
        x_end = x_start + length  # Calculate end position of this segment on the x-axis
        color = cmap(norm(length))  # Color based on the length, logarithmically scaled

        if x_end <= x_max:
            # Plot the line segment for the current state
            # plt.plot([x_start, x_end], [state, state], color=color, linewidth=3, solid_capstyle='butt')
            axs[axi].plot([x_start, x_end], [state, state], color=color, linewidth=3, solid_capstyle='butt')
        else:
            axs[axi].plot([x_start, x_max], [state, state], color=color, linewidth=3, solid_capstyle='butt')
            x_end -= x_max
            x_start = 0
            axi += 1
            axs[axi].plot([x_start, x_end], [state, state], color=color, linewidth=3, solid_capstyle='butt')
            
        
        # Move x_start to the end of the current segment for the next one
        x_start = x_end

    # Add color bar to indicate the log-scale of state lengths
    # sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    # # sm.set_array(lengths)
    # sm.set_array([])
    # plt.colorbar(sm, label='Log of State Length')
    
    # Label axes and set title
    # fig.supxlabel("Index in State Sequence")
    fig.supylabel("State Index")
    # fig.supxlabel("Frames")
    axs[-1].set_xlabel("Frames")
    # plt.title("Line Plot of State Trajectory with Length-Color Coding")
  
    for ax in axs:
        # ax.grid(visible=False)
        ax.set_ylim(unique_states.min() - 1, unique_states.max() + 1)

    # Set axis limits
    plt.xlim(0, x_max)
    # plt.ylim(np.min(unique_states) - 0.2, np.max(unique_states) + 0.2)
  
    # plt.subplots_adjust(wspace=0, hspace=0)
    plt.tight_layout()
    # Save the plot to the specified file
    plt.savefig(filename)
    plt.close()  # Close the plot to free memory


### CORRELATION ##############################################################

def plot_correlation_evolution(
        feature1,
        feature2,
        out,
        weights=None,
        label1="feature 1",
        label2="feature 2",
        clip_to_greater_zero=None,
    ):
    """
    Plot two features as a function of time.

    feature1, feature2 (list[np.ndarray]): list containing coordinates of feature as numpy array
    weights (list[np.ndarray]): list containing weights of the respective data points as numpy array
    label1, label2 (str): Label for the features
    clip_to_greater_zero (list[np.ndarray]): Consider only data points where np.ndarray > 0
    """
    if clip_to_greater_zero is not None:
        mask = [dq > 0 for dq in clip_to_greater_zero]
        feature1 = [f1[m] for m, f1 in zip(mask, feature1)]
        feature2 = [f2[m] for m, f2 in zip(mask, feature2)]
        if weights is not None:
            weights = [w[m] for m, w in zip(mask, weights)]

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    # fig, ax = plt.subplots()
    artists = []

    if weights is None:
        weights = []
        for f1 in feature1:
            weights.append(np.full(f1.shape, 1))

    for f1, f2, w in zip(feature1, feature2, weights):
        container = ax.scatter(f1, f2, s=w/1000, c="k", alpha=0.4)
        # container = ax.scatter(f1, f2, s=1, c="k")
        artists.append([container])

    ax.set_xlabel(label1)
    ax.set_ylabel(label2)

    ax.set_xscale("log")
    ax.set_yscale("log")

    ani = animation.ArtistAnimation(fig=fig, artists=artists, interval=500)
    ani.save(filename=out, writer="ffmpeg")
    plt.close()
    # plt.show()
    # ani.save(out, writer="pillow")


def plot_pearson(
        feature1,
        feature2,
        out,
        title="Correlation of Two Features",
        clip_to_greater_zero=None,
    ):
    if clip_to_greater_zero is not None:
        mask = [dq > 0 for dq in clip_to_greater_zero]
        feature1 = [f1[m] for m, f1 in zip(mask, feature1)]
        feature2 = [f2[m] for m, f2 in zip(mask, feature2)]

    pplt.use_style(
        figsize=4.8, colors="pastel_autumn", true_black=True, latex=False,
    )
    r = np.array([
        pearsonr(f1, f2) for f1, f2 in zip(feature1[:-1], feature2[:-1])
    ]).T
    l = len(r[0])

    fig, ax = plt.subplots(1, 1, figsize=(4, 3))

    ax.plot(r[0], label="Pearson r")
    # ax.invert_yaxis()
    # ax.plot(r[1], label="p-value (exact distribution)")

    num_transitions = np.array([len(f) for f in feature1])
    max_transitions = max(num_transitions)
    num_transitions = num_transitions / max_transitions
    ax.plot(num_transitions, label="Transitions P > 0")
    secax_y2 = ax.secondary_yaxis("right", (lambda x: x * max_transitions, lambda x: x * max_transitions))
    ax.set_ylabel("p value")
    secax_y2.set_ylabel("Number of Transitions")

    lim = (0, 261)
    ax.hlines(0, lim[0], lim[1], colors="k", lw=1)
    ax.hlines(
        [-0.05, 0.05],
        [lim[0]] * 2,
        [lim[1]] * 2,
        colors="grey",
        linestyle="dashed",
        # label="p-value = 0.05",
    )
    
    ax.set_title(title)
    ax.set_xlabel("Lumping Step")

    ax.grid(False)

    ax.legend()

    plt.savefig(out)
    plt.close()


def plot_correlation_scatter(
        feature1,
        feature2,
        out,
        macro_feature1=None,
        macro_feature2=None,
        weights=None,
        macro_weights=None,
        label1="feature 1",
        label2="feature 2",
        title="Correlation Scatter Plot",
        clip_to_greater_zero=None,
        clip_to_greater_zero_macro=None,
    ):
    """
    Scatter plot two features of a model, optionally add macro feature

    feature1, feature2 (np.ndarray): numpy array containing coordinates of feature
    out (str): file name to save plot to
    macro_feature1, macrofeature2 (np.ndarray): see feature1; highlighted points
    weights (np.ndarray): numpy array containing weights of the respective data points
    label1, label2 (str): Label for the features
    clip_to_greater_zero (np.ndarray): Consider only data points where array > 0
    """
    if clip_to_greater_zero is not None:
        mask = clip_to_greater_zero > 0
        feature1 = feature1[mask]
        feature2 = feature2[mask]
        if weights is not None:
            weights = weights[mask]

    if clip_to_greater_zero_macro is not None:
        mask = clip_to_greater_zero_macro > 0
        macro_feature1 = macro_feature1[mask]
        macro_feature2 = macro_feature2[mask]
        if macro_weights is not None:
            macro_weights = macro_weights[mask]

    if macro_weights is not None:
        macro_weights = np.sqrt(macro_weights)
        macro_weights = macro_weights / macro_weights.max() * 19
        macro_weights += 1

    pplt.use_style(
        figsize=4.8, colors="pastel_autumn", true_black=True, latex=False,
    )

    if weights is None:
        weights = np.full(feature1.shape, 1)
    else:
        weights = np.sqrt(weights)
        weights = weights / weights.max() * 39
        weights += 1

    fig, ax = plt.subplots(1, 1, figsize=(4, 3))

    ax.scatter(feature1, feature2, c="k", s=weights, label="Microstates")
    if macro_feature1 is not None and macro_feature2 is not None:
        if macro_weights is None:
            macro_weights = np.full(macro_feature1.shape, 1)
        ax.scatter(macro_feature1, macro_feature2, c="r", s=macro_weights, label="Macrostates")

    ax.set_xlabel(label1)
    ax.set_ylabel(label2)
    ax.set_title(title)
   
    ax.legend()
    
    plt.savefig(out)
    plt.close()


### CHAPMAN-KOLMOGOROV TEST ##################################################

def chapman_kolmogorov(mpt, out, frame_length=0.2):
    """Chapman-Kolmogorov Test. Frame length in ns"""
    ck = mh.msm.tests.chapman_kolmogorov_test(
        utils.get_multi_state_traj(mpt.macrotraj[:, mpt.n_i], mpt.limits),
        [50, 50, 50, 50, 50],
        4000,
        # int(1550*frame_length),
    )
    pplt.use_style(
        figsize=4.8, colors="pastel_autumn", true_black=True, latex=False,
    )

    nrows, ncols = utils.get_grid_format(mpt.n_macrostates[mpt.n_i])
    for chunk in mh.plot._ck_test._split_array(np.arange(1, mpt.n_macrostates[mpt.n_i]+1), nrows * ncols):
        fig = mh.plot.plot_ck_test(
            ck=ck,
            states=chunk,
            frames_per_unit=1/frame_length,
            unit="ns",
            grid=(ncols, nrows),
        )

    for ax in fig.axes:
        for text in ax.texts:
            text.set_position((0.15, 0.2))
    plt.savefig(out)
    plt.close()


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
    traj_length = f"Traj length: \\SI{{{cl.traj.shape[0]*frame_length*1e-3:.0f}}}{{\\micro\\second}}"

    title = f"Stochastic Clustering"
    kernel = f"\\verb|{cl.kernel}|"
    kl = f"KL = {cl.kernel.kullback_leibler}"
    thr = f"$\\mathrm{{pop}}_\\mathrm{{min}}={cl.pop_thr}$, $q_\\mathrm{{min}}={cl.q_min}$"
    if cl.kernel.method == "n":
        mode = f"n={cl.kernel.param}, c=\\SI{{{cl.kernel.c*100:.0f}}}{{\\percent}}"
    elif cl.kernel.method == "p":
        mode = f"p=\\SI{{{cl.kernel.param*100:.0f}}}{{\\percent}}, c=\\SI{{{cl.kernel.c*100:.0f}}}{{\\percent}}"
    else:
        mode = ""
    runs = f"{cl.n_runs} clusterings"
    thresholds = f"pop: \\SI{{{cl.pop_thr*100:.2f}}}{{\\percent}} $q_\\mathrm{{min}}$={cl.q_min}"

    if cl.feature_kernel:
        feature_kernel = f"\\verb|{cl.feature_kernel}|"
        feature_params = f"$\\sigma$={cl.feature_kernel.sigma}, b={cl.feature_kernel.b}"
    else:
        feature_kernel = "No feature"
        feature_params = ""

    header = f"""
\\newpage
\\begin{{analysis}}
\\label{{{label}}}
\\vspace{{-0.5cm}}
\\begin{{table}}[H]
\\centering
\\begin{{tabular}}{{lll}}
    General & Clustering & Feature \\\\\\midrule
    {lagtime} & {runs} & {feature_kernel} \\\\
    {traj_length} & {thr} & {feature_params} \\\\
    & {mode} & \\\\
    & {kl} &
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
    traj_length = f"Traj length: \\SI{{{cl.traj.shape[0]*frame_length*1e-3:.0f}}}{{\\micro\\second}}"

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

    if cl.feature_kernel:
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


def report(cl, multi_feature, cluster_file, out, helices=None, n_i=0, frame_length=0.2):
    """
    frame_length in ns / frame
    """
    ref = cl.reference
    cl.n_i = n_i

    if not os.path.isdir(out):
        os.makedirs(out)

    tex = os.path.join(out, os.path.basename(out)) + ".tex"

    dendrogram = os.path.join(out, "dendrogram.pdf")
    cl.plot(dendrogram, n_i)

    contact_rep_path = os.path.join(out, "contact_rep.pdf")
    contact_rep(multi_feature, cluster_file, cl.macrotraj[:, n_i], contact_rep_path, (4, 3))

    sankey_path = os.path.join(out, "sankey.pdf")
    cl.plot_sankey(sankey_path)

    rmsd_path = os.path.join(out, "rmsd.pdf")
    cl.plot_rmsd(rmsd_path, helices)

    # header
    # - Title 
    label = f"ana:{os.path.basename(out)}"

    formula = f"$T + b \\cdot T_\\mathrm{{{cl.kernel.similarity}}}"
    if isinstance(cl.feature_kernel, krnl.MultiFeatureKernel):
        feature_term = f" + c \\cdot F_\\mathrm{{{cl.feature_kernel.similarity}}}$"
    else:
        feature_term = "$"
    formula += feature_term

    head_line = f"\\multicolumn{{6}}{{l}}{{{formula}}} && abs. & rel. &&& abs. & rel. \\\\\\toprule"
    first_line =  f"$b$ & {cl.kernel.b:.2f} & \\quad & $q_\\mathrm{{min}}$ & {cl.q_min:.2f} & \\quad & "
    first_line += f"$\\tau_1$ & \\SI{{{cl.timescales[0, 0] * frame_length:.0f}}}{{\\nano\\second}} & "
    first_line += f"{cl.timescales[0, 0] / ref.timescales[0, 0]:.2f} & \\quad"
    first_line += f"& DBI & {cl.davies_bouldin_index(multi_feature)[0]:.2f} &"
    first_line += f"{cl.davies_bouldin_index(multi_feature)[0] / ref.davies_bouldin_index(multi_feature)[0]:.2f} \\\\"
    second_line = f"$c$ & {cl.kernel.c:.2f} & \\quad & $p_\\mathrm{{min}}$ & "
    second_line += f"{cl.pop_thr:.3f} && $H$ & {cl.shannon_entropy[0]:.2f} & "
    second_line += f"{cl.shannon_entropy[0] / ref.shannon_entropy[0]:.2f} && "
    second_line += f"GMRQ & {cl.gmrq[0]:.2f} & {cl.gmrq[0] / ref.gmrq[0]:.2f} \\\\"


    tex_file = f"""
\\newpage
\\begin{{analysis}}
\\label{{{label}}}

\\begin{{table}}[H]
	\\centering
	\\begin{{tabular}}{{lllllllllllll}}
        {head_line}
        {first_line}
        {second_line}
	\\end{{tabular}}
\\end{{table}}

\\includegraphics[width=0.34\\textwidth]{{{os.path.abspath(sankey_path)}}}
\\includegraphics[width=0.64\\textwidth]{{{os.path.abspath(rmsd_path)}}}

\\vspace{{-0.5cm}}
\\begin{{figure}}[H]
    \\centering
    \\includegraphics[width=0.48\\textwidth]{{{os.path.abspath(dendrogram)}}}
    \\includegraphics[width=0.48\\textwidth]{{{os.path.abspath(contact_rep_path)}}}
    \\end{{figure}}
    \\vspace{{-0.6cm}}
\\end{{analysis}}
"""
                # \\parbox{{\\textwidth}}{{
                # \\begin{{align*}}
                #     \\mathrm{{DBI}}&: {cl.davies_bouldin_index(multi_feature)[0]:.2f} | {cl.davies_bouldin_index(multi_feature)[0] / ref.davies_bouldin_index(multi_feature)[0]:.2f} \\\\
                #     \\mathrm{{GMRQ}}&: {cl.gmrq[0]:.2f} | {cl.gmrq[0] / ref.gmrq[0]:.2f}
                # \\end{{align*}}}}

    with open(tex, "w") as f:
        f.write(tex_file)

def report_(cl, multi_feature, cluster_file, out, n_i=0, frame_length=0.2):
    """
    frame_length in ns
    """
    ref = cl.reference

    if not os.path.isdir(out):
        os.makedirs(out)

    tex = os.path.join(out, os.path.basename(out)) + ".tex"

    dendrogram = os.path.join(out, "dendrogram.pdf")
    cl.plot(dendrogram, n_i)

    contact_rep_path = os.path.join(out, "contact_rep.pdf")
    contact_rep(multi_feature, cluster_file, cl.macrotraj[:, n_i], contact_rep_path, (4, 3))

    sankey_path = os.path.join(out, "sankey.pdf")

    # header
    # - Title 
    label = f"ana:{os.path.basename(out)}"

    lagtime = f"Lagtime: \\SI{{{cl.tlag*frame_length}}}{{\\nano\\second}}"
    traj_length = f"Traj length: \\SI{{{cl.traj.shape[0]*frame_length*1e-3:.0f}}}{{\\micro\\second}}"

    title = f"Clustering"
    kernel = f"\\verb|{cl.kernel}|"
    kl = f"KL = {cl.kernel.kullback_leibler}"
    thr = f"$\\mathrm{{pop}}_\\mathrm{{min}}={cl.pop_thr}$, $q_\\mathrm{{min}}={cl.q_min}$"
    if cl.kernel.method == "n":
        mode = f"n={cl.kernel.param}, c=\\SI{{{cl.kernel.c*100:.0f}}}{{\\percent}}"
    elif cl.kernel.method == "p":
        mode = f"p=\\SI{{{cl.kernel.param*100:.0f}}}{{\\percent}}, c=\\SI{{{cl.kernel.c*100:.0f}}}{{\\percent}}"
    else:
        mode = ""
    runs = f"Run {n_i}"
    its = f"rel its: ${cl.timescales[n_i, 0] / ref.timescales[0, 0]:.2f}\\cdot t_\\mathrm{{ref}}$"
    thresholds = f"pop: \\SI{{{cl.pop_thr*100:.2f}}}{{\\percent}} $q_\\mathrm{{min}}$={cl.q_min}"

    if cl.feature_kernel:
        feature_kernel = f"\\verb|{cl.feature_kernel}|"
        feature_params = f"$\\sigma$={cl.feature_kernel.sigma}, b={cl.feature_kernel.b}"
    else:
        feature_kernel = "No feature\\hspace{2cm}"
        feature_params = ""

    header = f"""
\\begin{{analysis}}
\\label{{{label}}}
\\vspace{{-0.6cm}}
\\begin{{table}}[H]
\\centering
\\begin{{tabular}}{{lll}}
    General & Clustering & Feature \\\\\\midrule
    {lagtime} & {thr} & {feature_kernel} \\\\
    {traj_length} & {mode} & {feature_params} \\\\
    & {kl} & \\\\
    & {its} & \\\\
    & {runs} & \\\\
\\end{{tabular}}
\\end{{table}}
\\vspace{{-2.0cm}}

\\begin{{figure}}[H]
\\centering
\\includegraphics[width=0.48\\textwidth]{{{os.path.abspath(dendrogram)}}}
\\includegraphics[width=0.48\\textwidth]{{{os.path.abspath(contact_rep_path)}}}
\\end{{figure}}
\\vspace{{-0.6cm}}
\\end{{analysis}}
"""

    with open(tex, "w") as f:
        f.write(header)


# \\begin{{table}}[H]
#     \\centering
#     \\begin{{tabular}}{{ll}}
#         \\includegraphics[width=0.4\\textwidth]{{{os.path.abspath(sankey_path)}}} &
#         \\begin{{minipage}}{{0.58\\textwidth}}
#             \\vspace{{-0.45\\textheight}}
#             \\begin{{minipage}}{{0.38\\textwidth}}
#                 \\vspace{{-1.95cm}}
#                 \\begin{{itemize}}
#                     \\setlength\\itemsep{{0.0cm}}
#                     \\item[$t$:] ${cl.timescales[0, 0]:.0f} | {cl.timescales[0, 0] / ref.timescales[0, 0]:.2f}$
#                     \\item[$H$:] ${cl.shannon_entropy[0]:.2f} | {cl.shannon_entropy[0] / ref.shannon_entropy[0]:.2f}$
#                     \\item[$b$] = {cl.kernel.b:.3f}
#                     \\item[$c$] = {cl.kernel.c:.3f}
#                 \\end{{itemize}}
#             \\end{{minipage}}
#             \\begin{{minipage}}{{0.08\\textwidth}}
#             \\end{{minipage}}
#             \\begin{{minipage}}{{0.55\\textwidth}}
#                 \\begin{{itemize}}
#                     \\setlength\\itemsep{{0.0cm}}
#                     \\item[DBI:] ${cl.davies_bouldin_index(multi_feature)[0]:.2f} | {cl.davies_bouldin_index(multi_feature)[0] / ref.davies_bouldin_index(multi_feature)[0]:.2f}$
#                     \\item[GMRQ:] ${cl.gmrq[0]:.2f} | {cl.gmrq[0] / ref.gmrq[0]:.2f}$
#                     \\item[T:] Transition probabilities
#                     \\item[$T_\\mathrm{{{cl.kernel.similarity}}}$:] Transition probability similarity
#                     \\item[$F_\\mathrm{{{cl.kernel.similarity}}}$:] Feature similarity
#                 \\end{{itemize}}
#             \\end{{minipage}}
#             \\begin{{minipage}}{{\\textwidth}}
#                 \\vspace{{0.5cm}}
#                 $ P = T + b \\cdot T_\\mathrm{{{cl.kernel.similarity}}} + c \\cdot F_\\mathrm{{{cl.kernel.similarity}}} $
#             \\end{{minipage}}
#         \\end{{minipage}}
#     \\end{{tabular}}
# \\end{{table}}
