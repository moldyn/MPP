#!/usr/bin/env python3
"""
plot.py
==================

Various plot functions used in this package.
"""

from os.path import splitext

import numpy as np
import prettypyplot as pplt
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import (
    LinearSegmentedColormap,
    LogNorm,
    ListedColormap,
)
from matplotlib import colors
from matplotlib.cbook import boxplot_stats
import matplotlib.patches as patches
from matplotlib.ticker import MultipleLocator
import msmhelper as mh
from msmhelper._cli.contact_rep import load_clusters

from . import utils
from .sankey_gap import sankey
from .graph import draw_knetwork

plt.rcParams["font.family"] = "sans-serif"

### DENDROGRAM ###############################################################


def plot_tree(root, macrostate_assignment, output_file, scale=1, offset=0):
    """
    Plot the dendrogram from a given state tree of BinaryTreeNode.
    """
    n_states = len(root.leaves)

    # setup matplotlib
    pplt.use_style(figsize=3.2 * scale, figratio="golden", true_black=True)
    plt.rcParams["font.family"] = "sans-serif"

    fig, (ax, ax_mat) = plt.subplots(
        2,
        1,
        gridspec_kw={
            "hspace": 0.05,
            "height_ratios": [9, 1],
        },
    )
    for key, spine in ax_mat.spines.items():
        spine.set_visible(False)

    ax = root.plot_tree(ax)

    ax.set_ylabel(r"Metastability $T_{ii}$")
    ax.set_xlabel("microstates")
    ax.set_xlim(-0.005 * n_states, 1.005 * n_states)
    ax.set_ylim(offset, 1.05)

    # plot legend
    cmap = plt.get_cmap("plasma_r", 10)
    label = r"Fraction of Contacts $q$"

    cmappable = ScalarMappable(root.feature_norm, cmap)
    plt.sca(ax)
    pplt.colorbar(cmappable, width="5%", label=label, position="top")

    # bring microstates in the right order
    macrostate_assignment = macrostate_assignment[:, [l.name for l in root.leaves]]

    yticks = np.arange(0.5, 1.5 + macrostate_assignment.shape[0])
    xticks = np.arange(0, n_states + 1)
    cmap = LinearSegmentedColormap.from_list(
        "binary",
        [(0, 0, 0, 0), (0, 0, 0, 1)],
    )

    xvals = 0.5 * (xticks[:-1] + xticks[1:])
    for idx, assignment in enumerate(macrostate_assignment):
        xmean = np.median(xvals[assignment == 1])

        pplt.text(
            xmean,
            yticks[idx] - (yticks[1] - yticks[0]),
            f"{idx + 1:.0f}",
            ax=ax_mat,
            va="top",
            contour=True,
            size="small",
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
    ax_mat.grid(visible=True, axis="y", ls="-", lw=0.5)
    ax_mat.tick_params(axis="y", length=0, width=0)
    ax_mat.set_xlim(ax.get_xlim())
    ax.set_xlabel("")
    ax_mat.set_xlabel("Macrostates")
    ax_mat.set_ylabel("")
    fig.align_ylabels([ax, ax_mat])

    ax_mat.set_xticks(np.arange(0.5, 0.5 + n_states))

    # Hide microstate labels
    for axes in (ax, ax_mat):
        axes.set_xticks([])
        axes.set_xticks([], minor=True)
        axes.set_xticklabels([])
        axes.set_xticklabels([], minor=True)

    pplt.savefig(output_file)
    plt.close()


### SIMILARITY ###############################################################


def stochastic_state_similarity(mpt1, mpt2, out):
    """
    Plot similarity values for a reference and a stochastic clustering.
    """
    ref, sto, S = mpt1 + mpt2
    s1, s2, s3 = S
    n_states = S.shape[1]
    x, y = utils.get_grid_format(n_states)
    fig, axs = plt.subplots(y, x, figsize=(2 * x, 2 * y))
    for state, ax in enumerate(axs.flatten()[:n_states]):
        m = 0
        # Set left limit to minimum instead of 0
        m = min([min(s1[state]), min(s2[state]), min(s3[state])]) - 0.02

        ax.hist(s1[state], bins=np.linspace(m, 1, 21), color="g", alpha=0.7)
        ax.hist(s2[state], bins=np.linspace(m, 1, 21), color="b", alpha=0.7)
        ax.hist(s3[state], bins=np.linspace(m, 1, 21), color="r", alpha=0.7)
        ax.set_title(f"state {state + 1}")
    fig.supxlabel("Macrostate similarity")
    fig.supylabel(f"Count of clusterings ({sto.n_runs} clusterings)")
    leg = plt.figlegend(
        ["union", "reference", "clustering"],
        ncols=3,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.05),
    )
    plt.tight_layout(rect=(0, 0.04, 1, 1))
    plt.savefig(out)
    plt.close()


### IMPLIED TIMESCALES #######################################################


def implied_timescales(
    trajectorys,
    lagtimes,
    out,
    titles="",
    frame_length=0.2,
    first_ref=False,
    scale=1,
    use_ref=True,
    ntimescales=3,
):
    """
    frame_length in ns / frame
    """
    if first_ref:
        ref_trajectory = trajectorys.pop(0)
    x, y = utils.get_grid_format(len(trajectorys))
    pplt.use_style(
        figsize=(2.7 * scale, 2.7 * scale), latex=False, colors="pastel_autumn"
    )
    fig, axs = plt.subplots(y, x, sharex=True, sharey=True)
    plt.grid(False)
    if not isinstance(axs, np.ndarray):
        axs = np.array([axs])

    if titles != "":
        titles = titles
    else:
        titles = [""] * len(trajectorys)

    min_it = None
    max_it = None

    if first_ref:
        it_ref = mh.msm.implied_timescales(
            ref_trajectory, lagtimes, ntimescales=ntimescales
        )
        # change from frames to ns
        it_ref *= frame_length
        min_it = it_ref.min()
        max_it = it_ref.max()

    lagtime = lagtimes[-1] / 4.5 * frame_length
    lagtimes_ns = lagtimes * frame_length
    for ax, traj, title in zip(axs.flatten(), trajectorys, titles):
        ax.axvline(lagtime, color="pplt:grid")
        it = mh.msm.implied_timescales(traj, lagtimes, ntimescales=ntimescales)
        # change from frames to ns
        it *= frame_length
        if min_it is None:
            min_it = it.min()
        else:
            min_it = min(it.min(), min_it)
        if max_it is None:
            max_it = it.max()
        else:
            max_it = max(it.max(), max_it)

        if first_ref:
            if not use_ref:
                _plot_impl_times(it_ref, lagtimes_ns, ax, ls="--")
            else:
                _plot_impl_times(it_ref, lagtimes_ns, ax, ls=":")
        _plot_impl_times(it, lagtimes_ns, ax)
        ax.set_yscale("log")
        ax.set_title(title)

    for ax in axs.flatten():
        # ax.set_ylim(min(min_it * 0.9, int(lagtimes_ns.shape[0] / 4)), max_it * 1.5)
        ax.set_ylim(min_it * 0.7, max_it * 1.5)

    if len(axs.shape) == 2:
        for ax in axs[-1]:
            ax.set_xlabel(r"lag time $\tau$ / ns")
        for axx in axs:
            for ax in axx[1:]:
                plt.setp(ax.get_yticklabels(), visible=False)
        for ax in axs[:, 0]:
            ax.set_ylabel("time scale / ns")
    elif len(axs.shape) == 1:
        axs[0].set_ylabel("time scale / ns")
        for ax in axs:
            ax.set_xlabel(r"lag time $\tau$ / ns")
        for ax in axs[1:]:
            plt.setp(ax.get_yticklabels(), visible=False)

    # Get handles and labels
    handles, labels = plt.gca().get_legend_handles_labels()

    # Reorder the handles and labels manually to achieve column-major ordering
    desired_order = np.array(
        [(i + ntimescales, i) for i in range(ntimescales)]
    ).flatten()
    handles = [handles[i] for i in desired_order]
    labels = [labels[i] for i in desired_order]

    pplt.legend(
        handles=handles, labels=labels, outside="top", frameon=False, ncols=ntimescales
    )

    plt.tight_layout()
    plt.savefig(out)
    plt.close()


def _plot_impl_times(impl_times, lagtimes, ax, ls="-"):
    """Plot the implied timescales"""
    colors = ["#264653", "#2A9D8F", "#E9C46A", "#f4a261", "#e76f51"] * 4
    for idx, impl_time in enumerate(impl_times.T):
        if ls == ":":
            label = f"$t_{{\\mathrm{{ref}},{idx + 1}}}$"
        elif ls == "--":
            label = f"$t_{{\\mathrm{{mic}},{idx + 1}}}$"
        else:
            label = f"$t_{idx + 1}$"
        ax.plot(lagtimes, impl_time, label=label, color=colors[idx], ls=ls)

    xlim = lagtimes[0], lagtimes[-1]
    ref_low = int(lagtimes.shape[0] / 9 * 2)
    # ref_low = int(lagtimes[-1] / 9 * 2)
    ax.set_xlim(xlim)
    # highlight diagonal
    x_i = np.arange(ref_low, xlim[1])
    ax.fill_between(x_i, x_i, color="pplt:grid")


def relative_implied_timescales(cl, out):
    pplt.use_style(figsize=(8, 2.5), latex=False, colors="pastel_autumn")

    ref = cl.reference
    its = cl.timescales / ref.timescales

    fig = plt.figure()
    ax1 = fig.add_subplot(1, 3, 1)
    ax2 = fig.add_subplot(1, 3, 2, sharey=ax1)
    ax3 = fig.add_subplot(1, 3, 3)

    for ax in (ax1, ax2, ax3):
        ax.grid(False)

    ax1.hist(its[:, 0], bins=20)
    ax1.set_title("its 1")
    ax1.set_xlabel(
        r"Relative Implied Timescale $\left(\frac{t_\mathrm{stoch}}{t_\mathrm{ref}}\right)$"
    )
    ax1.set_ylabel("Count of Clusterings")
    ax2.hist(its.mean(axis=1), bins=20)
    ax2.set_title(f"Mean its {1}-{3}")
    ax2.set_xlabel(
        r"Relative Implied Timescale $\left(\frac{t_\mathrm{stoch}}{t_\mathrm{ref}}\right)$"
    )

    bins = np.array(range(min(cl.n_macrostates) - 1, max(cl.n_macrostates) + 1)) + 0.5

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
    ax.set_aspect("equal", "box")

    ax.set_xticks(np.arange(a.shape[1]))
    ax.set_yticks(np.arange(a.shape[0]))
    if title:
        ax.set_title(title)
    ax.set_xlabel("Macrostate")
    ax.set_ylabel("Macrostate")
    plt.tight_layout()
    plt.savefig(out)
    plt.close()


def transition_matrix(a, out, title="Transition Matrix", color_thr=0.01):
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

    off_diag_norm = LogNorm(
        vmin=threshold * (1 - color_thr), vmax=off_diag_values.max()
    )

    # Create a custom colormap for off-diagonal values including light gray
    colors_list = plt.cm.viridis(np.linspace(0, 1, 256))
    gray = np.array([0.9, 0.9, 0.9, 1.0])
    colors_list[: int(color_thr * 256)] = gray
    custom_off_diag_cmap = colors.ListedColormap(colors_list)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect("equal", "box")
    ax.grid(False)

    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            value = a[i, j]
            if value == 0:
                color = (1, 1, 1, 1)  # Zero probabilities are white
            elif i == j:
                color = diag_cmap_custom(diag_norm(value))
            else:
                color = (
                    gray
                    if value < threshold
                    else custom_off_diag_cmap(off_diag_norm(value))
                )

            ax.add_patch(patches.Rectangle((j - 0.5, i - 0.5), 1, 1, color=color))

            # Add text with transition probabilities
            if value != 0:
                grayscale = np.sum(
                    np.array(color[:3]) * np.array([0.299, 0.587, 0.114])
                )
                text_color = "white" if grayscale < 0.5 else "black"
                ax.text(
                    j,
                    i,
                    f"{value:.2f}%",
                    ha="center",
                    va="center",
                    color=text_color,
                    fontsize=10,
                )

    ax.set_xticks(np.arange(a.shape[1]))
    ax.set_yticks(np.arange(a.shape[0]))
    ax.set_xticklabels(np.arange(1, a.shape[1] + 1))
    ax.set_yticklabels(np.arange(1, a.shape[0] + 1))
    ax.set_xlim(-0.5, a.shape[1] - 0.5)
    ax.set_ylim(-0.5, a.shape[0] - 0.5)

    # Add a colorbar for diagonal values
    cbar_diag = fig.colorbar(
        plt.cm.ScalarMappable(norm=diag_norm, cmap=diag_cmap), ax=ax, shrink=0.5
    )
    cbar_diag.set_label("Self Transition Probabilities / \\%")

    # Add a colorbar for off-diagonal values
    cbar_off_diag = fig.colorbar(
        plt.cm.ScalarMappable(norm=off_diag_norm, cmap=custom_off_diag_cmap),
        ax=ax,
        shrink=0.5,
    )
    cbar_off_diag.set_label("Transitiion Probabilities / \\%")

    if title:
        ax.set_title(title)

    ax.set_xlabel("From Macrostate")
    ax.set_ylabel("To Macrostate")
    plt.tight_layout()
    plt.savefig(out)
    plt.close()


def transition_time(
    a,
    out,
    lagtime=50.0,
    frame_length=0.2,
    title=r"Transition Times $\frac{t_\mathrm{lag}}{P}$",
    color_thr=0.01,
):
    """
    Plot heatmap from a matrix. This is supposed for a similarity matrix as
    returned from the multiplication of two MPT objects.
    frame_length in ns
    """
    with np.errstate(divide="ignore"):
        a = lagtime / a * frame_length

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

    # Threshold for light gray
    threshold = off_diag_values.min() / color_thr
    print(f"Threshold for probabilities: {threshold:.2f} ns")

    off_diag_norm = LogNorm(
        vmin=off_diag_values.min(), vmax=threshold / (1 - color_thr)
    )

    # Create a custom colormap for off-diagonal values including light gray
    colors_list = plt.cm.viridis_r(np.linspace(0, 1, 256))
    gray = np.array([0.9, 0.9, 0.9, 1.0])
    colors_list[int((1 - color_thr) * 256) :] = gray
    custom_off_diag_cmap = colors.ListedColormap(colors_list)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect("equal", "box")
    ax.grid(False)

    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            value = a[i, j]
            if value == np.inf:
                color = (1, 1, 1, 1)  # Zero probabilities are white
            elif i == j:
                color = diag_cmap_custom(diag_norm(value))
            else:
                color = (
                    gray
                    if value > threshold
                    else custom_off_diag_cmap(off_diag_norm(value))
                )

            ax.add_patch(patches.Rectangle((j - 0.5, i - 0.5), 1, 1, color=color))

            # Add text with transition probabilities
            if value != np.inf:
                grayscale = np.sum(
                    np.array(color[:3]) * np.array([0.299, 0.587, 0.114])
                )
                text_color = "white" if grayscale < 0.5 else "black"
                if value >= threshold:
                    pre_text = f"{value:.1g}"
                    text = pre_text[:2] + pre_text[-1]
                else:
                    if value >= 100:
                        text = f"{value:.0f}"
                    else:
                        text = f"{value:#.3g}"
                ax.text(
                    j, i, text, ha="center", va="center", color=text_color, fontsize=10
                )

    ax.set_xticks(np.arange(a.shape[1]))
    ax.set_yticks(np.arange(a.shape[0]))
    ax.set_xticklabels(np.arange(1, a.shape[1] + 1))
    ax.set_yticklabels(np.arange(1, a.shape[0] + 1))
    ax.set_xlim(-0.5, a.shape[1] - 0.5)
    ax.set_ylim(-0.5, a.shape[0] - 0.5)

    # Add a colorbar for diagonal values
    cbar_diag = fig.colorbar(
        plt.cm.ScalarMappable(norm=diag_norm, cmap=diag_cmap), ax=ax, shrink=0.5
    )
    cbar_diag.set_label("Self Transition Times / ns")

    # Add a colorbar for off-diagonal values
    cbar_off_diag = fig.colorbar(
        plt.cm.ScalarMappable(norm=off_diag_norm, cmap=custom_off_diag_cmap),
        ax=ax,
        shrink=0.5,
    )
    cbar_off_diag.set_label("Transitiion Times / ns")

    if title:
        ax.set_title(title)

    ax.set_xlabel("From Macrostate")
    ax.set_ylabel("To Macrostate")
    plt.tight_layout()
    plt.savefig(out)
    plt.close()


### MACROSTATE FEATURES ######################################################


def macro_feature(micro_feature, out, ref=None, pop=None):
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

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(bins[:-1], bins=bins, weights=norm_counts, label="Stochastic Clustering")
    if ref is not None:
        # for mas, mfs, c, l, w in ref:
        # add_ref(mas, mfs, ax, color=c, label=l, weights=w)
        add_ref(ref.macrostate_assignment[ref.n_i], ref.macrostate_feature[ref.n_i], ax)
    ax.set_xlabel("Fraction of Contacts")
    ax.set_ylabel("Population")
    ax.set_title(f"Macrostate Features, {micro_feature.shape[1]} clusterings")
    ax.set_yscale("log")
    ylim = ax.get_ylim()
    ax.set_ylim((y_min, ylim[1]))
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(out)
    plt.close()


def add_ref(
    macrostate_assignment,
    macrostate_feature,
    ax,
    color="r",
    label="Reference",
    weights=None,
):
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
            weights = np.array([1])
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
            size="small",
        )


### CONTACT REPRESENTATION ###################################################


def contact_rep(contacts, cluster_file, state_trajectory, output, grid, scale=1):
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
        figsize=1.2 * scale,
        colors="pastel_autumn",
        true_black=True,
        latex=False,
    )

    # load files
    states = np.unique(state_trajectory)
    clusters = load_clusters(cluster_file)

    contact_idxs = np.hstack(clusters)
    n_idxs = len(contact_idxs)
    n_frames = len(contacts)

    xtickpos = (
        np.cumsum(
            [
                0,
                *[len(clust) for clust in clusters[:-1]],
            ]
        )
        - 0.5
    )
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
            gridspec_kw={"wspace": wspace, "hspace": hspace},
        )

        # ignore outliers
        for state, ax in zip(chunk, axs.flatten()):
            contacts_state = contacts[state_trajectory == state]
            pop_state = len(contacts_state) / n_frames

            # get colormap
            c1, c2, c3 = pplt.categorical_color(3, "C0")

            stats = {
                idx: boxplot_stats(contacts_state[:, idx])[0] for idx in contact_idxs
            }

            for color, (key_low, key_high), label in (
                (c3, ("whislo", "whishi"), r"$Q_{1/3} \pm 1.5\mathrm{IQR}$"),
                (c2, ("q1", "q3"), r"$\mathrm{IQR} = Q_3 - Q_1$"),
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
                [stats[idx]["med"] for idx in contact_idxs],
                xmin=np.arange(n_idxs) - 0.5,
                xmax=np.arange(n_idxs) + 0.5,
                label="median",
                color=c1,
            )

            pplt.text(
                0.5,
                0.95,
                rf"S{state + 1} {pop_state:.1%}",
                ha="center",
                va="top",
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
                ax.axvline(pos, color="pplt:grid", lw=1.0)

        pplt.hide_empty_axes()
        pplt.legend(
            ax=axs[0, 0],
            outside="top",
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
            xlabel="contact clusters",
            ylabel="distances [nm]",
        )

        # save figure and continue
        if output is None:
            plt.show()
            # output = f"{state_file}.contactRep.pdf"
        # insert state_str between pathname and extension
        path, ext = splitext(output)
        if counter == 0:
            pplt.savefig(output)
            plt.close()
        else:
            pplt.savefig(f"{path}.state{chunk[0]:.0f}-{chunk[-1]:.0f}{ext}")
            plt.close()
        counter += 1


### SANKEY ###################################################################


def sankey_diagram(cl, ref, out, ax=None, scale=1):
    features = []
    for macrostate in cl.tree[cl.n_i].macrostates:
        features.append(macrostate.feature)
    ma_order = np.argsort(features)[::-1]
    colorDict = {}
    for i, o in enumerate(ma_order):
        colorDict[str(i + 1)] = cl.tree[cl.n_i].macrostates[o].color
    if ax is None:
        pplt.use_style(figsize=(1.7 * scale, 3.6 * scale), true_black=True)
    sankey(
        left=(cl.macrostate_map[cl.n_i] + 1).astype(str),
        right=(ref.macrostate_map[0] + 1).astype(str),
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


### RMSD LINES ###############################################################


def rmsd(rmsds, pops, helices=None, filename=None):
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
        raise ValueError(
            "All values in `rmsds` must be positive for logarithmic scaling."
        )

    if rmsds.shape[0] != len(pops):
        raise ValueError("Length of `pops` must match the number of rows in `rmsds`.")

    if helices is not None:
        n_plots = rmsds.shape[0] + 1
    else:
        n_plots = rmsds.shape[0]

    w = 0.08 * rmsds.shape[1] + 3  # 8.6
    h = 1 + 0.4 * n_plots  # 6
    pplt.use_style(
        figsize=(w, h),
        colors="pastel_autumn",
        true_black=True,
        latex=False,
    )
    fig, axs = plt.subplots(
        n_plots,
        3,
        sharex="col",
        width_ratios=[rmsds.shape[1], 8, 8],
        gridspec_kw={"wspace": 0, "hspace": 0},
    )

    ylim = 0.5 * rmsds.min(), 2 * rmsds.max()
    pops = pops / pops.sum()
    ylim_hist = 0, 1.05 * pops.max()

    rmsd_sums = rmsds[:, 2:-2].sum(axis=1)
    ylim_rmsd = 0, 1.05 * rmsd_sums.max()

    for i, ((ax, hist_ax, rmsd_ax), rmsd, pop) in enumerate(
        zip(axs[:-1] if helices is not None else axs, rmsds, pops)
    ):
        rect = patches.Rectangle(
            (0, 0.3),  # Position of the block
            pop,
            0.4,  # color='black'
        )
        hist_ax.add_patch(rect)
        hist_ax.set_xlim(ylim_hist)
        hist_ax.set_yticks([], [])
        hist_ax.grid(False)

        rect = patches.Rectangle(
            (0, 0.3),  # Position of the block
            rmsd[2:-2].sum(),
            0.4,  # color='black'
        )
        rmsd_ax.add_patch(rect)
        rmsd_ax.set_xlim(ylim_rmsd)
        rmsd_ax.set_yticks([], [])
        rmsd_ax.grid(False)

        ax.plot(np.arange(rmsd.shape[0]) + 1, rmsd)
        ax.fill_between(
            np.arange(rmsd.shape[0]) + 1,
            [ylim[0]] * rmsd.shape[0],
            rmsd,
            alpha=0.5,
            # facecolor="none",
            # hatch="/",
        )

        ax.set_yscale("log")
        ax.set_ylabel(f"{i + 1}")
        ax.set_xlim((0.5, rmsd.shape[0] + 0.5))
        ax.set_ylim(ylim)
        ax.grid(True)

    if helices is not None:
        line_start = 1
        helices_ax = axs[-1, 0]
        # helices_ax.plot([1, rmsds.shape[1]], [0.5, 0.5]) #, c="k")
        for start, end in helices:
            if start > 0:
                # Helices
                start -= 0.3
                end += 0.3
                rect = patches.Rectangle(
                    (start, 0.3),  # Position of the block
                    end - start,
                    0.4,  # color='black'
                    fc="#264653",
                    ec="#264653",
                    lw=2,
                )
            else:
                # Sheets
                start, end = -start, -end
                start -= 0.5
                end += 0.5
                rect = patches.Rectangle(
                    (start, 0.3),  # Position of the block
                    end - start,
                    0.4,  # color='black'
                    fc="white",
                    ec="#264653",
                    lw=2,
                )
            helices_ax.plot(
                [line_start, start],
                [0.5, 0.5],
                solid_capstyle="butt",
                c="#264653",
                lw=2,
            )
            line_start = end
            helices_ax.add_patch(rect)
        helices_ax.plot(
            [line_start, rmsds.shape[1]],
            [0.5, 0.5],
            solid_capstyle="butt",
            c="#264653",
            lw=2,
        )

        helices_ax.set_ylim((0, 1))
        helices_ax.set_ylabel("H")
        helices_ax.set_yticks([], [])
        helices_ax.grid(False)

        axs[-1, 1].grid(False)
        axs[-1, 1].set_yticks([], [])
        axs[-1, 2].grid(False)
        axs[-1, 2].set_yticks([], [])

    hist_ticks = axs[-1, 1].get_xticks()
    hist_labels = axs[-1, 1].get_xticklabels()
    hist_labels[0] = ""
    axs[-1, 1].set_xticks(hist_ticks, hist_labels)

    rmsd_ticks = axs[-1, 2].get_xticks()
    rmsd_labels = axs[-1, 2].get_xticklabels()
    rmsd_labels[0] = ""
    axs[-1, 2].set_xticks(rmsd_ticks, rmsd_labels)

    axs[-1, 0].xaxis.set_major_locator(MultipleLocator(5))
    axs[-1, 0].xaxis.set_minor_locator(MultipleLocator(1))
    axs[-1, 0].set_xlabel("Residue")
    axs[-1, 1].set_xlabel("Population", rotation=20)
    axs[-1, 2].set_xlabel(r"$\sum$ RMSD / nm", rotation=20)
    fig.supylabel("Macrostate; RMSD Variance / nm")

    # Save to file if filename is provided
    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=192)  # , bbox_inches="tight"
    else:
        plt.show()
    plt.close()


def delta_rmsd(rmsds, pops, helices=None, filename=None):
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
        raise ValueError(
            "All values in `rmsds` must be positive for logarithmic scaling."
        )

    if rmsds.shape[0] != len(pops):
        raise ValueError("Length of `pops` must match the number of rows in `rmsds`.")

    if helices is not None:
        n_plots = rmsds.shape[0] + 1
    else:
        n_plots = rmsds.shape[0]

    w = 0.08 * rmsds.shape[1] + 3  # 8.6
    h = 1 + 0.4 * n_plots  # 6
    pplt.use_style(
        figsize=(w, h),
        colors="pastel_autumn",
        true_black=True,
        latex=False,
    )
    fig, axs = plt.subplots(
        n_plots,
        3,
        sharex="col",
        width_ratios=[rmsds.shape[1], 8, 8],
        gridspec_kw={"wspace": 0, "hspace": 0},
    )

    delta_rmsd = rmsds
    # delta_rmsd[1:] = rmsds[1:] - rmsds[:-1]
    delta_rmsd[1:] = rmsds[1:] - rmsds[0]
    rmsds = delta_rmsd

    rmsd_max = rmsds.max()
    rmsd_min = rmsds.min()
    rmsd_delta = rmsd_max - rmsd_min

    ylim = rmsd_min - rmsd_delta * 0.1, rmsd_max + rmsd_delta * 0.15
    pops = pops / pops.sum()
    ylim_hist = 0, 1.05 * pops.max()

    rmsd_sums = rmsds.sum(axis=1)
    ylim_rmsd = 0, 1.05 * rmsd_sums.max()

    for i, ((ax, hist_ax, rmsd_ax), rmsd, pop) in enumerate(
        zip(axs[:-1] if helices is not None else axs, rmsds, pops)
    ):
        rect = patches.Rectangle(
            (0, 0.3),  # Position of the block
            pop,
            0.4,  # color='black'
        )
        hist_ax.add_patch(rect)
        hist_ax.set_xlim(ylim_hist)
        hist_ax.set_yticks([], [])
        hist_ax.grid(False)

        rect = patches.Rectangle(
            (0, 0.3),  # Position of the block
            abs(rmsd).sum(),
            0.4,  # color='black'
        )
        rmsd_ax.add_patch(rect)
        rmsd_ax.set_xlim(ylim_rmsd)
        rmsd_ax.set_yticks([], [])
        rmsd_ax.grid(False)

        ax.plot(np.arange(rmsd.shape[0]) + 1, rmsd)
        ax.fill_between(
            np.arange(rmsd.shape[0]) + 1,
            # [ylim[0]]*rmsd.shape[0],
            [0] * rmsd.shape[0],
            rmsd,
            alpha=0.5,
            # facecolor="none",
            # hatch="/",
        )

        # ax.set_yscale("log")
        ax.set_ylabel(f"{i + 1}")
        # ax.set_ylabel(f"{i+1}-{i}")
        # ax.set_ylabel(f"{i+1}-{i}", rotation=90, position=(-0.2, 0))
        ax.set_xlim((0.5, rmsd.shape[0] + 0.5))
        ax.set_ylim(ylim)
        ax.grid(True)

    if helices is not None:
        line_start = 1
        helices_ax = axs[-1, 0]
        # helices_ax.plot([1, rmsds.shape[1]], [0.5, 0.5]) #, c="k")
        for start, end in helices:
            if start > 0:
                # Helices
                start -= 0.3
                end += 0.3
                rect = patches.Rectangle(
                    (start, 0.3),  # Position of the block
                    end - start,
                    0.4,  # color='black'
                    fc="#264653",
                    ec="#264653",
                    lw=2,
                )
            else:
                # Sheets
                start, end = -start, -end
                start -= 0.5
                end += 0.5
                rect = patches.Rectangle(
                    (start, 0.3),  # Position of the block
                    end - start,
                    0.4,  # color='black'
                    fc="white",
                    ec="#264653",
                    lw=2,
                )
            helices_ax.plot(
                [line_start, start],
                [0.5, 0.5],
                solid_capstyle="butt",
                c="#264653",
                lw=2,
            )
            line_start = end
            helices_ax.add_patch(rect)
        helices_ax.plot(
            [line_start, rmsds.shape[1]],
            [0.5, 0.5],
            solid_capstyle="butt",
            c="#264653",
            lw=2,
        )

        helices_ax.set_ylim((0, 1))
        helices_ax.set_ylabel("H")
        helices_ax.set_yticks([], [])
        helices_ax.grid(False)

        axs[-1, 1].grid(False)
        axs[-1, 1].set_yticks([], [])
        axs[-1, 2].grid(False)
        axs[-1, 2].set_yticks([], [])

    axs[0, 0].set_ylim(0.5 * rmsds[0].min(), 2 * rmsds[0].max())
    axs[0, 0].set_yscale("log")
    # axs[0, 0].set_ylabel("1") #, rotation=90)

    hist_ticks = axs[-1, 1].get_xticks()
    hist_labels = axs[-1, 1].get_xticklabels()
    hist_labels[0] = ""
    axs[-1, 1].set_xticks(hist_ticks, hist_labels)

    rmsd_ticks = axs[-1, 2].get_xticks()
    rmsd_labels = axs[-1, 2].get_xticklabels()
    rmsd_labels[0] = ""
    axs[-1, 2].set_xticks(rmsd_ticks, rmsd_labels)

    axs[-1, 0].xaxis.set_major_locator(MultipleLocator(5))
    axs[-1, 0].xaxis.set_minor_locator(MultipleLocator(1))
    axs[-1, 0].set_xlabel("Residue")
    axs[-1, 1].set_xlabel("Population", rotation=20)
    axs[-1, 2].set_xlabel(r"$\sum$ |$\Delta$RMSD| / nm", rotation=20)
    fig.supylabel(r"$\Delta$Macrostate; $\Delta$RMSD Variance / nm wrt. 1st State")

    # Save to file if filename is provided
    plt.tight_layout()
    if filename:
        plt.savefig(filename, dpi=192)  # , bbox_inches="tight"
    else:
        plt.show()
    plt.close()


### TRAJECTORY ###############################################################


def state_trajectory(trajectory, filename, row_length=0.2, frame_length=0.2):
    """
    Plot state trajectory

    trajectory (np.ndarray): state trajectory
    filename (str): file name to save the plot to
    row_length (int|float):
        row_length > 1: number of frames in each row
        0 < row_length <= 1: fraction of total frames per row (1/n_rows)
    frame_length (float): frame length in ns
    """
    if row_length > 1:
        x_max = int(row_length)
    elif row_length > 0:
        x_max = int(np.ceil(trajectory.shape[0] * row_length))
    else:
        raise ValueError("row_lengthg must be > 0")

    frame_length /= 1000.0
    # Calculate unique states and their lengths
    unique_states, lengths = utils.find_state_lengths(trajectory)
    unique_states += 1
    lengths = lengths * frame_length
    n_rows = int(np.ceil(trajectory.shape[0] / x_max))

    x_max *= frame_length

    figsize = (11.7, 8.3)
    # # Set up figure size proportional to data
    # width = max(6, x_max * 0.0001)  # Minimum width of 6 inches
    # height = max(
    #     2, (unique_states.max() - unique_states.min() + 1) * 0.05 * n_rows + 0.6
    # )  # Minimum height of 4 inches
    # figsize = (width * 1.5, height * 1.5)

    u_states = np.unique(unique_states)
    cmap = mpl.cm.turbo
    norm = mpl.colors.BoundaryNorm(np.concatenate([[0], u_states]) + 0.5, cmap.N)

    fig, axs = plt.subplots(
        n_rows,
        1,
        sharex=True,
        figsize=figsize,
        gridspec_kw={"wspace": 0, "hspace": 0},
        squeeze=False,
        layout="compressed",
    )
    axs = axs[:, 0]
    axi = 0

    # Plot each state occurrence as a line segment
    x_start = 0  # Initial x-coordinate for the first segment
    for state, length in zip(unique_states, lengths):
        x_end = x_start + length  # Calculate end position of this segment on the x-axis
        color = cmap(norm(state))

        while x_end > x_max:
            axs[axi].plot(
                [x_start, x_max],
                [state, state],
                color=color[:3],
                linewidth=3,
                solid_capstyle="butt",
            )
            x_end -= x_max
            x_start = 0
            axi += 1
        if not np.isclose(x_start, x_end):
            axs[axi].plot(
                [x_start, x_end],
                [state, state],
                color=color,
                linewidth=3,
                solid_capstyle="butt",
            )

        # Move x_start to the end of the current segment for the next one
        x_start = x_end

    # Label axes and set title
    cbar = fig.colorbar(
        ScalarMappable(norm=norm, cmap=cmap),
        ax=axs,
        orientation="vertical",
        label="Macrostate",
    )
    cbar.set_ticks(ticks=u_states, labels=u_states, minor=False)
    fig.axes[-1].tick_params(length=0)

    fig.supylabel("State Index")
    axs[-1].set_xlabel(r"t / $\mu$s")

    for ax in axs:
        ax.set_ylim(unique_states.min() - 1, unique_states.max() + 1)

    # Set axis limits
    plt.xlim(0, x_max)

    # Save the plot to the specified file
    plt.savefig(filename)
    plt.close()  # Close the plot to free memory


### CHAPMAN-KOLMOGOROV TEST ##################################################


def chapman_kolmogorov(mpt, out, frame_length=0.2):
    """Chapman-Kolmogorov Test. Frame length in ns"""
    ck = mh.msm.tests.chapman_kolmogorov_test(
        utils.get_multi_state_trajectory(
            mpt.macrostate_trajectory[mpt.n_i], mpt.limits
        ),
        [50, 50, 50, 50, 50],
        4000,
        # int(1550*frame_length),
    )
    pplt.use_style(
        figsize=4.8,
        colors="pastel_autumn",
        true_black=True,
        latex=False,
    )

    nrows, ncols = utils.get_grid_format(mpt.n_macrostates[mpt.n_i])
    for chunk in mh.plot._ck_test._split_array(
        np.arange(mpt.n_macrostates[mpt.n_i]), nrows * ncols
    ):
        fig = plot_ck_test(
            ck=ck,
            states=chunk,
            frames_per_unit=1 / frame_length,
            unit="ns",
            grid=(ncols, nrows),
        )

    for ax in fig.axes:
        for text in ax.texts:
            text.set_position((0.15, 0.2))
    plt.savefig(out)
    plt.close()


def plot_ck_test(
    ck,
    states=None,
    frames_per_unit=1,
    unit="frames",
    grid=(3, 3),
):
    """Plot CK-Test results.

    This routine is a basic helper function to visualize the results of
    [msmhelper.msm.chapman_kolmogorov_test][].

    Parameters
    ----------
    ck : dict
        Dictionary holding for each lagtime the CK equation and with 'md' the
        reference.
    states : ndarray, optional
        List containing all states to plot the CK-test.
    frames_per_unit : float, optional
        Number of frames per given unit. This is used to scale the axis
        accordingly.
    unit : ['frames', 'fs', 'ps', 'ns', 'us'], optional
        Unit to use for label.
    grid : (int, int), optional
        The number of `(n_rows, n_cols)` to use for the grid layout.

    Returns
    -------
    fig : matplotlib.Figure
        Figure holding plots.

    Notes
    -----
    Adapted from msmhelper.

    """
    # load colors
    pplt.load_cmaps()
    pplt.load_colors()

    lagtimes = np.array([key for key in ck.keys() if key != "md"])
    if states is None:
        states = np.array(list(ck["md"]["ck"].keys()))

    nrows, ncols = grid
    needed_rows = int(np.ceil(len(states) / ncols))

    fig, axs = plt.subplots(
        needed_rows,
        ncols,
        sharex=True,
        sharey="row",
        gridspec_kw={"wspace": 0, "hspace": 0},
    )
    axs = np.atleast_2d(axs)

    max_time = np.max(ck["md"]["time"])
    for irow, states_row in enumerate(mh.plot._ck_test._split_array(states, ncols)):
        for icol, state in enumerate(states_row):
            ax = axs[irow, icol]

            pplt.plot(
                ck["md"]["time"] / frames_per_unit,
                ck["md"]["ck"][state],
                "--",
                ax=ax,
                color="pplt:gray",
                label="MD",
            )
            for lagtime in lagtimes:
                pplt.plot(
                    ck[lagtime]["time"] / frames_per_unit,
                    ck[lagtime]["ck"][state],
                    ax=ax,
                    label=lagtime / frames_per_unit,
                )
            pplt.text(
                0.5,
                0.9,
                "S{0}".format(state + 1),
                contour=True,
                va="top",
                transform=ax.transAxes,
                ax=ax,
            )

            # set scale
            ax.set_xscale("log")
            ax.set_xlim(
                [
                    lagtimes[0] / frames_per_unit,
                    max_time / frames_per_unit,
                ]
            )
            ax.set_ylim([0, 1])
            if irow < len(axs) - 1:
                ax.set_yticks([0.5, 1])
            else:
                ax.set_yticks([0, 0.5, 1])

            ax.grid(True, which="major", linestyle="--")
            ax.grid(True, which="minor", linestyle="dotted")
            ax.set_axisbelow(True)

    # set legend
    legend_kw = (
        {
            "outside": "right",
            "bbox_to_anchor": (2.0, (1 - nrows), 0.2, nrows),
        }
        if ncols in {1, 2}
        else {
            "outside": "top",
            "bbox_to_anchor": (0.0, 1.0, ncols, 0.01),
        }
    )
    if ncols == 3:
        legend_kw["ncol"] = 3
    pplt.legend(
        ax=axs[0, 0],
        **legend_kw,
        title=rf"$\tau_\mathrm{{lag}}$ [{unit}]",
        frameon=False,
    )

    ylabel = (
        (r"self-transition probability $P_{i\to i}$")
        if nrows >= 3
        else (r"$P_{i\to i}$")
    )

    pplt.hide_empty_axes()
    pplt.label_outer()
    pplt.subplot_labels(
        ylabel=ylabel,
        xlabel=r"time $t$ [{unit}]".format(unit=unit),
    )
    return fig


### MACROSTATE GRAPH #########################################################


def state_network(lumping, out):
    draw_knetwork(
        lumping.macrostate_trajectory[lumping.n_i],
        lumping.lagtime,
        lumping.mean_feature_trajectory,
        out,
    )
