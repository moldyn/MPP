#!/usr/bin/env python3
"""
plot_dendrogram.py
==================

Plot dendrogram from linkage and qmin. Most of the code originates from
procss_mpp.py from Daniel Nagel.
"""

from functools import lru_cache
import numpy as np
import prettypyplot as pplt
import msmhelper as mh
from matplotlib import pyplot as plt
from matplotlib import ticker
from matplotlib.cm import ScalarMappable
from matplotlib.collections import LineCollection
from matplotlib.colors import to_hex, Normalize, LinearSegmentedColormap
from scipy.cluster.hierarchy import dendrogram

MIN_EDGE_WEIGHT = 6 * 0.025

def plot_dendrogram_mpt(
    mpt,
    output_file: str,
    hide_labels: bool = True,
    color_threshold: float = 1.0,
):
    # setup matplotlib
    pplt.use_style(figsize=2.6, figratio='golden', true_black=True)
    
    # microstates that are included
    merge_states = []
    cum_states = {}
    for o, t in mpt.full_linkage:
        # cummulate states
        if t in cum_states:
            cum_states[t].add(o)
        else:
            cum_states[t] = {o, t}
        merge_states.append(cum_states[t])


    macrostates = np.unique(mpt.state_map[:, 1])
    macrostates_assignment = np.zeros((len(macrostates), mpt.n_states))
#    macrostates_assignment[mpt.state_map[:, 1]-1, mpt.state_map[:, 0]-1] = 1
    for s in macrostates:
        macrostates_assignment[s-1, mpt.state_map[:, 1] == s] = 1

    edge_widths = {
        state: 6 * pop for state, pop in enumerate(list(mpt.pop) + list(mpt.merge_pop))
    }

    linkage_mat = mpt.Z
    labels = [""] * mpt.n_states

    # NOTE:
    # Original q_state is dict {state_idx: q_mean} for all states
    feature_state_merge = _feature_in_state_merge(merge_states, mpt.features["fnc"], mpt.traj)
    feature_state = _feature_in_state_merge(np.expand_dims(mpt.states, -1), mpt.features["fnc"], mpt.traj)
    feature_merge = feature_state + feature_state_merge

    # NOTE:
    # Probably define colors for all state (microstates and merged states)
    #
    # define colors
    colors = {
        idx_state: _color_by_feature(feature_merge[idx_state])
        for idx_state in range(len(feature_merge))
    }
    # add global value
    colors[2 * (mpt.n_states - 1)] = _color_by_feature(1.0)

    fig, (ax, ax_mat) = plt.subplots(
        2,
        1,
        gridspec_kw={
            'hspace': 0.05 if hide_labels else 0.3,
            'height_ratios': [9, 1],
        },
    )
    # hide spines of lower mat
    for key, spine in ax_mat.spines.items():
        spine.set_visible(False)

    dendrogram_dict = _dendrogram(
        ax=ax,
        linkage_mat=linkage_mat,
        colors=colors,
        threshold=color_threshold,
        labels=labels,
        qmin=0,
        edge_widths=edge_widths,
    )

    # permute macrostate assignment and label them
    macrostates_assignment = macrostates_assignment.T[
        dendrogram_dict['leaves']
    ].T

    # plot legend
    cmap, bins = _color_by_q(None)
    norm = Normalize(bins[0], bins[-1])
    label = r'$\langle Q \rangle_\text{state} $'

    cmappable = ScalarMappable(norm, cmap)
    plt.sca(ax)
    pplt.colorbar(cmappable, width='5%', label=label, position='top')

    yticks = np.arange(0.5, 1.5 + len(macrostates))
    xticks = 10 * np.arange(0, mpt.n_states + 1)
    cmap = LinearSegmentedColormap.from_list(
        'binary', [(0, 0, 0, 0), (0, 0, 0, 1)],
    )

    xvals = 0.5 * (xticks[:-1] + xticks[1:])
    for idx, assignment in enumerate(macrostates_assignment):
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
        macrostates_assignment,
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

    ax_mat.set_xticks(np.arange(0.5, 0.5 + mpt.n_states))

    if hide_labels:
        for axes in (ax, ax_mat):  # if statemat_file else [ax]:
            axes.set_xticks([])
            axes.set_xticks([], minor=True)
            axes.set_xticklabels([])
            axes.set_xticklabels([], minor=True)

    pplt.savefig(f'{output_file}.pdf')

    return dendrogram_dict












def _feature_in_state_merge(states: list[list], feature_traj: np.ndarray, traj: np.ndarray) -> list:
    """Mean feature per state. Each state is defined as list of microstates"""
    feature_states = []
    for s in states:
        feature_states.append(feature_traj[np.isin(traj, list(s))].mean())
    return feature_states

def _color_by_feature(q, qmax=1, qmin=0, steps=10):
    cmap = plt.get_cmap('plasma_r', steps)
    colors = [cmap(idx) for idx in range(cmap.N)]

    bins = np.linspace(
        qmin, qmax, steps + 1,
    )

    if q is None:
        return cmap, bins

    for color, rlower, rhigher in zip(colors, bins[:-1], bins[1:]):
        if rlower <= q <= rhigher:
            return color
    return 'k'









def _color_by_q(q, qmax=1, qmin=0, steps=10):
    cmap = plt.get_cmap('plasma_r', steps)
    colors = [cmap(idx) for idx in range(cmap.N)]

    bins = np.linspace(
        qmin, qmax, steps + 1,
    )

    if q is None:
        return cmap, bins

    for color, rlower, rhigher in zip(colors, bins[:-1], bins[1:]):
        if rlower <= q <= rhigher:
            return color
    return 'k'


def _fraction_of_native_contacts(states, q_of_t, traj):
    """Mean fraction of native contacts per state."""
    if len(states):
        mask = np.full(q_of_t.shape[0], False)
        for state in states:
            mask = np.logical_or(
                mask,
                traj == state,
            )
        cs = q_of_t[mask]
    else:
        cs = q_of_t
    return np.mean(cs)


def _dendrogram(
    *, ax, linkage_mat, colors, threshold, labels, qmin, edge_widths,
):
    nstates = len(linkage_mat) + 1
    # convert color dictionary to array
    colors_arr = np.array(
        [
            to_hex(colors[state]) for state in range(2 * nstates - 1)
        ],
        dtype='<U7',
    )

    dendrogram_dict = dendrogram(
        linkage_mat,
        leaf_rotation=90,
        get_leaves=True,
        color_threshold=1,
        link_color_func=lambda state_idx: colors_arr[state_idx],
        no_plot=True,
    )
    _plot_dendrogram(
        icoords=dendrogram_dict['icoord'],
        dcoords=dendrogram_dict['dcoord'],
        ivl=dendrogram_dict['ivl'],
        color_list=dendrogram_dict['color_list'],
        threshold=threshold,
        ax=ax,
        colors=colors_arr,
        labels=labels,
        qmin=qmin,
        edge_widths=edge_widths,
    )

    ax.set_ylabel(r'metastability $Q_\text{min}$')
    ax.set_xlabel('microstates')
    ax.grid(visible=False, axis='x')

    return dendrogram_dict


def _show_xlabels(*, ax, states_perm):
    """Show the xticks together with the corresponding state names."""
    # undo changes of scipy dendrogram
    xticks = ax.get_xticks()
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    for line in ax.get_xticklines():
        line.set_visible(True)

    for is_major, length_scale in ((True, 4), (False, 1)):
        ax.tick_params(
            axis='x',
            length=length_scale * plt.rcParams['xtick.major.size'],
            labelrotation=90,
            pad=2,
            labelsize='xx-small',
            width=plt.rcParams['xtick.major.width'],
            which='major' if is_major else 'minor',
            top=False,
        )
        offset = 0 if is_major else 1
        ax.set_xticks(xticks[offset::2], minor=not is_major)
        ax.set_xticklabels(states_perm[offset::2], minor=not is_major)


def _plot_dendrogram(
    *,
    icoords,
    dcoords,
    ivl,
    color_list,
    threshold,
    ax,
    colors,
    labels,
    qmin,
    edge_widths,
):
    """Plot dendrogram with colors at merging points."""
    threshold_color = to_hex('pplt:grey')
    # Independent variable plot width
    ivw = len(ivl) * 10
    # Dependent variable plot height
    dvw = 1.05

    iv_ticks = np.arange(5, len(ivl) * 10 + 5, 10)

    ax.set_ylim([qmin, dvw])
    ax.set_xlim([-0.005 * ivw, 1.005 * ivw])
    ax.set_xticks(iv_ticks)

    ax.xaxis.set_ticks_position('bottom')
    ax.set_xticklabels(np.asarray(labels)[np.asarray(ivl).astype(int)])

    get_ancestor = _get_ancestor_func(
        icoords, dcoords, ivl,
    )

    # Let's use collections instead. This way there is a separate legend item
    # for each tree grouping, rather than stupidly one for each line segment.
    colors_used = np.unique(colors)
    color_to_lines = {color: [] for color in (*colors_used, threshold_color)}
    width_to_lines = {color: [] for color in (*colors_used, threshold_color)}
    for xline, yline, color in zip(icoords, dcoords, color_list):
        if np.max(yline) <= threshold:
            # split into left and right to color separately
            xline_l = [*xline[:2], np.mean(xline[1:3])]
            xline_r = [np.mean(xline[1:3]), *xline[2:]]

            color_l = _get_ancestor_color(
                icoords,
                dcoords,
                xline[0],
                yline[0],
                color_list,
                ivl,
                colors,
            )
            ancestors_l = get_ancestor(xline[0], yline[0])
            weight_l = np.sum([
                edge_widths[ancestor] for ancestor in ancestors_l
            ])
            color_r = _get_ancestor_color(
                icoords,
                dcoords,
                xline[3],
                yline[3],
                color_list,
                ivl,
                colors,
            )
            ancestors_r = get_ancestor(xline[3], yline[3])
            weight_r = np.sum([
                edge_widths[ancestor] for ancestor in ancestors_r
            ])
            color_to_lines[color_l].append(list(zip(xline_l, yline[:3])))
            width_to_lines[color_l].append(
                max(weight_l, MIN_EDGE_WEIGHT),
            )
            color_to_lines[color_r].append(list(zip(xline_r, yline[1:])))
            width_to_lines[color_r].append(
                max(weight_r, MIN_EDGE_WEIGHT),
            )

        elif np.min(yline) >= threshold:
            color_to_lines[threshold_color].append(list(zip(xline, yline)))
        else:
            yline_bl = [yline[0], np.max([threshold, yline[1]])]
            yline_br = [np.max([threshold, yline[2]]), yline[3]]
            color_to_lines[color].append(list(zip(xline[:2], yline_bl)))
            color_to_lines[color].append(list(zip(xline[2:], yline_br)))

            yline_thr = np.where(np.array(yline) < threshold, threshold, yline)
            color_to_lines[threshold_color].append(list(zip(xline, yline_thr)))

    # Construct the collections.
    colors_to_collections = {
        color: LineCollection(
            color_to_lines[color], colors=(color,),
            linewidths=width_to_lines[color],
        )
        for color in (*colors_used, threshold_color)
    }

    # Add all the groupings below the color threshold.
    for color in colors_used:
        ax.add_collection(colors_to_collections[color])
    # If there's a grouping of links above the color threshold, it goes last.
    ax.add_collection(colors_to_collections[threshold_color])


def _get_ancestor_color(
    xlines, ylines, xval, yval, color_list, ivl, colors,
):
    """Get the color of the ancestors."""
    # if ancestor is root
    if not yval:
        ancestor = int(ivl[int((xval - 5) // 10)])
        return colors[ancestor]

    # find ancestor color
    xy_idx = np.argwhere(
        np.logical_and(
            np.array(ylines)[:, 1] == yval,
            np.array(xlines)[:, 1:3].mean(axis=1) == xval,
        ),
    )[0][0]
    return color_list[xy_idx]


def _get_ancestor_func(
    xlines, ylines, ivl,
):
    """Get the color of the ancestors."""
    @lru_cache(maxsize=1024)
    def _get_ancestor_rec(xval, yval):
        # if ancestor is root
        if not yval:
            ancestor = int(ivl[int((xval - 5) // 10)])
            return (ancestor, )

        # find ancestor color
        xy_idx = np.argwhere(
            np.logical_and(
                np.array(ylines)[:, 1] == yval,
                np.array(xlines)[:, 1:3].mean(axis=1) == xval,
            ),
        )[0][0]
        xleft, yleft = xlines[xy_idx][0], ylines[xy_idx][0]
        xright, yright = xlines[xy_idx][3], ylines[xy_idx][3]

        return (
            *_get_ancestor_rec(xleft, yleft),
            *_get_ancestor_rec(xright, yright),
        )

    return _get_ancestor_rec







def main():
    pass

if __name__ == "__main__":
    main()
