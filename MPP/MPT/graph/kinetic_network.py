import msmhelper as mh
import networkx as nx
import numpy as np
import math
import prettypyplot as pplt
from .curved_edges import curved_edges
from fa2_modified import ForceAtlas2
from scipy.spatial import distance_matrix
from matplotlib import pyplot as plt
from matplotlib.colors import to_hex
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize

pplt.use_style(figsize=1.8, figratio=1)

USE_FA2 = True
DRAW_FLUX = True


def assign_color(qoft, states, traj, levels):
    states_qoft = np.array([1 - np.mean(qoft[traj == state]) for state in states])
    norm = Normalize(vmin=states_qoft.min(), vmax=states_qoft.max())
    states_bin = np.array([_bin(q, levels) for q in states_qoft])
    states_bin = np.array([_bin(norm(q), levels) for q in states_qoft])
    colors_list = [_color(q_bin, levels) for q_bin in states_bin]
    return colors_list


def _color(val, levels):
    cmap = plt.get_cmap("plasma", levels)
    return to_hex(
        cmap(val),
    )


def _bin(val, levels):
    # get bin
    bins = np.linspace(0, 1, levels + 1)

    for rlower, rhigher in zip(bins[:-1], bins[1:]):
        if rlower <= val <= rhigher:
            return rlower

    return bins[-1]


def get_luminance(hex_color):
    color = hex_color[1:]
    hex_red = int(color[0:2], base=16)
    hex_green = int(color[2:4], base=16)
    hex_blue = int(color[4:6], base=16)
    return hex_red * 0.2126 + hex_green * 0.7152 + hex_blue * 0.0722


def draw_knetwork(traj, tlag, qoft, out, u=0, f=0, set_min_node_size=True):
    _, ax = plt.subplots()
    tmat, states = mh.msm.estimate_markov_model(traj, tlag)
    n_nodes = len(np.unique(states))
    color_list = assign_color(qoft, states, traj, levels=10)

    # get detailed balance
    pop_eq = mh.msm.equilibrium_population(tmat)
    mat = tmat * pop_eq[:, np.newaxis]
    mat = 0.5 * (mat + mat.T)

    # prepare mats for networkx
    mat[np.diag_indices_from(mat)] = 0
    mat[mat < 2e-5] = 0

    # node size
    node_size = 1000 * np.log(pop_eq + 1)

    # set minimum node size
    if set_min_node_size:
        node_size = np.where(
            node_size < (np.min(node_size) + np.max(node_size)) / 2,
            0.7 * (np.min(node_size) + np.max(node_size)) / 2,
            node_size,
        )

    graph = nx.from_numpy_array(mat, create_using=nx.Graph)

    # get position
    # initial guess of simple spring model
    pos = nx.spring_layout(
        graph,
        fixed=None,
        iterations=1000,
        threshold=1e-4,
        scale=0.1,
        weight="weight",
    )
    if USE_FA2:
        # improve pos by forceatlas2
        forceatlas2 = ForceAtlas2(
            adjustSizes=False,
            verbose=False,
            strongGravityMode=True,
            scalingRatio=1000,
            gravity=0.0,
        )

        pos = forceatlas2.forceatlas2_networkx_layout(
            graph,
            pos=pos,
            iterations=1000,
        )
        coords2D = np.asarray(list(pos.values()))

        # rotate network so that the folded basin - native basin axis is parallel to the x axis
        if u != 0 and f != 0:
            coords_u = coords2D[u, :]
            coords_f = coords2D[f, :]
            a = np.mean(coords_u[:, 0]) - np.mean(coords_f[:, 0])
            b = np.mean(coords_u[:, 1]) - np.mean(coords_f[:, 1])
            theta = math.atan2(b, a)
        else:
            theta = 0
        rotated_coords = []
        for i in range(n_nodes):
            x = coords2D[i, 0] * math.cos(-theta) - coords2D[i, 1] * math.sin(-theta)
            y = coords2D[i, 0] * math.sin(-theta) + coords2D[i, 1] * math.cos(-theta)
            rotated_coords.append((x, y))
        coords2D = rotated_coords
        keys = list(pos.keys())
        pos = dict(zip(keys, coords2D))

    if DRAW_FLUX:
        edge_width = 0.1 + 300 * np.array(
            [graph[i][j]["weight"] for i, j in graph.edges],
        )
        curves = curved_edges(graph, pos)
        lc = LineCollection(
            curves,
            color="black",
            linewidth=edge_width,
            alpha=1,
        )
        ax.add_collection(lc)

    if not DRAW_FLUX:
        # create directed graph to draw edges
        digraph = nx.from_numpy_array(tmat, create_using=nx.DiGraph)
        edge_width = 0.2 + 5 * np.array(
            [digraph[i][j]["weight"] for i, j in digraph.edges],
        )
        nx.draw_networkx_edges(
            digraph,
            arrowstyle="-",
            pos=pos,
            connectionstyle="arc3,rad=0.4",
            width=edge_width,
            edge_color="black",
            node_size=node_size,
            arrowsize=3,
        )

    nx.draw_networkx_nodes(
        graph,
        pos=pos,
        node_color=color_list,
        node_size=node_size,
        linewidths=0.55,
        edgecolors="black",
    )
    # write node labels
    for node_idx, (x, y) in pos.items():
        luminance = get_luminance(color_list[node_idx])
        if luminance < 140 and set_min_node_size:
            c_text = "white"
        else:
            c_text = "black"
        pplt.text(
            x, y, states[node_idx] + 1, contour=False, fontsize="medium", color=c_text
        )
    # calc limits
    lims = np.array(
        [
            (
                x - max(node_size),
                x + max(node_size),
                y - max(node_size),
                y + max(node_size),
            )
            for n, (x, y) in pos.items()
        ]
    )
    ax.set_xlim(lims[:, 0].min(), lims[:, 1].max())
    ax.set_ylim(lims[:, 2].min(), lims[:, 3].max())

    ax.set_axis_off()
    plt.tight_layout()
    pplt.savefig(out)
