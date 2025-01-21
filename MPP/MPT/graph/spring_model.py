#!/usr/bin/env python3

import argparse
import json
import numpy as np
import networkx as nx
import msmhelper as mh
import matplotlib.pyplot as plt
import prettypyplot as pplt
from msmhelper.msm import row_normalize_matrix as normalize
from pathlib import Path

try:
    from fa2 import ForceAtlas2
except ModuleNotFoundError:
    try:
        from fa2_modified import ForceAtlas2
    except ModuleNotFoundError:
        raise ModuleNotFoundError(
            "Neither of fa2 or fa2_modified found. Please install ForceAtlas2."
        )

def parse_args():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Calculate a spring model graph from a transition matrix file"
        )
    )
    parser.add_argument(
        "trajectory",
        type=str,
        help="Path to state trajectory file"
    )
    parser.add_argument(
        "lagtime",
        type=int,
        help='Lag time to be used',
    )
    parser.add_argument(
        "output",
        type=str,
        help="Path to the output file"
    )
    parser.add_argument(
        "-l",
        "--linkage",
        type=str,
        help="Path to macrostates file"
    )
    parser.add_argument(
        "--store-seed",
        type=str,
        help="Store seed positions"
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=str,
        help="Use seed positions"
    )
    return parser.parse_args()

# Function to save the dictionary to a JSON file
def save_dict_to_json(filename, dictionary):
    # Ensure correct extension
    if not filename.endswith(".json"):
        filename += ".json"
    # Convert NumPy arrays to lists
    processed_dict = {key: value.tolist() if isinstance(value, np.ndarray) else value
                      for key, value in dictionary.items()}
    # Save to JSON file
    with open(filename, 'w') as json_file:
        json.dump(processed_dict, json_file)

# Function to load the dictionary from a JSON file
def load_dict_from_json(filename):
    with open(filename, 'r') as json_file:
        processed_dict = json.load(json_file)
    # Convert lists back to NumPy arrays
    original_dict = {int(key): np.array(value) if isinstance(value, list) else value
                     for key, value in processed_dict.items()}
    return original_dict

def draw_graph(traj, tlag, output, linkage=False, store_seed=False, seed=None):
    tmat, states = mh.msm.estimate_markov_model(traj, tlag)
    _, pop = np.unique(traj, return_counts=True)
    node_size = np.sqrt(pop) * 0.5

    _, ax = plt.subplots()
    graph = nx.from_numpy_array(tmat, create_using=nx.Graph)

    if not seed:
        # get position
        # initial guess of simple spring model
        pos = nx.spring_layout(
            graph, fixed=None, iterations=1000, threshold=1e-4, scale=1e3, weight='weight',
        )

        if store_seed:
            save_dict_to_json(store_seed, pos)
    else:
        pos = load_dict_from_json(seed)

    # improve pos by forceatlas2
    forceatlas2 = ForceAtlas2(
        adjustSizes=False, verbose=False, strongGravityMode=True,
    )

    name_cmap = "gist_rainbow"
    if linkage:
        merge = {micro: macro for micro, macro in linkage}
        cmap = plt.get_cmap(name_cmap, len(np.unique(linkage[:, 1])))
        color_list = [cmap(merge[i]) for i in states]
    else:
        cmap = plt.get_cmap(name_cmap, len(states))
        color_list = [cmap(i) for i in states]

    nx.draw_networkx_nodes(
        graph,
        pos=pos,
        node_color=color_list,
        node_size=node_size,
        linewidths=0.55,
        edgecolors='black'
    )

    # calc limits
    lims = np.array([
        (
            x - max(node_size),
            x + max(node_size),
            y - max(node_size),
            y + max(node_size),
        )
        for n, (x, y) in pos.items()
    ])
    ax.set_xlim(lims[:, 0].min(), lims[:, 1].max())
    ax.set_ylim(lims[:, 2].min(), lims[:, 3].max())

    ax.set_axis_off()
    pplt.savefig(output)

def main():
    args = parse_args()
    traj = np.loadtxt(args.trajectory, dtype=int, comments='#')
    if args.linkage:
        linkage = np.loadtxt(args.linkage, dtype=int, comments='#')
    else:
        linkage = False
    draw_graph(traj, args.lagtime, Path(args.output), linkage, args.store_seed, args.seed)

if __name__ == "__main__":
    main()
