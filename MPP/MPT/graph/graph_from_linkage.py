#!/usr/bin/env python3

import sys
import argparse
from collections import defaultdict
import numpy as np
import networkx as nx
import matplotlib
import matplotlib.pyplot as plt


def get_macrostates(macrostates_file):
    states = np.loadtxt(macrostates_file, dtype=int)
    return {micro: macro for micro, macro in states}

def main(args):
    linkage = np.loadtxt(args.linkage_file)
    linkage = [(int(i), int(j), k) for i, j, k in linkage]
    fig, ax = plt.subplots(1, 1, figsize=(16,9), dpi=192)
    graph = nx.Graph()
    graph.add_weighted_edges_from(linkage)
    name_cmap = "gist_rainbow"
    if args.input_macrostate:
        macrostate = get_macrostates(args.input_macrostate)
        cmap = plt.get_cmap(name_cmap, len({macrostate[n] for n in graph.nodes}))
        nx.draw(graph, with_labels=True, node_color=[cmap(macrostate[n]) for n in graph.nodes])
    else:
        cmap = plt.get_cmap(name_cmap, len(graph.nodes))
        nx.draw(graph, with_labels=True, node_color=[cmap(n) for n in graph.nodes])
    if args.save:
        plt.tight_layout()
        plt.savefig(args.save)
    else:
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="graph_from_linkage.py",
        description="Plot graph diagram from linkage file."
    )
    parser.add_argument(
        "linkage_file",
        help="Linkage file with transition probabilities",
        type=argparse.FileType('r', encoding='latin-1')
    )
    parser.add_argument(
        "-m",
        "--input-macrostate",
        type=argparse.FileType("r", encoding="latin-1"),
        help="Macrostate file from MPT",
    )
    parser.add_argument(
        "-s",
        "--save",
        help="Save diagram to file",
    )
    args = parser.parse_args()
    main(args)
