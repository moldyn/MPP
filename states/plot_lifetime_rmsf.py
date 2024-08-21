#!/usr/bin/env python3

import sys
from pathlib import Path
import argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib import gridspec

matplotlib.rcParams.update({'font.size': 16})

def parse_args():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Plot RMSF vs. mean frames in a row."
    )
    parser.add_argument(
        "rmsf_file",
        type=str,
        help="Path to the RMSF file."
    )
    parser.add_argument(
        "states_row",
        type=str,
        help="Path to the states in row file."
    )
    parser.add_argument(
        "-d",
        "--data-set",
        nargs=2,
        action="append",
        metavar=("RMSF_FILE", "STATES_ROW"),
        help="Add another pair of rmsf and row trajectory files.",
    )
    parser.add_argument(
        "-t",
        "--threshold",
        default=1000,
        type=int,
        help="Threshold for min count of frames per state."
    )
    parser.add_argument(
        "-s",
        "--save_plot",
        type=str,
        help="Save plot to file."
    )

    return parser.parse_args()

def prepare_data(rmsf, row, threshold=1000):
    # Only states with more than n counts
    rmsf_2k = rmsf[rmsf[:,1] > threshold]
    row_2k = row[np.isin(row[:, 0], rmsf_2k[:, 0])]
    rmsf_dict = {}
    for state in row_2k[:, 0]:
        rmsf_dict[state] = rmsf_2k[np.where(rmsf_2k[:, 0] == state)[0]]

    rmsf_2k_ordered = np.array([rmsf_dict[s][0] for s in row_2k[:, 0]])
    x = row_2k[:, 3]
    y = rmsf_2k_ordered[:, 2]
    c = rmsf_2k_ordered[:, 1] / rmsf[:, 1].sum()
    return x, y, c


def scatter(ax, x, y, c, vlim):
    cmap = "cool"
    scatter = ax.scatter(
        x,
        y,
        c=c,
        cmap=cmap,
        norm=LogNorm(
            vmin=vlim[0],
            vmax=vlim[1],
        ),
    )
        #norm=LogNorm(),
    ax.set_ylabel("RMSF / A")
    return scatter

def plot(rmsf, row, threshold=1000, save_plot=False):
    fig, ax = plt.subplots(1, 1)
    ax.set_title(f"States with at least {threshold} frames")

    x, y, c = prepare_data(rmsf, row, threshold)
    scatter_var = scatter(ax, x, y, c, (min(c), max(c)))

    colorbar = plt.colorbar(scatter_var, orientation='vertical')
    colorbar.set_label('Fraction of trajectory')

    ax.set_xlabel("Average stability / frames")

    plt.tight_layout()
    if save_plot:
        plt.savefig(save_plot)
    plt.show()

def plot_multi(rmsf_list, row_list, titles, threshold=1000, save_plot=False):
    fig, axs = plt.subplots(
        len(rmsf_list),
        1,
        sharex=True,
        figsize=(12, len(rmsf_list)*2.5)
    )
    xs = []
    ys = []
    cs = []

    for rmsf, row in zip(rmsf_list, row_list):
        x, y, c = prepare_data(rmsf, row, threshold)
        xs.append(x)
        ys.append(y)
        cs.append(c)

    cs_flat = np.concatenate((cs)).flatten()
    vlim = cs_flat.min(), cs_flat.max()

    for ax, x, y, c, t in zip(axs, xs, ys, cs, titles):
        scatter_var = scatter(ax, x, y, c, vlim)
        ax.set_title(t)
    
    axs[-1].set_xlabel("Average stability / frames")

    plt.tight_layout()

    fig.subplots_adjust(right=0.85)

    cbar_ax = fig.add_axes([0.88, 0.15, 0.03, 0.7])
    colorbar = fig.colorbar(scatter_var, orientation='vertical', cax=cbar_ax)#, cax=cbar_ax)
    colorbar.set_label('Fraction of trajectory')

    if save_plot:
        plt.savefig(save_plot)
    plt.show()

def main():
    args = parse_args()

    titles = [Path(args.rmsf_file).stem]

    rmsf = np.loadtxt(args.rmsf_file)
    row = np.loadtxt(args.states_row)[:len(rmsf)]

    if args.threshold:
        thr = args.threshold
    else:
        thr = int(len(rmsf) / 1000)


    # Plot states
    if args.data_set:
        rmsf_list = [rmsf]
        row_list = [row]
        for rmsf_file, row_file in args.data_set:
            # Read row file and trajectory
            rmsf_list.append(np.loadtxt(rmsf_file))
            row_list.append(np.loadtxt(row_file))
            titles.append(Path(rmsf_file).stem)

        plot_multi(rmsf_list, row_list, titles, thr, args.save_plot)
    else:
        plot(rmsf, row, thr, args.save_plot)

if __name__ == "__main__":
    main()
