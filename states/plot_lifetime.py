#!/usr/bin/env python3

import argparse
import numpy as np
import matplotlib.pyplot as plt 

def parse_args():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Plot row file."
    )
    parser.add_argument(
        "row_file",
        type=str,
        help="Path to the row file."
    )
    parser.add_argument(
        "state_traj",
        type=str,
        help="Path to the state trajectory file."
    )
    parser.add_argument(
        "-d",
        "--data-set",
        nargs=2,
        action="append",
        metavar=("ROW_FILE", "STATE_TRAJ"),
        help="Add another pair of row and state trajectory files.",
    )
    parser.add_argument(
        "-s",
        "--save_plot",
        type=str,
        help="Save plot to this file."
    )
    return parser.parse_args()

def count_states(state_traj):
    state_counts = {}
    for state, count in zip(*np.unique(state_traj, return_counts=True)):
        state_counts[state] = count
    return state_counts

def scatter(ax, states, state_counts):
    counts = [state_counts[s] for s in states[:, 0]]
    counts = counts / sum(counts)
    # Plot max
    ax.scatter(counts, states[:, 1], c="r", label="max")
    # Plot mean
    ax.scatter(counts, states[:, 3], c="g", label="mean")
    # Plot min
    ax.scatter(counts, states[:, 2], c="b", label="min")
    return ax

def plot(states, state_counts, save_plot=False):
    fig, ax = plt.subplots(1, 1)

    scatter(ax, states, state_counts)

    ax.set_xlabel("Fraction of frames in state")
    ax.set_ylabel("Frames in a row in the state")
    ax.legend()
    ax.set_xscale("log")
    ax.set_yscale("log")

    plt.tight_layout()
    if save_plot:
        plt.savefig(save_plot)
    plt.show()

def plot_multi(states_list, state_counts_list, save_plot=False):
    all_counts = []
    for counts, states in zip(state_counts_list, states_list):
        c = [counts[s] for s in states[:, 0]]
        c = c / sum(c)
        all_counts.append(max(c)*1.1)
        all_counts.append(min(c)*0.9)

    x_max = max(all_counts)
    x_min = min(all_counts)

    fig, axs = plt.subplots(len(states_list), 1)

    for ax, states, state_counts in zip(axs, states_list, state_counts_list):
        scatter(ax, states, state_counts)
        ax.set_ylabel("Frames in a row in the state")
        ax.legend()
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim((x_min, x_max))
    
    axs[-1].set_xlabel("Fraction of frames in state")

    plt.tight_layout()
    if save_plot:
        plt.savefig(save_plot)
    plt.show()


def main():
    # Parse arguments
    args = parse_args()

    # Read row file and trajectory
    states = np.loadtxt(args.row_file)
    state_traj = np.loadtxt(args.state_traj, dtype=int)

    # Count frames per state
    state_counts = count_states(state_traj)

    # Plot states
    if args.data_set:
        states_list = [states]
        state_counts_list = [state_counts]
        for row_file, state_traj_file in args.data_set:
            # Read row file and trajectory
            states_list.append(np.loadtxt(row_file))
            state_traj = np.loadtxt(state_traj_file, dtype=int)

            # Count frames per state
            state_counts_list.append(count_states(state_traj))

        plot_multi(states_list, state_counts_list, args.save_plot)
    else:
        plot(states, state_counts, args.save_plot)

if __name__ == "__main__":
    main()

