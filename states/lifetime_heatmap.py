#!/usr/bin/env python3

"""
This script determines the lifetime of the states. It writes the minimum,
maximum and mean lifetime to a file (.row).
"""

import argparse
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

def state_lengths(arr):
    # Dictionary to store lengths of sequences for each state
    state_dict = {}

    # Variables to track the current state and its sequence length
    current_state = arr[0]
    current_length = 1

    for i in range(1, len(arr)):
        if arr[i] == current_state:
            current_length += 1
        else:
            # Append the length of the sequence to the respective state's list
            if current_state not in state_dict:
                state_dict[current_state] = []
            state_dict[current_state].append(current_length)
            
            # Update current_state and reset current_length
            current_state = arr[i]
            current_length = 1

    # Don't forget to append the last sequence length
    if current_state not in state_dict:
        state_dict[current_state] = []
    state_dict[current_state].append(current_length)

    return state_dict

def plot_histogram_heatmap(data_dict, output, ax, show_cbar=True, plot=True):
    # Determine the range of values for the histogram
    all_values = [value for sublist in data_dict.values() for value in sublist]
    
    # Define the bins for the histogram in log scale
    bin_edges = np.array([2**i for i in range(11)]) - 0.5
    bins = len(bin_edges) - 1
    
    # Prepare the 2D array to store the histogram counts
    histogram_data = np.zeros((len(data_dict), bins))
    
    # Fill the histogram data
    keys = sorted(data_dict.keys())
    for i, key in enumerate(keys):
        values = data_dict[key]
        counts, _ = np.histogram(values, bins=bin_edges)
        histogram_data[i, :] = counts

    # Plot the heatmap
    sns.heatmap(
        histogram_data,
        norm=LogNorm(vmin=1, vmax=900),
        cmap='viridis',
        cbar_kws={'label': 'count'},
        cbar=show_cbar,
        ax=ax,
    )
    
    ax.set_xlabel('lifetime')
    ax.set_ylabel('State')
    
    # Set log scale for x-axis
#    plt.xscale('log')
    ax.set_xticks(ticks=np.array([i for i in range(len(bin_edges))]), labels=[f'{edge+0.5:.0f}' for edge in bin_edges], rotation=45)
    
    # Set the y-axis labels
    ylabels = []
    for i, k in enumerate(keys):
        if not i % 10:
            ylabels.append(k)
        else:
            ylabels.append("")

    ax.set_yticks(ticks=np.arange(len(keys)) + 0.5, labels=ylabels, rotation=0)
    if plot:
        plt.tight_layout()
        plt.savefig(output)
        plt.show()

def main():
    plt.rcParams.update({'font.size': 30})

    parser = argparse.ArgumentParser(
        description=(
            "This script determines the lifetime of the states. It writes the "
            "minimum, maximum and mean lifetime to a file (.row)."
        )
    )
    parser.add_argument(
        "input_file",
        type=str,
        help="Path to the state trajectory file."
    )
    parser.add_argument(
        "output_file",
        type=str,
        help=(
            "Path to the output text file to write the results to. The "
            "extension '.row' is added, if not already present."
        )
    )

    args = parser.parse_args()

    # Load the array from the input file
    arr = np.loadtxt(args.input_file, dtype=int)

    # Calculate state lengths
    state_dict = state_lengths(arr)
    s1 = {}
    s2 = {}
    for k in state_dict:
        if k < 274:
            s1[k] = state_dict[k]
        else:
            s2[k] = state_dict[k]

    #plt.figure(figsize=(18, 24))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 24))
    fig.suptitle('Lifetime of states')
#    plot_histogram_heatmap(state_dict, args.output_file)
    plot_histogram_heatmap(s1, args.output_file, ax1, show_cbar=False, plot=False)
    plot_histogram_heatmap(s2, args.output_file, ax2)
    #plot_heatmap(state_dict)
    # Write results to the output file
#    write_results_to_file(results, args.output_file)

if __name__ == "__main__":
    main()
