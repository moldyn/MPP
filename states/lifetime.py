#!/usr/bin/env python3

"""
This script determines the lifetime of the states. It writes the minimum,
maximum and mean lifetime to a file (.row).
"""

import argparse
import numpy as np

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

    # Now calculate max, min, and mean for each state
    results = {}
    for state, lengths in state_dict.items():
        max_length = max(lengths)
        min_length = min(lengths)
        mean_length = np.mean(lengths)
        results[state] = (max_length, min_length, mean_length)
    
    return results

def write_results_to_file(results, filename):
    if filename[-4:] != ".row":
        filename += ".row"
    with open(filename, 'w') as file:
        # Write header
        file.write("#State Max_Length Min_Length Mean_Length\n")
        for state, (max_length, min_length, mean_length) in results.items():
            file.write(f"{state} {max_length} {min_length} {mean_length:.2f}\n")

def main():
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
    results = state_lengths(arr)

    # Write results to the output file
    write_results_to_file(results, args.output_file)

if __name__ == "__main__":
    main()
