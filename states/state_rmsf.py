#!/usr/bin/env python3
"""
This script analyzes the the states of a given trajectory and calculates the
RMSF for each state.
"""

import argparse
import os
import datetime
import getpass
import socket
from tqdm import tqdm
import MDAnalysis as mda
from MDAnalysis.analysis.rms import RMSF
import numpy as np

def parse_args():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Analyze protein MD trajectory and calculate RMSF for each state. "
            "The output file contains the state id, the number of frames in "
            "that state and the RMSF for the state."
        )
    )
    parser.add_argument(
        "topology",
        type=str,
        help="Path to the topology file"
    )
    parser.add_argument(
        "trajectory",
        type=str,
        help="Path to the aligned trajectory file"
    )
    parser.add_argument(
        "state_trajectory",
        type=str,
        help="Path to the state trajectory file"
    )
    parser.add_argument(
        "output_file",
        type=str,
        help=(
            "Path to the output file. The extension '.rmsf' will be added, if "
            "not already present."
        )
    )
    return parser.parse_args()

def load_state_trajectory(state_file):
    """
    Load the state trajectory file.
    
    Parameters:
        state_file (str): Path to the state trajectory file.
        
    Returns:
        np.ndarray: Array of state IDs.
    """
    return np.loadtxt(state_file, dtype=int)

def calculate_rmsf(universe, state_ids):
    """
    Calculate RMSF for each state.
    
    Parameters:
        universe (MDAnalysis.Universe): The MDAnalysis universe object.
        universe (MDAnalysis.Universe): Reference state.
        state_ids (np.ndarray): Array of state IDs.
        
    Returns:
        dict: Dictionary with state IDs as keys and RMSF values as values.
    """
    rmsf_results = {}
    unique_states = np.unique(state_ids)
    
    for state in tqdm(unique_states):
        selection = universe.select_atoms("name CA")
        frames = np.where(state_ids == state)[0]
        rmsfer = RMSF(selection, verbose=True).run(frames=frames)
        rmsf_results[state] = (len(frames), rmsfer.results.rmsf)
    
    return rmsf_results

def write_rmsf_to_file(trajectory_file, output_file, rmsf_results):
    """
    Write RMSF results to a file.
    
    Parameters:
        trajectory_file (str): The name of the trajectory file.
        output_file (str): Path to the output file.
        rmsf_results (dict): Dictionary with state IDs as keys and RMSF values as values.
    """
    if output_file[-5:] != ".rmsf":
        output_file += ".rmsf"
    with open(output_file, "w") as f:
        timestamp = datetime.datetime.now().strftime("%d.%m.%Y %H:%M:%S")
        user_name = getpass.getuser()
        host_name = socket.gethostname()
        
        header = (
            f"# Command: {' '.join(os.sys.argv)}\n"
            f"# Timestamp: {timestamp}\n"
            f"# User: {user_name}\n"
            f"# Host: {host_name}\n"
            "#\n"
            "# StateID NumFrames RMSF\n"
        )
        f.write(header)
        
        for state, (num_frames, rmsf_values) in rmsf_results.items():
            #            for rmsf in rmsf_values:
            f.write(f"{state} {num_frames} {rmsf_values.mean():.5f}\n")

def main():
    args = parse_args()
    
    # Load the aligned MD trajectory
    universe = mda.Universe(args.topology, args.trajectory, guess_bonds=True)
    
    # Load the state trajectory
    state_ids = load_state_trajectory(args.state_trajectory)
    
    # Calculate RMSF for each state
    rmsf_results = calculate_rmsf(universe, state_ids)
    
    # Write the RMSF results to the output file
    write_rmsf_to_file(args.trajectory, args.output_file, rmsf_results)

if __name__ == "__main__":
    main()
