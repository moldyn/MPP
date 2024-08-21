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
import MDAnalysis.transformations as trans
from MDAnalysis.analysis import align
from MDAnalysis.analysis.rms import RMSF
import numpy as np

FIRST = 1526041
#FIRST = 1000

def parse_args():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Analyze protein MD trajectory and calculate RMSF for each state."
    )
    parser.add_argument(
        "topology",
        type=str,
        help="Path to the topology file"
    )
    parser.add_argument(
        "trajectory",
        type=str,
        help="Path to the trajectory file"
    )
    parser.add_argument(
        "-a",
        '--aligned',
        action='store_true',
        help='Set this flag if the trajectory is already prealigned',
    )
    parser.add_argument(
        "output",
        type=str,
        help="Path to the output file"
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
    return np.loadtxt(state_file, dtype=int)[:FIRST]

def prepare_universe(topology, trajectory, aligned=False):
    """
    Load the topology and trajectory and center it in the box.

    Parameters:
        topology (str): Path to topology file.
        trajectory (str): Path to trajectory file.

    Returns:
        mda.Universe: Universe with prepared trajectory.
    """
    print("Reading trajectory")
    u = mda.Universe(topology, trajectory, guess_bonds=True)
    ag = u.atoms
    print("Define transformations")
    transforms = [
        trans.unwrap(ag),
        trans.center_in_box(ag, wrap=True),
    ]
    print("Add transformations")
    u.trajectory.add_transformations(*transforms)
    if not aligned:
        print("Prealign trajectory")
        prealigner = align.AlignTraj(
            u,
            u,
            select="name CA",
            in_memory=True,
            verbose=True,
        ).run()
        print("Create ref coords")
        ref_coordinates = u.trajectory.timeseries(atomgroup=ag).mean(axis=1)
    else:
        print("Create ref coords")
        ref_coordinates = u.trajectory.timeseries(atomgroup=ag).mean(axis=0)
    print("Merge ref")
    reference = mda.Merge(ag).load_new(
        ref_coordinates[:, None, :],
        order="afc"
    )
    print("Align trajectory")
    aligner = align.AlignTraj(
        u,
        reference,
        select="name CA",
        in_memory=True,
    ).run()
    return u
    
def main():
    args = parse_args()
    
    # Load the MD trajectory
    universe = prepare_universe(args.topology, args.trajectory, args.aligned)
    ag = universe.select_atoms("all")
    
    # Write aligned trajectory
    with mda.Writer(args.output, ag.n_atoms) as W:
        for ts in universe.trajectory:
            W.write(ag)

if __name__ == "__main__":
    main()
