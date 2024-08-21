#!/usr/bin/env python3

import argparse
import numpy as np
import msmhelper as mh
from msmhelper.msm import row_normalize_matrix as normalize

def parse_args():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Write calculate transition matrix from state trajectory and "
            "write it in numpy format."
        )
    )
    parser.add_argument(
        "trajectory",
        type=str,
        help="Path to the state trajectory file"
    )
    parser.add_argument(
        "lagtime",
        type=int,
        help='Lag time to be used',
    )
    parser.add_argument(
        "output",
        type=str,
        help="Path to the output file (extension: .npy)"
    )
    return parser.parse_args()

def get_tmat(state_traj, tlag):
    traj = np.loadtxt(state_traj, dtype=int, comments='#')
    return normalize(mh.msm.estimate_markov_model(traj, tlag)[0])

def main():
    args = parse_args()

    tmat = get_tmat(args.trajectory, args.lagtime)

    if not args.output.endswith(".npy"):
        args.output += ".npy"

    with open(args.output, "wb") as f:
        np.save(f, tmat)

if __name__ == "__main__":
    main() 
