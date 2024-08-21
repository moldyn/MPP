#!/usr/bin/env python3

import sys
import argparse
from collections import defaultdict
import numpy as np

states = np.loadtxt(sys.argv[1], dtype=int)
state_assignment = defaultdict(list)
for microstate, macrostate in states:
    state_assignment[macrostate].append(microstate)

with open(sys.argv[2], "w") as f:
    for macrostate in state_assignment:
        f.write(f"{macrostate:2d}: {' '.join([str(i) for i in state_assignment[macrostate]])}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="check_rmsd.py",
        description="Caluclate the RMSD within each macrostate and evaluates the distribution",
        epilog="by Felix Guischard",
    )
    parser.add_argument(
        "input_macrostate",
        type=argparse.FileType("r", encoding="latin-1"),
        help="Macrostate file from MPT",
    )
    parser.add_argument(
        "input_top",
        type=argparse.FileType("r"),
        help="Input topology (.pdb or .top)",
    )
    parser.add_argument(
        "input_traj",
        type=argparse.FileType("r"),
        help="Input trajectory (.xtc)",
    )
    parser.add_argument("-o", "--output", default="macrostate_statistics.txt")

