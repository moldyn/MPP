#!/usr/bin/env python3

import argparse
import MDAnalysis as mda
from MDAnalysis.analysis import align

def calculate_average_structure(topology, trajectories):
    for traj in trajectories:
        # Load the topology and trajectory
        u = mda.Universe(topology, traj)

        # Select all atoms for alignment and averaging
        atom_selection = 'protein'
        ref = u.select_atoms(atom_selection)

        # Perform the average structure calculation
        avg_struct = align.AverageStructure(u, ref, select=atom_selection).run()

        # Get the average positions and set them to the reference atom group
        ref.positions = avg_struct.results.positions

        # Write the averaged structure to a PDB file
        avg_filename = traj.rsplit('.', 1)[0] + '_avg.pdb'
        ref.write(avg_filename)
        print(avg_struct.results.rmsd)
        print(f"Average structure written to {avg_filename}")

def main():
    parser = argparse.ArgumentParser(description="Calculate average structures from MD trajectories.")
    parser.add_argument('topology', help="Topology file (e.g., PSF, PDB, GRO)")
    parser.add_argument('trajectories', nargs='+', help="Trajectory files (e.g., DCD, XTC)")
    args = parser.parse_args()

    calculate_average_structure(args.topology, args.trajectories)

if __name__ == "__main__":
    main()

