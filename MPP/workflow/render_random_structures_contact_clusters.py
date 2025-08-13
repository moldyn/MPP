from pymol import cmd
import sys
import os
import numpy as np


"""Render random structures and color contact clusters.

Arguments: FRAMES OUTFILE CLUSTERS CONTACTS LEAST_MOVING_RESIDUES
FRAMES
    pdb file containing the frames randomly drawn for one state
OUTFILE
    output file to store the image to
CLUSTERS
    definition of the contact clusters. Contains the indices of one
    contact cluster per line, space separated.
CONTACTS
    Contains the contacts used in the clusters file. The line index is
    the index of the contact.
LEAST_MOVING_RESIDUES
    Contains the least moving residues which are used for the alignment.
"""


FRAMES = sys.argv[1]
OUTFILE = sys.argv[2]
CLUSTERS = sys.argv[3]
CONTACTS = sys.argv[4]
LEAST_MOVING_RESIDUES = sys.argv[5]


def load_residue_indices(file_name):
    """Load least moving residue indices from file."""
    residue_indices = []
    with open(file_name, "r") as f:
        for line in f:
            residue_indices.append(line.rstrip().split(" "))
    return residue_indices


def load_clusters(cluster_file):
    """Returns the indices of start and end of contact indices."""
    clusters = []
    with open(CLUSTERS, "r") as f:
        for line in f:
            if line.startswith("#"):
                continue
            l_split = [int(i) for i in line.split(" ")]
            clusters.append((min(l_split), max(l_split) + 1))
    return clusters


def unique_contacts(clusters, contacts):
    """Returns the residue indices belonging to each cluster."""
    uc = []
    for i, j in clusters:
        uc.append(np.unique(contacts[i:j]))
    return uc


def split_cluster(unique_residues):
    """Returns two sets of residues, which are furthest appart in the
    sequence.
    """
    split_residues = []
    for residues in unique_residues:
        first_n = np.argmax(np.diff(np.unique(residues))) + 1
        split_residues.append((residues[:first_n], residues[first_n:]))
    return split_residues


i_macrostate = int(os.path.basename(FRAMES).split(".")[0]) - 1
alignment_residue_indices = load_residue_indices(LEAST_MOVING_RESIDUES)[i_macrostate]
contacts = np.loadtxt(CONTACTS, dtype=int)
clusters = load_clusters(CLUSTERS)

unique_residues = unique_contacts(clusters, contacts)
split_residues = split_cluster(unique_residues)

# Load and split frames
cmd.load(FRAMES, "frames")
cmd.split_states("frames")
cmd.delete("frames")

# Align all frames to the first based on the least moving residues
objects = cmd.get_object_list(selection="(all)")
align_selection = "name CA and resi " + "+".join(alignment_residue_indices)
for o in objects[1:]:
    cmd.align(f"{o} and " + align_selection, f"{objects[0]} and " + align_selection)

# Color the clusters
cmd.orient()
cmd.zoom(complete=1)
for i, cluster in enumerate(split_residues):
    cmd.spectrum("resi", "gray20 gray80")
    cmd.color("cyan", "resi " + "+".join(cluster[0].astype(str)))
    cmd.color("magenta", "resi " + "+".join(cluster[1].astype(str)))
    cmd.png(OUTFILE + f"_cc{i + 1:02d}.png", 1000, 1000, ray=1)
