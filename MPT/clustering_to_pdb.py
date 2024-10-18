#!/usr/bin/env python

import MDAnalysis as mda

"""
This Scirpt converts a cluestering to an average structure for each macrostate and writes the variance for each Ca to the b-factor field.
"""

def clustering_to_pdb(mpt, topology, trajectory):
    u = mda.Universe(topology, trajectory)
    return u
