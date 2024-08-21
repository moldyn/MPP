# Routine to find close contacts in a trajectry

## run_contacts_routine.sh
Executes all scripts. This bash script as well as extract_contacts.py and extract_indices.py were extended to allow for an upper and a lower threshold.

The required .ndx file are just the residue index pairs to consider.

This routine calculates minimal distances in the following way:

1 - for pairs of residues given, calculate the frequency of forming a contact (closest heavy atoms d<=0.45 nm) in the trajectory
2 - save indices of the residues that form a contact for more than thr % of the trajectory
3 - calculate the value of the distance for all atom pairs between pairs of residues identified in the previous step
4 - calculate minimum distance for couple of residues where the same atom pair forms contact for more than thr % of the trajectory
