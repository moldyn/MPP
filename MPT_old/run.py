#!/usr/bin/env python3

import numpy as np
from plot_dendrogram import plot_dendrogram_mpt
from MPT import MPTBase

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

traj = np.loadtxt("/home/fg149/Dokumente/data_production/MPT/MPT/hp35.selected_contacts.gaussian10f_microstates_pcs5_p153", dtype=int)
feature_traj = np.loadtxt("/home/fg149/Dokumente/data_production/MPT/MPT/hp35.mindists2.gaussian10f.q")


mpt = MPTBase(traj, 50, method="mpt", params={"%": 0.9})
mpt.add_feature("fnc", feature_traj)
#mpt.apply_feature("fnc")
mpt.mpt(2)
dd = plot_dendrogram_mpt(mpt, f"/home/fg149/Dokumente/data_production/MPT/MPT/h35_fg_dendrogram_mpt_fnc")

# for i in range(1, 7):
#     mpt = MPTBase(traj, 50, method="smpt", params={"%": 0.5})
#     #mpt = MPTBase(traj, 50, method="smpt", params={"n": 2})
#     #mpt = MPTBase(traj, 50, method="mpt", params={"n": 2})
#     mpt.add_feature("fnc", feature_traj)
#     mpt.mpt(4)
#
#     dd = plot_dendrogram_mpt(mpt, f"/home/fg149/Dokumente/data_production/MPT/MPT/h35_fg_dendrogram_smpt_p50_{i}")

