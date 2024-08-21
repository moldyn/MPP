#!/usr/bin/env python3

import numpy as np
from plot_dendrogram import plot_dendrogram_mpt
from MPT import MPTBase

traj = np.loadtxt(
    "/path/to/hp35.selected_contacts.gaussian10f"
            "_microstates_pcs5_p153",
    dtype=int
)
feature_traj = np.loadtxt(
    "/path/to/hp35.mindists2.gaussian10f.q"
)


mpt = MPTBase(traj, 50, method="smpt", params={"%": 0.5})
mpt.add_feature("fnc", feature_traj)
mpt.apply_feature("fnc")
mpt.mpt(4)
dd = plot_dendrogram_mpt(
    mpt,
    f"/path/to/h35_fg_dendrogram_smpt_p50_fnc"
)

