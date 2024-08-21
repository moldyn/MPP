#!/usr/bin/env python3

import time
from plot_dendrogram import plot_dendrogram_mpt
import numpy as np
from MPT import MPTBase

def print_runtime(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        runtime = end_time - start_time
        print(f"Function '{func.__name__}' executed in {runtime:.4f} seconds")
        return result
    return wrapper

pre_linkage = np.loadtxt("/home/fg149/Dokumente/data_production/MPT/MPT/hp35.selected_contacts.gaussian10f_microstates_pcs5_p153_linkage.dat")

linkage = pre_linkage[:, :2].astype(int)

macrostates_map = np.loadtxt("/home/fg149/Dokumente/data_production/MPT/MPT/hp35.selected_contacts.gaussian10f_microstates_pcs5_p153_linkage.dat_q.pop0.005_qmin0.20.macrostates", dtype=int)

traj = np.loadtxt("/home/fg149/Dokumente/data_production/MPT/MPT/hp35.selected_contacts.gaussian10f_microstates_pcs5_p153", dtype=int)

q_merges = pre_linkage[:, 2]

feature_traj = np.loadtxt("/home/fg149/Dokumente/data_production/MPT/MPT/hp35.mindists2.gaussian10f.q")

out = "/home/fg149/Dokumente/data_production/MPT/MPT/dendrogram_test"

t0 = time.time()
#mpt = MPTBase(traj, 50, method="smpt", params={"%": 0.9})
mpt = MPTBase(traj, 50, method="smpt", params={"n": 2})
#mpt = MPTBase(traj, 50, method="mpt", params={"n": 2})
print(f"Time for creation of mpt object: {time.time()-t0:.4f} s")

mpt.add_feature("fnc", feature_traj)

t1 = time.time()
mpt.mpt(4)
print(f"Time to run mpt: {time.time()-t1:.4f} s")
params = (
    mpt.full_linkage,
    mpt.state_map,
    traj,
    mpt.full_stp,
    feature_traj,
    "/home/fg149/Dokumente/data_production/MPT/MPT/h35_fg_dendrogram_test"
)
t2 = time.time()
dd = plot_dendrogram_mpt(mpt, "/home/fg149/Dokumente/data_production/MPT/MPT/h35_fg_dendrogram_smpt_n_6")
# h35_fg_dendrogram_smpt_p_1
print(f"Time to plot dendrogram: {time.time()-t2:.4f} s")

# t2 = time.time()
# plot_dendrogram(*params)
# print(f"Time to plot dendrogram: {time.time()-t2:.4f} s")

# t1 = np.concatenate([
#     np.random.randint(0, 5, 50),
#     np.random.randint(5, 8, 40),
#     np.random.randint(7, 10, 60),
#     np.random.randint(8, 10, 20),
#     np.random.randint(0, 5, 40),
#     np.random.randint(5, 8, 50),
#     np.random.randint(7, 10, 60),
#     np.random.randint(8, 10, 20),
#     np.random.randint(0, 5, 50),
#     np.random.randint(5, 8, 30),
#     np.random.randint(7, 10, 50),
#     np.random.randint(8, 10, 20),
#     np.random.randint(0, 5, 50),
# ])
#
# ft = t1 + np.random.uniform(-0.3, 0.3, t1.shape[0])
#
#
# mpt:
# - full_linkage
# - state_map
# - n_states
# - pop
# - full_merge_pop
# - features
# - traj
#
