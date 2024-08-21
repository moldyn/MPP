#!/usr/bin/env python

import sys
import os
sys.path.append("/data/evaluation/MPP/stochastic_MPP_Felix/tools/MPT")

from tqdm import tqdm
import numpy as np
from itertools import combinations

import MPT
from msmhelper.md import compare_discretization
import mosaic

traj = np.loadtxt("/home/fg149/Dokumente/data_production/MPT/MPT/hp35.selected_contacts.gaussian10f_microstates_pcs5_p153", dtype=int)
feature_traj = np.loadtxt("/home/fg149/Dokumente/data_production/MPT/MPT/hp35.mindists2.gaussian10f.q")
out_base = "/home/fg149/Dokumente/data_production/MPT/MPT/"

lagtime = 50
smpt_kernel = MPT.kernel.SMPTKernel(method="n", param=2)

i = out_base + "macrotrajs_n2.npy"
#i = "/tmp/macrotrajs_n2.npy"
# trajs = np.load(i).T
# trajs = trajs[:, :100]
#
# n_combinations = trajs.shape[1] * (trajs.shape[1] - 1) / 2
# S = np.ones((trajs.shape[1], trajs.shape[1]))
# for i, j in tqdm(combinations(range(trajs.shape[1]), 2)):
#     S[i, j] = S[j, i] = compare_discretization(trajs[:, i], trajs[:, j])
# np.save("S100.npy", S)
S = np.load("S100.npy")

# S_min = (S - S.min())
# S = S_min / S_min.max()




# Cluster the correlation matrix
clustering = mosaic.Clustering(
    #mode='modularity',
    mode='CPM',
    resolution_parameter=0.88,
)
clustering.fit(S)


idxs = np.argsort(
    [len(cluster) for cluster in clustering.clusters_],
)[::-1]
clusters_sorted = clustering.clusters_[idxs]
clusters_sorted_flattened = np.concatenate(clustering.clusters_[idxs])

# sort the matrix accordingly
# matrix_sorted = S[
#     np.ix_(clusters_sorted_flattened, clusters_sorted_flattened)
# ]
matrix_sorted = S[clusters_sorted_flattened][:, clusters_sorted_flattened]
ticks = np.cumsum([len(cluster) for cluster in clustering.clusters_[idxs]])
ticks = [0, *ticks[:-1]]  # ticks start with 0



import matplotlib.pyplot as plt


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
#im1 = ax1.pcolormesh(clustering.matrix_)
im1 = ax1.pcolormesh(S)
ax1.invert_yaxis()  # origin to the upper left
ax1.set_aspect('equal')  # 1:1 ratio
ax1.set_xticks(clustering.ticks_)
ax1.set_yticks(clustering.ticks_)
ax1.set_xticklabels([])
ax1.set_yticklabels([])
ax1.set_xlabel('clusters')
ax1.set_ylabel('clusters')
ax1.grid(False)
plt.colorbar(im1, label=r'$|\rho|$')


# Perform the same plot again, but with sorted clusters
#fig, ax = plt.subplots()
im2 = ax2.pcolormesh(
    matrix_sorted,
    snap=True,
    # vmin=0,
    # vmax=1,
)
ax2.invert_yaxis()  # origin to the upper left
ax2.set_aspect('equal')  # 1:1 ratio
ax2.set_xticks(ticks[:4])  # we focus only on the first three clusters
ax2.set_yticks(ticks[:4])
ax2.set_xticklabels(np.arange(4)+1)
ax2.set_yticklabels(np.arange(4)+1)
ax2.set_xlabel('clusters')
ax2.set_ylabel('clusters')
ax2.grid(True)
plt.colorbar(im2, label=r'$|\rho|$')
plt.tight_layout()
plt.savefig(out_base + "img/mosaic.pdf")
