#!/usr/bin/env python

import numpy as np

from scipy.stats import entropy # aka Kullback-Leibler divergence
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist

import matplotlib.pyplot as plt

dir = "/data/evaluation/MPP/stochastic_MPP_Felix/data_source/"
out_dir = "/data/evaluation/MPP/stochastic_MPP_Felix/data_production/MPT/MPT/"

distances = np.loadtxt(dir + "hp35.mindists2")
traj = np.loadtxt(dir + "hp35.selected_contacts.gaussian10f_microstates_pcs5_p153", dtype=int)

states, pop = np.unique(traj, return_counts=True)
n_states = states.shape[0]



pca = PCA()
#pca.fit(distances)
distances_rot = pca.fit_transform(distances)

ratio = pca.explained_variance_ratio_

mean_d = np.zeros((n_states, distances.shape[1]))
var_d = np.zeros((n_states, distances.shape[1]))

for i in range(n_states):
    mean_d[i] = distances_rot[traj == i + 1].mean(axis=0)
    var_d[i] = np.var(distances_rot[traj == i + 1], axis=0)


d42 = pdist(mean_d)
d5 = pdist(mean_d[:, :5])
d3 = pdist(mean_d[:, :3])
n = [42, 5, 3]
labels = ["all 42 PCs", "5 PCs", "3 PCs"]
cs = ["tab:blue", "tab:orange", "tab:green"]

fig, ax = plt.subplots()
for nd, l, c in zip(n, labels, cs):
    d = pdist(mean_d[:, :nd])
    ax.hist(d, bins=20, alpha=0.5, label=l, color=c, density=True)
    v = var_d[:, :nd].sum(axis=1)
    ax.hist(v, bins=20, alpha=1, color=c, histtype="step", density=True)

plt.legend()
plt.title("Significance of principal components of native contacts")
plt.xlabel("Bars: Euclidean distance between means of microstates; Steps: variance")
plt.ylabel("Density")
plt.tight_layout()
plt.savefig(out_dir + "img/contacts_var_pca.pdf")


# fig, axs = plt.subplots(6, 7, figsize=(12, 14))
#
# cum_ratio = 0
# for i, ax in enumerate(axs.flatten()):
#     #ax.hist(var_d[:, i] / np.abs(mean_d[:, i]), bins=np.linspace(0, 20, 21))
#     ax.hist(var_d[:, i])
#     #    ax.set_xlim(0, 10)
#     ax.set_xlim(0, 1.5)
#     ax.set_xlabel("var")
#     ax.set_ylabel("count")
#     cum_ratio += ratio[i]
#     ax.set_title(f"PC {i + 1} - {cum_ratio*100:.1f} %")
#
# plt.tight_layout()
# plt.savefig(out_dir + "features_pca.pdf")



# mean_d = np.zeros((n_states, distances.shape[1]))
# var_d = np.zeros((n_states, distances.shape[1]))
#
# for i in range(n_states):
#     mean_d[i] = distances[traj == i + 1].mean(axis=0)
#     var_d[i] = np.var(distances[traj == i + 1])

# fig, axs = plt.subplots(6, 7, figsize=(12, 14))
#
# for i, ax in enumerate(axs.flatten()):
#     ax.hist(var_d[:, i] / mean_d[:, i])
#     ax.set_xlim(0, 1.3)
#     ax.set_xlabel("var")
#     ax.set_ylabel("count")
#     ax.set_title(f"distance {i + 1}")
#
# plt.tight_layout()
# plt.savefig(out_dir + "features.pdf")
