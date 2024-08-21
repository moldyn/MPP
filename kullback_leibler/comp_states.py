#!/usr/bin/env python

import numpy as np

from scipy.stats import entropy # aka Kullback-Leibler divergence
from scipy.spatial.distance import pdist

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import prettypyplot as pplt
pplt.colors.load_cmaps

import mosaic

def get_alpha_cmap(cmap, alpha_fraction=0.1):
    """Add alpha channel to cmap."""
    cmap = plt.get_cmap(cmap)
    cmap_alpha = cmap(np.arange(cmap.N))
    ncolors = len(cmap_alpha)

    alpha = np.ones(ncolors)
    alpha_n = int(alpha_fraction * ncolors)
    alpha[:alpha_n] = np.linspace(0, 1, alpha_n)
    cmap_alpha[:, -1] = alpha
    return ListedColormap(cmap_alpha)

dir = "/data/evaluation/MPP/stochastic_MPP_Felix/data_source/"
out_dir = "/data/evaluation/MPP/stochastic_MPP_Felix/data_production/MPT/MPT/"

distances = np.loadtxt(dir + "hp35.mindists2")
# Microtraj
# traj = np.loadtxt(dir + "hp35.selected_contacts.gaussian10f_microstates_pcs5_p153", dtype=int)
# Macrotraj
traj = np.loadtxt(out_dir + "hp35_det_macrotraj", dtype=int)

states, pop = np.unique(traj, return_counts=True)
n_states = states.shape[0]

c1 = [2, 1, 9, 5]
c2 = [12, 13, 8, 11, 7, 10]
c3 = [18, 17, 19, 20]
c4 = [26, 24, 28]
c5 = [33, 29, 34]
c6 = [40, 39, 41, 38]
c7 = [16, 14, 15]
#contacts = c2 + c3 + c4 #+ c5 + c6 + c7
contacts = c1 + c2 + c3 + c4 + c5 + c6 + c7


def feature_correlation(dist):
    sim = mosaic.Similarity(metric="correlation")
    #sim = mosaic.Similarity(normalize_method="arithmetic")
    sim.fit(dist)
    return sim

def cluster_sim(mat, res=None):
    cl = mosaic.Clustering(resolution_parameter=res)
#    cl = mosaic.Clustering(mode="CPM", resolution_parameter=0.78)
    #cl = mosaic.Clustering(mode="modularity")
    cl.fit(mat)
    return cl

def plot_matrix(clust, output_file):
    _, ax = plt.subplots(figsize=(12, 10), dpi=192)
    pplt.use_style()
    mat = clust.matrix_.astype(np.float64)
    cmap = get_alpha_cmap('macaw_r', alpha_fraction=0.4)
    cmap.set_under(color='w')
    cmap.set_bad(color='pplt:gray')
#    mat[np.diag_indices_from(mat)] = np.nan
    #im = ax.pcolormesh(
    im = ax.imshow(
        #mat, snap=True, vmin=0, vmax=np.nanmax(mat), norm="log",
        mat,
        snap=True,
        #, norm="log",
        aspect="equal",
        cmap=cmap,
        origin="upper",
    )

    ticks = np.array([0, *clust.ticks_[: -1]]) - 0.5
    major_mask = np.array([
        # len(cluster) > 2 for cluster in clust.clusters_
        len(cluster) > 0 for cluster in clust.clusters_
    ])
    # major_mask[:] = True
    ticklabels = np.arange(len(ticks)) + 1
    for set_ticks, set_ticklabels in (
        (ax.set_xticks, ax.set_xticklabels),
        (ax.set_yticks, ax.set_yticklabels),
    ):
        set_ticks(ticks[major_mask])
        set_ticklabels(ticklabels[major_mask])
        set_ticks(ticks[~major_mask], minor=True)
        set_ticklabels([], minor=True)

    ax.grid(True, ls='-', lw=2)
    ax.grid(True, ls='-', which='minor', lw=0.4)

    ax.set_title(r"Macrostates clustered by 7 native contact clusters, $\gamma$=" + f"{clust.resolution_parameter:.2f}")
    ax.set_xlabel('clusters')
    ax.set_ylabel('clusters')

    plt.colorbar(im)#, width='3%')
    plt.tight_layout()
    plt.savefig(f'{output_file}')


def prob_states(traj, dist, thr=0.45):
    # thr in nm
    #probs = np.zeros((n_states, dist.shape[1]))
    probs = np.zeros((n_states, len(contacts)))
    for state in range(n_states):
        probs[state] = (dist[:, contacts][traj==state+1] < thr).mean(axis=0)
    return probs

def kl_dist(probs):
    p_mat = np.zeros((probs.shape[0], probs.shape[0]))
    for i, pi in enumerate(probs[:-1]):
        for j, pj in enumerate(probs[i+1:], start=i+1):
            p_mat[i, j] = p_mat[j, i] = entropy(pi, pj)
    return p_mat

#sim = feature_correlation(distances)
#cl = cluster_sim(sim.matrix_)

print("Calculate probs")
probs = prob_states(traj, distances)
print("Calculate KL")
mat = kl_dist(probs+1)
#mat = np.load(out_dir + "mat_c_all_micro.npy")
print("Cluster")
# nmat = mat / mat.max()
# pnmat = 1 - nmat
# np.fill_diagonal(pnmat, 1)

# # Norm 1
# lmat = np.log(mat + 1) # 0 - 0.69
# lpmat = lmat - lmat.min() # 0 - 0.69
# np.fill_diagonal(lpmat, 0)
# lpnmati = lpmat / lpmat[lpmat<1].max() # 0 - 1
# lp = 1 - lpnmati # 0 - 1
# np.fill_diagonal(lp, 1) # 0 - 1

lp = (mat.max() - mat) / (mat.max() - mat.min())

# lpnmat = np.load(out_dir + "lpnmat.npy")
#
lpnmat = lp # 0.83 - 1
cl = cluster_sim(lpnmat)
plot_matrix(cl, out_dir + "img/tmp_clustering.pdf")

# cl3 = cluster_sim(lpnmat, res=0.55)
# plot_matrix(cl3, out_dir + "img/tmp_clustering_055c.svg")
# cl2 = cluster_sim(lpnmat, res=0.6)
# plot_matrix(cl2, out_dir + "img/tmp_clustering_060c.svg")
# cl1 = cluster_sim(lpnmat, res=0.65)
# plot_matrix(cl1, out_dir + "img/tmp_clustering_065c.svg")

# TODO:
# - Correlation between features
# - select features of first 7 clusters (mosaic)
# - Calc prob for states
# - KL for state combinations
# - mosaic
