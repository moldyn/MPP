import mosaic
import numpy as np
import matplotlib.pyplot as plt
import prettypyplot as pplt
from matplotlib.colors import ListedColormap
from tqdm import tqdm
pplt.use_style(figsize=3.8, figratio=1)


def get_alpha_cmap(cmap, alpha_fraction):
    """Add alpha channel to cmap."""
    cmap = plt.get_cmap(cmap)
    cmap_alpha = cmap(np.arange(cmap.N))
    ncolors = len(cmap_alpha)
    alpha = np.ones(ncolors)
    alpha_n = int(alpha_fraction * ncolors)
    alpha[:alpha_n] = np.linspace(0, 1, alpha_n)
    cmap_alpha[:, -1] = alpha
    return ListedColormap(cmap_alpha)


def mosaic_clusters(matrix_dir, resolution, decimals):
    """Cluster the correlation matrix

    Args:
        matrix_dir: directory to correlation matrix
        resolution: resolution parameter
        decimals: decimals to save figure

    Returns:
        filename, clustering: clusters of matrix, their filename
    """
    clustering = mosaic.Clustering(
        mode='CPM',
        weighted=True,
        resolution_parameter=resolution
    )
    matrix = np.loadtxt(matrix_dir, delimiter=' ')
    clustering.fit(matrix)
    filename = (
                matrix_dir
                + f'_mosaic_clusters_resolution_{resolution:.{decimals}f}'
    )
    mosaic.utils.save_clusters(filename, clustering.clusters_)
    return filename, clustering


def visualize_matrix(matrix_dir, resolution, cluster_amount, decimals):
    """Plot MoSAIC clustering of matrix

    Args:
        matrix_dir: directory to correlation matrix
        resolution: resolution parameter
        cluster_amount: number of clusters to plot
        decimals: decimals to save figure
    """
    matrix = np.loadtxt(matrix_dir, delimiter=' ')
    cluster_dir, clustering = mosaic_clusters(matrix_dir, resolution, decimals)
    idxs = np.argsort([
        len(cluster) for cluster in clustering.clusters_
    ])[::-1]
    clusters_sorted_flattened = np.concatenate(clustering.clusters_[idxs])
    # sort the matrix accordingly
    matrix_sorted = matrix[
        np.ix_(clusters_sorted_flattened, clusters_sorted_flattened)
    ]
    ticks = np.cumsum([len(cluster) for cluster in clustering.clusters_[idxs]])
    ticks = [0, *ticks[:-1]]  # ticks start with 0

    fig, ax = plt.subplots()
    cmap = get_alpha_cmap('macaw_r',  alpha_fraction=0.4)
    cmap.set_under(color='w')
    cmap.set_bad(color='pplt:gray')
    im = ax.imshow(
        matrix_sorted,
        vmin=np.min(matrix_sorted),
        vmax=np.max(matrix_sorted),
        zorder=0,
        interpolation='none',
        rasterized=True,
        cmap=cmap,
        )

    ax.invert_yaxis()  # origin to the upper left
    ax.set_aspect('equal')  # 1:1 ratio
    ax.set_xticks(ticks[:cluster_amount])
    ax.set_yticks(ticks[:cluster_amount])
    ax.set_xticklabels(np.arange(cluster_amount)+1)
    ax.set_yticklabels(np.arange(cluster_amount)+1)
    ax.set_xlabel('clusters')
    ax.set_ylabel('clusters')
    ax.grid(True)
    plt.colorbar(im, label=r'$|\rho|$')
    plt.savefig(cluster_dir + '.pdf')
    plt.close()


def visualize_matrices(matrix_dir, resolutions, cluster_amounts):
    """Plot MoSAIC clustering of comparison matrix for several resolutions

    Args:
        matrix_dir: directory to correlation matrix
        resolution: resolution parameter
        cluster_amount: number of clusters to plot
    """
    decimals_list = [
        len(str(resolution).split('.')[1]) for resolution in resolutions
    ]
    decimals = np.max(decimals_list)
    for i, resolution in tqdm(enumerate(resolutions)):
        visualize_matrix(matrix_dir, resolution, cluster_amounts[i], decimals)


if __name__ == '__main__':
    comp_mat_dir = '/data/MPP_MC/Lukas/Scripts/MCMC_test_HP35/'\
        'macrotraj_compmat'
    res_params = [0.83, 0.84, 0.85, 0.86, 0.87, 0.88, 0.89, 0.90]
    amount_of_clusters = [2, 2, 2, 2, 2, 2, 2, 2]

    visualize_matrices(comp_mat_dir, res_params, amount_of_clusters)
