import numpy as np
import matplotlib.pyplot as plt
import pathlib
from tqdm import tqdm
import math


def compare_macrotrajs(ref_macrotraj, comp_macrotraj):
    ref_macros = np.unique(ref_macrotraj)
    comp_macros = np.unique(comp_macrotraj)
    macros_sim = []
    for ref_macro in ref_macros:
        mask_ref = ref_macrotraj == ref_macro
        macro_sim_union = []
        macro_sim_ref = []
        macro_sim_clus = []
        for comp_macro in comp_macros:
            mask_comp = comp_macrotraj == comp_macro
            mask_union = np.logical_or(mask_ref, mask_comp)
            mask_intersection = np.logical_and(mask_ref, mask_comp)
            count_inter = np.count_nonzero(mask_intersection)
            sim_union = count_inter / np.count_nonzero(mask_union)
            sim_ref = count_inter / np.count_nonzero(mask_ref)
            sim_clus = count_inter / np.count_nonzero(mask_comp)
            macro_sim_union.append(sim_union)
            macro_sim_ref.append(sim_ref)
            macro_sim_clus.append(sim_clus)
        macros_sim.append([np.max(macro_sim_union),
                           np.max(macro_sim_ref),
                           np.max(macro_sim_clus)])
        # macros_sim.append(np.max(macro_sim_union))
    return macros_sim


def compare_multiple_macrotrajs(
    ref_macrotraj_dir,
    folder_comp_macrotrajs,
    nbins
):
    comp_macrotrajs_dir = traj_name_list(folder_comp_macrotrajs)
    ref_macrotraj = np.loadtxt(ref_macrotraj_dir, dtype=int, comments='#')
    macro_sim_multiple = []
    n_macrostates = []
    for comp_macrotraj_dir in tqdm(comp_macrotrajs_dir):
        comp_macrotraj = np.loadtxt(
            comp_macrotraj_dir,
            dtype=int,
            comments='#'
        )
        n_macrostates.append(np.max(comp_macrotraj))
        macro_sim_multiple.append(
            compare_macrotrajs(ref_macrotraj, comp_macrotraj)
        )
    histo_file = folder_comp_macrotrajs + '/comparison_state_similarity'
    sim_mat = np.array(macro_sim_multiple)
    plot_state_sim_histograms(
        sim_mat,
        nbins,
        folder_comp_macrotrajs + '/comparison_state_similarity_histograms.pdf'
    )
    np.savetxt(
        histo_file,
        sim_mat.flatten(),
        delimiter=' ',
        comments='#',
        header=f'shape: {np.shape(sim_mat)}, (nclusterings, nmacros, nsims)'
    )
    np.savetxt(folder_comp_macrotrajs + '/count_macrostates',  n_macrostates)
    return sim_mat, np.shape(sim_mat)


def traj_name_list(folder_dir):
    """Generate list of macrotrajs in folder

    Args:
        folder_dir (str): directory of folder
        name_end (str): type of files in folder

    Returns:
        traj_names: list of all specific filenames in folder
    """
    folder = pathlib.Path(folder_dir)
    traj_names = list(folder.glob("*.macrotraj"))
    return traj_names


def plot_state_sim_histograms(sim_mat, nbins, save_dir):
    states, columns, nsims = np.shape(sim_mat)
    plot_cols = math.ceil(columns/2)
    colors = ['green', 'blue', 'red']
    labels = ['union', 'reference', 'clustering']
    fig, axs = plt.subplots(
        2, plot_cols, sharey=False, tight_layout=True, figsize=(12, 5.5)
    )
    for i in range(columns):
        if i < plot_cols:
            bins = np.histogram(
                np.hstack(
                    (sim_mat[:, i, 0], sim_mat[:, i, 1], sim_mat[:, i, 2])
                ), bins=nbins
            )[1]
            for idx in range(len(labels)):
                axs[0, i].hist(
                    sim_mat[:, i, idx],
                    bins=bins,
                    color=colors[idx],
                    label=labels[idx],
                    alpha=0.7,
                    align='mid'
                )
            axs[0, i].set_title(f'State {i + 1}')
        else:
            bins = np.histogram(
                np.hstack(
                    (sim_mat[:, i, 0], sim_mat[:, i, 1], sim_mat[:, i, 2])
                ), bins=nbins
            )[1]
            for idx in range(len(labels)):
                axs[1, i-plot_cols].hist(
                    sim_mat[:, i, idx],
                    bins=bins,
                    color=colors[idx],
                    label=labels[idx],
                    alpha=0.7,
                    align='mid'
                )
            axs[1, i-plot_cols].set_title(f'State {i + 1}')
            handles, labels = axs[0, i-plot_cols].get_legend_handles_labels()
    fig.legend(handles, labels, loc=(0.65, 0), ncol=3)
    fig.supxlabel('Macrostate Similarity')
    fig.supylabel('Count of Clusterings')
    plt.savefig(save_dir)


def plot_histograms_from_file(sim_mat_file, shape, nbins):
    sim_mat_flat = np.loadtxt(sim_mat_file, dtype=float, comments='#')
    sim_mat = sim_mat_flat.reshape(shape)
    folder = str(pathlib.Path(sim_mat_file).parent)
    plot_state_sim_histograms(
        sim_mat,
        nbins,
        folder + '/comparison_state_similarity_histograms.pdf'
    )


if __name__ == '__main__':
    reference = '/data/MPP_MC/Lukas/Datasets/HP35_reference_data/'\
        'contacts.gaussian10f_microstates_pcs5_p153_linkage.dat'\
        '_q.pop0.005_qmin0.50.macrotraj'
    nbins = 100
    folder = '/data/MPP_MC/Lukas/Datasets/MPT_MCMC_FNC/'\
        'MPT_MCMC_var_0.05_exp2_cut0.15'
    compare_multiple_macrotrajs(reference, folder, nbins)
