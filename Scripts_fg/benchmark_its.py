import click
import numpy as np
import msmhelper as mh
import pathlib
import matplotlib.pyplot as plt
from tqdm import tqdm
from benchmark_macrostates import compare_multiple_macrotrajs


@click.command(no_args_is_help='-h')
@click.option(
    '--trajs-folder',
    'trajs_folder',
    required=True,
    type=click.Path(exists=True),
    help='Folder of macrostate trajs',
)
@click.option(
    '--traj-benchmark',
    'traj_benchmark_dir',
    required=True,
    type=click.Path(exists=True),
    help='Traj of benchmark clustering',
)
@click.option(
    '--tlag',
    required=True,
    type=click.IntRange(min=1),
    help='Lagtime in frames',
)
@click.option(
    '--ntimescales',
    default=3,
    type=click.IntRange(min=1),
    help='Number of timescales for benchmarking'
)
@click.option(
    '--nbins',
    default=20,
    type=click.IntRange(min=1),
    help='Bins for its histogram'
)
def traj_comparison(
    trajs_folder,
    traj_benchmark_dir,
    tlag,
    ntimescales,
    nbins
):
    traj_names = trajs_names(trajs_folder)
    traj_benchmark = read_traj(traj_benchmark_dir)
    (
        its_mat,
        its_sum,
        traj_sim,
    ) = compare_trajs(traj_names, tlag, ntimescales, traj_benchmark)
    np.savetxt(trajs_folder + '/comparison_its_mat',
               its_mat, delimiter=' ',
               fmt='%.5f'
               )
    np.savetxt(trajs_folder + '/comparison_its_sum',
               its_sum, delimiter=' ',
               fmt='%.5f'
               )
    np.savetxt(trajs_folder + '/comparison_discretization',
               traj_sim, delimiter=' ',
               fmt='%.5f'
               )
    its_histogram(its_mat,
                  its_sum,
                  nbins,
                  trajs_folder + '/comparison_its_histogram.pdf'
                  )
    overview = [[f'mean its: {np.mean(its_sum):.3f}'],
                [f'max its: {np.max(its_sum):.3f} {np.argmax(its_sum)}'],
                [f'min its: {np.min(its_sum):.3f} {np.argmin(its_sum)}'],
                [f'mean sim: {np.mean(traj_sim):.3f}'],
                [f'max sim: {np.max(traj_sim):.3f} {np.argmax(traj_sim)}'],
                [f'min sim: {np.min(traj_sim):.3f} {np.argmin(traj_sim)}'],
                [f'sim(max its): {traj_sim[np.argmax(its_sum)]:.3f}'],
                [f'its(max sim): {its_sum[np.argmax(traj_sim)]:.3f}']]
    np.savetxt(trajs_folder + '/comparison_overview',
               overview, delimiter=' ',
               fmt='%s'
               )
    compare_multiple_macrotrajs(traj_benchmark_dir, trajs_folder, nbins)


def read_traj(traj):
    return np.loadtxt(traj, dtype='int', comments='#')


def trajs_names(trajs_folder):
    """Generate list of macrotrajs in folder

    Args:
        folder_dir (str): directory of folder

    Returns:
        traj_names: list of all specific filenames in folder
    """
    folder = pathlib.Path(trajs_folder)
    traj_names = list(folder.glob("*.macrotraj"))
    return traj_names


def compare_trajs(traj_names, tlag, ntimescales, traj_benchmark):
    """Compares trajectories to reference by similarity and implied timescales

    Args:
        traj_names: filenames oftrajectories to compare
        lagtime: lagtime to calculate timescales
        ntimescales: number of timescales calculated
        traj_benchmark: filename of reference trajectory

    Returns:
        tuple(its_mat, its_sum, sim): benchmarked traj similarities and its
    """
    shape = [len(traj_names), ntimescales]
    its_mat = np.empty(shape)
    its_sum = np.empty(shape[0])
    traj_sim = np.empty(shape[0])
    its_benchmark = mh.msm.implied_timescales(
        traj_benchmark,
        tlag,
        ntimescales
    )
    for idx, traj_name in enumerate(tqdm(traj_names)):
        traj = read_traj(traj_name)
        its = mh.msm.implied_timescales(traj, tlag, ntimescales)
        its_mat[idx] = its / its_benchmark
        its_sum[idx] = np.mean(np.sum(its / np.sum(its_benchmark)))
        traj_sim[idx] = mh.md.compare_discretization(
            traj,
            traj_benchmark,
            method='symmetric'
        )
    return its_mat, its_sum, traj_sim


def its_histogram(its_mat, its_sum, nbins, save_dir):
    """Plot histograms for its of stochastic clusterings

    Args:
        its_mat: matrix of individual its
        its_sum: sum of its
        nbins: bins for plotting histogram
        save_dir: directory to save histogram
    """
    fig, axx = plt.subplots(2, 2, tight_layout=True, figsize=(4, 4))
    axs = axx.flatten()
    for i in range(4):
        if i < 3:
            axs[i].hist(
                its_mat[:, i],
                bins=nbins,
                color='blue',
                alpha=0.7,
                align='mid',
                label='its'
            )
            axs[i].set_title(f'its_{i+1}')
        else:
            axs[i].hist(
                its_sum,
                bins=nbins,
                color='blue',
                alpha=0.7,
                align='mid',
                label='its'
            )
            axs[i].set_title(r'mean(its(1, 2, 3)')
            handles, label = axs[i].get_legend_handles_labels()
    #    fig.legend(handles, label, loc=(0.65, 0), ncol=1)
    fig.supxlabel('Relative Implied Timescales')
    fig.supylabel('Count of Clusterings')
    plt.tight_layout()
    plt.savefig(save_dir)


def its_histogram_from_file(its_mat_dir, its_sum_dir, nbins):
    its_comparison = np.loadtxt(its_mat_dir, dtype=float, comments='#')
    its_sum = np.loadtxt(its_sum_dir, dtype=float, comments='#')
    folder = str(pathlib.Path(its_mat_dir).parent)
    save_dir = folder + '/comparison_its_histogram.pdf'
    its_histogram(its_comparison, its_sum, nbins, save_dir)


if __name__ == '__main__':
    traj_comparison()
