import numpy as np
import msmhelper as mh
import pathlib
from tqdm import tqdm


def read_traj(traj):
    return np.loadtxt(traj, dtype='int', comments='#')


def traj_name_list(folder_dir):
    """Generate list of macrotrajs in folder

    Args:
        folder_dir: directory of folder

    Returns:
        traj_names: list of all specific filenames in folder
    """
    folder = pathlib.Path(folder_dir)
    traj_names = list(folder.glob("*.macrotraj"))
    return traj_names


def compare_trajs(traj_names):
    """Compare similarity of each macrotraj to another

    Args:
        traj_names: all macrotrajs to be compared

    Returns:
        matrix: comparison matrix
    """
    n = len(traj_names)
    comp_matrix = np.identity(n)
    for i in tqdm(range(n)):
        traj1 = read_traj(traj_names[i])
        for j in range(n):
            if comp_matrix[i, j] == 0 and comp_matrix[j, i] == 0:
                traj2 = read_traj(traj_names[j])
                comparison = mh.md.compare_discretization(
                    traj1, traj2, method='symmetric'
                )
                comp_matrix[i, j] = comparison
                comp_matrix[j, i] = comparison
    return comp_matrix


def comparison_matrix(folder_dir):
    """Run macrotraj comparison from their parent folder

    Args:
        folder_dir: directory of folder
    """
    traj_names = traj_name_list(folder_dir)
    comp_matrix = compare_trajs(traj_names)
    np.savetxt(
        folder_dir + '/macrotraj_compmat',
        comp_matrix, delimiter=' ',
        fmt='%.15f'
    )


if __name__ == '__main__':
    directory_stochastic_trajs = '/data/MPP_MC/Lukas/Datasets/MCMC_test_HP35'
    comparison_stochastic_macrotrajs = comparison_matrix(
        directory_stochastic_trajs)
