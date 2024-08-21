import numpy as np
import msmhelper as mh
import pathlib
from tqdm import tqdm
import multiprocessing as mp


def read_traj(traj):
    return np.loadtxt(traj, dtype='int', comments='#')


def traj_name_list(folder_dir):
    folder = pathlib.Path(folder_dir)
    traj_names = list(folder.glob("*.macrotraj"))
    return traj_names


def compare_trajs(traj_names, i, j):
    traj1 = read_traj(traj_names[i])
    traj2 = read_traj(traj_names[j])
    comparison = mh.md.compare_discretization(traj1, traj2, method='symmetric')
    return i, j, comparison


def comparison_matrix(folder_dir):
    traj_names = traj_name_list(folder_dir)
    n = len(traj_names)
    comp_matrix = np.identity(n)

    # Create a pool of worker processes
    num_workers = mp.cpu_count()
    pool = mp.Pool(processes=num_workers)

    tasks = []
    for i in tqdm(range(n)):
        for j in range(i, n):  # Only compare each pair once (symmetric matrix)
            if comp_matrix[i, j] == 0:
                tasks.append((traj_names, i, j))

    results = list(tqdm(pool.starmap(compare_trajs, tasks), total=len(tasks)))

    # Update the comparison matrix with the results
    for i, j, comparison in results:
        comp_matrix[i, j] = comparison
        comp_matrix[j, i] = comparison

    pool.close()
    pool.join()

    np.savetxt(
        folder_dir + '/macrotraj_compmat',
        comp_matrix, delimiter=' ',
        fmt='%.15f'
    )
    return comp_matrix


if __name__ == '__main__':
    directory_stochastic_trajs = '/data/MPP_MC/Lukas/Datasets/MCMC_test_HP35'
    comparison_stochastic_macrotrajs = comparison_matrix(
        directory_stochastic_trajs)
