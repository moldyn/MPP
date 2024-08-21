from functools import partial
from multiprocessing import cpu_count, Pool, Lock

import click
import MDAnalysis
import numpy as np
from MDAnalysis.lib.distances import capped_distance, distance_array
from tqdm import tqdm

CUTOFF = 4.5  # [AA]


@click.command(
    no_args_is_help='-h',
    help='Mindist computes the minimal distance between pairs of residues.',
)
@click.option(
    '--traj',
    '-f',
    'trajfile',
    required=True,
    type=click.Path(exists=True),
    help='Path to trajectory file (.xtc)',
)
@click.option(
    '--top',
    '-s',
    'topfile',
    required=True,
    type=click.Path(exists=True),
    help='Path to topology file (.tpr or .pdb)',
)
@click.option(
    '--index',
    '-n',
    'ndxfile',
    required=True,
    type=click.Path(exists=True),
    help='Path to index file, 1 (!) indexed of shape (n, 2).',
)
@click.option(
    '--output',
    '-o',
    required=True,
    type=click.Path(),
    help='Path to output file',
)
@click.option(
    '--count-hydrogen',
    is_flag=True,
    help='Count hydrogen atoms.',
)
def main(trajfile, topfile, ndxfile, output, count_hydrogen):
    contact_pairs = np.loadtxt(ndxfile, dtype=int)
    n_contact_pairs = len(contact_pairs)

    select_atoms_str = (
        'resid {res}' if count_hydrogen else 'resid {res} and not type H'
    )

    universe = MDAnalysis.Universe(topfile, trajfile)
    atoms_by_res = {}
    residues = np.unique(contact_pairs)
    for res in residues:
        atoms_by_res[res] = universe.select_atoms(
            select_atoms_str.format(res=res),
        )

    # defince slices
    n_jobs = cpu_count()
    n_frames = universe.trajectory.n_frames
    n_blocks = n_jobs
    n_frames_per_slice = n_frames // n_blocks

    slices_values = [
        range(
            i * n_frames_per_slice,
            (i + 1) * n_frames_per_slice,
        )
        for i in range(n_blocks - 1)
    ]
    slices_values.append(
        range(
            (n_blocks - 1) * n_frames_per_slice,
            n_frames,
        )
    )

    # loop through trajectory
    run_per_slice = partial(
        cmap_per_traj_slice,
        blockslices=slices_values,
        universe=universe,
        contact_pairs=contact_pairs,
        atoms_by_res=atoms_by_res,
    )

    with Pool(
        n_jobs, initializer=tqdm.set_lock, initargs=(Lock(), ),
    ) as workers:
        result = workers.map(run_per_slice, np.arange(n_blocks))

    cmap = np.sum(
        result,
        axis=0,
    ) / n_frames

    np.savetxt(
        output,
        [
            f'{i:.0f} {j:.0f} {cmap[idx]:.5f}'
            for idx, (i, j) in enumerate(contact_pairs)
        ],
        header=(
            ''
            'FC = idx_i idx_j FC'
        ),
        fmt='%s',
    )


def cmap_per_traj_slice(
    slice_idx, blockslices, universe, contact_pairs, atoms_by_res,
):
    # set frame
    box = universe.dimensions
    cmap = np.zeros(len(contact_pairs))
    blockslice = blockslices[slice_idx]
    for _ in tqdm(
        universe.trajectory[blockslice.start:blockslice.stop],
        position=slice_idx,
        desc=f'process {slice_idx:>2.0f}',
        leave=False,
        mininterval=1,
    ):
        cmap += np.array([
            np.min(
                distance_array(
                    atoms_by_res[resi].positions,
                    atoms_by_res[resj].positions,
                    box=box,
                    backend='serial',
                ),
            ) <= CUTOFF
            for idx, (resi, resj) in enumerate(contact_pairs)
        ]).astype(int)
    return cmap


if __name__ == "__main__":
    main()
