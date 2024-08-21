import itertools

import click
import numpy as np
import mdtraj as md
from tqdm import tqdm


@click.command(
    no_args_is_help='-h',
    help='Mindist computes the minimal distance between pairs of residues.',
)
@click.option(
    '--trajectory',
    '-f',
    'trajfile',
    required=True,
    type=click.Path(exists=True),
    help='Path to trajectory file (.xtc)',
)
@click.option(
    '--topology',
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
def main(trajfile, topfile, ndxfile, output):
    # load files
    index_pairs = np.loadtxt(ndxfile, dtype=int) - 1

    # load topology
    top = md.load(topfile).topology

    # convert residue indices to heavy atoms
    atoms_per_res = {
        index: [
            atom.index for atom in top.residue(index).atoms
            if atom.element.symbol != 'H'  # heavy atom
        ]
        for index in np.unique(index_pairs)
    }

    atom_pairs = [
        list(
            itertools.product(
                atoms_per_res[i], atoms_per_res[j],
            ),
        ) for i, j in index_pairs
    ]

    index_output = np.concatenate([
        [
            (res_i, res_j, atom_i, atom_j)
            for atom_i, atom_j in atom_pair
        ]
        for (res_i, res_j), atom_pair in zip(index_pairs, atom_pairs)
    ])
    np.savetxt(
        f'{output}.atom_indices',
        index_output + 1,
        fmt='%.0f',
        header='res_i res_j atom_i atom_j',
    )

    # flatten atom_pairs to loop over all
    atom_pairs = np.concatenate(atom_pairs)

    with open(output, 'w') as ostream:
        for distances in compute_distances(
            trajfile, topfile, atom_pairs,
        ):
            ostream.write(
                ' '.join([f'{d:.5f}' for d in distances]) + '\n',
            )


def load_xtc(trajfile, topfile):
    top = md.load_topology(topfile)
    with md.open(trajfile) as xtc:
        while (
            frame := xtc.read_as_traj(top, n_frames=1)
        ).n_frames:
            yield frame


def compute_distances(trajfile, topfile, atom_pairs):
    for frame in tqdm(load_xtc(trajfile, topfile)):
        yield md.compute_distances(
            frame,
            atom_pairs=atom_pairs,
        )[0].flatten()


if __name__ == '__main__':
    main()
