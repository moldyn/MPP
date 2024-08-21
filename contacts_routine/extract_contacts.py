import click
import numpy as np
import msmhelper as mh
import prettypyplot as pplt
from matplotlib import pyplot as plt
from tqdm import tqdm

pplt.use_style()
CUTOFF = 0.45


@click.command(
    no_args_is_help='-h',
    help='Mindist computes the minimal distance between pairs of residues.',
)
@click.option(
    '--contacts',
    'contactfile',
    required=True,
    type=click.Path(exists=True),
    help='Path to all atom-pairwise contact file.',
)
@click.option(
    '--index',
    'indexfile',
    required=True,
    type=click.Path(exists=True),
    help='Path to 4 column index file of type (resi resj atomi atomj).',
)
@click.option(
    '--output',
    '-o',
    required=True,
    type=click.Path(),
    help='Path to output file',
)
@click.option(
    '--lower-threshold',
    required=True,
    type=click.FloatRange(min=0, max=1),
    help='Lower threshold of fraction of formed contacts to be selected.',
)
@click.option(
    '--upper-threshold',
    required=False,
    default=1,
    type=click.FloatRange(min=0, max=1),
    help='Upper threshold of fraction of formed contacts to be selected.',
)
def main(contactfile, indexfile, output, lower_threshold, upper_threshold=1):
    # load files
    indices_raw = np.loadtxt(indexfile, dtype=int)

    # convert indices in tuples
    indices = np.empty((len(indices_raw), 2), dtype=object)
    indices[:] = [
        ((resi, resj), (atomi, atomj))
        for resi, resj, atomi, atomj in indices_raw
    ]
    res_pairs = np.unique(indices[:, 0])

    contact_indices_per_res_pair = {
        res_pair: [
            idx
            for idx, ((resi, resj), _) in enumerate(indices)
            if resi == res_pair[0] and resj == res_pair[1]
        ]
        for res_pair in res_pairs
    }

    contact_is_formed_count = np.zeros(len(indices), dtype=int)
    for idx, distances in enumerate(load_txt_gen(filename=contactfile)):
        contact_is_formed_count += (distances <= CUTOFF).astype(int)
    n_frames = idx + 1

    contact_is_formed = contact_is_formed_count / n_frames
    mh.savetxt(
        f'{contactfile}.is_formed',
        contact_is_formed,
        fmt='%.5f',
        header=(
            'Fraction of formed contacts (r < 0.45nm)'
        ),
    )

    # loop over each residue pair
    selected_contact_indices_per_res_pair = {}
    for res_pair in res_pairs:
        atom_idxs = contact_indices_per_res_pair[res_pair]
        formed_fraction = contact_is_formed[atom_idxs]
        idx_sort = np.argsort(formed_fraction)[::-1]

        selected_atom_idxs = np.array(
            atom_idxs,
        )[(formed_fraction >= lower_threshold) & (formed_fraction <= upper_threshold)]
        if len(selected_atom_idxs):
            selected_contact_indices_per_res_pair[
                res_pair
            ] = selected_atom_idxs
        print(f'res {res_pair[0]:>2.0f}-{res_pair[1]:>2.0f}')
        for contact, formed in zip(
            indices[:, 1][atom_idxs][idx_sort],
            formed_fraction[idx_sort],
        ):
            if formed < 1e-2:
                break
            print(
                f'    atom {contact[0]:>3.0f}-{contact[1]:>3.0f}: '
                f'{formed:.4f}',
            )

    # get selected residue pairs
    selected_res_pairs = [
        res_pair
        for res_pair in res_pairs
        if res_pair in selected_contact_indices_per_res_pair
    ]
    mh.savetxt(
        f'{output}.ndx',
        selected_res_pairs,
        header=(
            'residue indices, where for each residue the all atom pairs '
            f'formed more than {lower_threshold:g} and less than '
            f'{upper_threshold:g} were selected',
        ),
        fmt='%.0f',
    )

    with open(output, 'w') as file_output:
        file_output.write(
            '# minimal distances where for each residue the atom pairs '
            f'formed more than {lower_threshold:g} and less than '
            f'{upper_threshold:g} were selected\n',
        )
        for distances in load_txt_gen(filename=contactfile):
            distances = [
                distances[
                    selected_contact_indices_per_res_pair[res_pair]
                ].min()
                for res_pair in selected_res_pairs
            ]
            file_output.write(
                ' '.join([f'{d:.5f}' for d in distances]),
            )
            file_output.write('\n')


def load_txt_gen(*, filename, dtype=np.float64, comments=('#', '@')):
    """Return all non comment lines as generator."""
    with open(filename) as file_obj:
        for line in tqdm(file_obj):
            if line.startswith(comments):
                continue
            yield np.array(line.split()).astype(dtype)


if __name__ == '__main__':
    main()
