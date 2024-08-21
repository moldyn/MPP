import click
import msmhelper as mh
import numpy as np


@click.command(
    no_args_is_help='-h',
    help='Plot average formed contacts',
)
@click.option(
    '--is-contacts',
    'iscontactfile',
    required=True,
    type=click.Path(exists=True),
    help='Path to contacts file',
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
def main(iscontactfile, lower_threshold, upper_threshold=1):
    # load files
    is_contacts = mh.opentxt(iscontactfile, usecols=2)
    idxs = mh.opentxt(iscontactfile, usecols=(0, 1))  # starting from 1

    # draw rectangles to highlight native contacts
    selected_idxs = [
        (i, j)
        for (i, j), is_contact in zip(idxs, is_contacts)
        if is_contact >= lower_threshold and is_contact <= upper_threshold
    ]

    mh.savetxt(
        f'{iscontactfile}.thr{lower_threshold:g}-{upper_threshold:g}.ndx',
        selected_idxs,
        fmt='%.0f',
    )


if __name__ == '__main__':
    main()
