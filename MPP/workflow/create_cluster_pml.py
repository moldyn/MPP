#!/usr/bin/env python3
import numpy as np
import argparse
import yaml


PYMOL_HEADER = """reset; cd /data/HP35_contactsVSdihedrals/Contacts/mosaic/CPM0.78; reinitialize; load villin_360K.pdb; set cartoon_discrete_colors, on; util.cbss("all","white","white","white",_self=cmd); bg_color white;
set ray_trace_fog, 0
set ray_trace_mode, 1
set ray_trace_gain, 1
set set_ray_trace_slope, 50
set light_count, 0
set ray_texture, 0
set antialias, 5
set ambient, 1
set dash_gap, 0
set dash_radius, .15
set depth_cue, 0
set sphere_scale, 0.3
set sphere_quality, 10
color 0xdddfe5, resi 1-35
set_view (    -0.092613213,   -0.974311054,   -0.205098271,    -0.811263561,   -0.045582026,    0.582863152,    -0.577268422,    0.220377758,   -0.786223531,     0.001260591,   -0.000029638,  -64.085952759,     8.856776237,   -0.973796844,   66.159248352,  -199.907592773,  328.409393311,  -20.000000000 )
"""


def parse_cluster_selecton(cluster_string):
    clusters_list = cluster_string.split(",")
    clusters = set()
    for arg in clusters_list:
        if "-" in arg:
            split_arg = arg.split("-")
            clusters.update(range(int(split_arg[0]), int(split_arg[0]) + 1))
        else:
            clusters.add(int(arg))
    return clusters


def save_cluster(pymol, cluster, coords, label, color, ndx):
    pymol.writelines(f"\n \n # Cluster {label} \n \n")
    for num, coord in enumerate(cluster, coords):
        C1, C2 = ndx[coord]
        pymol.writelines(
            f"distance dist{num}, {C1}/CA, {C2}/CA \n"
            f"hide labels, dist{num} \n"
            f"set dash_color, {color}, dist{num} \n",
        )
        pymol.writelines(
            f"select CA{num}, {C1}/CA or {C2}/CA \nshow spheres, CA{num} \n",
        )

    coordinate_str = ", ".join(
        [f"$r_{{{ndx[coord][0]}, {ndx[coord][1]}}}$" for coord in cluster]
    )
    print(coordinate_str)


def estimate_pymol(filename, clusters, ndx, ncs):
    ticks = np.cumsum([len(c) for c in clusters])
    ticks = [0, *ticks[:-1]]
    # colors = ['#ef476f', '#ffd166', '#06d6a0', '#118ab2']
    # colors = ['#e30b5d', '#f2ce49', '#00a877', '#367588']
    colors = ["#db3575", "#e2ca71", "#519a8e", "#554bb4"]
    colors = ["#7ab5cd", "#f4a261", "#2a9d8f", "#e76f51"]
    colors = [
        "#e9c46a",
        "#264653",
        "#e76f51",
        "#2a9d8f",
        "#f4a261",
        "#264653",
        "#e76f51",
    ]
    colors = [colors[idx % len(colors)] for idx in range(ncs)]
    colors = [f"0x{c[1:]}" for c in colors]

    with open(f"{filename}.pml", "w") as pymol:
        for numf in np.arange(0, len(clusters), len(colors)):
            pymol.writelines(PYMOL_HEADER)
            for numc, color in enumerate(colors):
                if numc + numf >= len(clusters):
                    break
                save_cluster(
                    pymol,
                    clusters[numf + numc],
                    ticks[numf + numc],
                    numf + numc + 1,
                    color,
                    ndx,
                )
            pymol.writelines(
                f"\nray 500, 500\npng {filename}.pymol{numf}.png\n\n",
            )


def parse_args():
    parser = argparse.ArgumentParser(
        prog="Create a PyMol script to draw contact clusters",
        decription=(
            "This script draws the contacts of given clusters into a "
            "pdb structure and stores a png file."
        ),
    )
    parser.add_argument(
        "output",
        help="Output directory",
    )
    parser.add_argument(
        "config",
        help="Config file (yml)",
    )
    parser.add_argument(
        "structures",
        help="PDB file which may contain several models",
    )
    parser.add_argument(
        "model",
        help="If the PDB contains multiple models, select one",
    )
    parser.add_argument(
        "clusters",
        help="Which clusters to draw. A list like '1-3,5,8'",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    clusters = []
    with open(cfg["source"] + cfg["cluster file"], "r") as f:
        for line in f:
            if line.startswith("#"):
                continue
            clusters.append([int(i) for i in line.split()])

    ndx = np.loadtxt(cfg["source"] + cfg["contact index file"], dtype=int)

    for clusterfile in (
        "hp35.selected_contacts.gaussian10f.cor.CPM0.78.clusters.sorted",
        "hp35.selected_contacts.gaussian10f.cor.CPM0.78.clusters.sorted2",
    ):
        clusters = np.array(
            [
                [int(val) for val in cluster.split()]
                for cluster in np.loadtxt(
                    clusterfile,
                    delimiter="\n",
                    dtype=str,
                )
            ]
        )
        estimate_pymol(clusterfile, clusters, ndx, 7)


if __name__ == "__main__":
    main()
