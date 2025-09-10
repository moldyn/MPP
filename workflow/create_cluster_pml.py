#!/usr/bin/env python3

import __main__

__main__.pymol_argv = ["pymol", "-cq"]

import pymol

pymol.finish_launching()

from pymol import cmd

import numpy as np
import argparse
import yaml
import itertools
import os


COLORS = [
    "0xe9c46a",
    "0x264653",
    "0xe76f51",
    "0x2a9d8f",
    "0xf4a261",
    "0x264653",
    "0xe76f51",
]


def setup_pymol():
    cmd.bg_color("white")
    settings = [
        ("ray_trace_fog", 0),
        ("ray_trace_mode", 1),
        ("ray_trace_gain", 1),
        ("ray_trace_slope", 50),
        ("light_count", 0),
        ("ray_texture", 0),
        ("antialias", 5),
        ("ambient", 1),
        ("dash_gap", 0),
        ("dash_radius", 0.15),
        ("depth_cue", 0),
        ("sphere_scale", 0.3),
        ("sphere_quality", 10),
        ("cartoon_discrete_colors", "on"),
    ]
    for name, value in settings:
        cmd.set(name, value)


def setup_structures(cfg: dict, structure: str) -> list:
    cmd.load(structure, "structure")
    cmd.split_states("structure")
    cmd.delete("structure")
    cmd.load(os.path.join(cfg["source"], cfg["topology file"]), "reference")
    objects = cmd.get_object_list(selection="(all)")[:-1]
    for obj in objects:
        cmd.align(obj, "reference")
    return objects


def parse_cluster_selecton(cluster_string: str) -> list:
    """Returns a list containing the selected cluster indices"""
    clusters_list = cluster_string.split(",")
    clusters = list()
    for arg in clusters_list:
        if "-" in arg:
            split_arg = arg.split("-")
            clusters += list(range(int(split_arg[0]), int(split_arg[1]) + 1))
        else:
            clusters.append(int(arg))
    return clusters


def setup_cluster(obj, cluster, color, ndx):
    for contact in cluster:
        r1, r2 = ndx[contact]
        cmd.distance(f"dist{contact}_{obj}", f"{obj}///{r1}/CA", f"{obj}///{r2}/CA")
        cmd.hide("labels", f"dist{contact}_{obj}")
        cmd.set("dash_color", color, f"dist{contact}_{obj}")
        cmd.select(f"CA{contact}_{obj}", f"{obj}///{r1}/CA or {obj}///{r2}/CA")
        cmd.show("spheres", f"CA{contact}_{obj}")


def delete_cluster(obj, cluster, color, ndx):
    for contact in cluster:
        r1, r2 = ndx[contact]
        cmd.delete(f"dist{contact}_{obj}")
        cmd.hide("spheres", f"CA{contact}_{obj}")
        cmd.delete(f"CA{contact}_{obj}")


def save_cluster(f, cluster, color, ndx):
    for contact in cluster:
        r1, r2 = ndx[contact]
        f.writelines(
            f"distance dist{contact}, {r1}/CA, {r2}/CA \n"
            f"hide labels, dist{contact} \n"
            f"set dash_color, {color}, dist{contact} \n"
            f"select CA{contact}, {r1}/CA or {r2}/CA \nshow spheres, CA{contact} \n"
        )


def set_view(cfg):
    cmd.set_view(tuple(np.loadtxt(os.path.join(cfg["source"], cfg["view"]))))


def write_distances_script(filename, clusters, ndx):
    colors = list(itertools.islice(itertools.cycle(COLORS), len(clusters)))
    with open(filename, "w") as f:
        for (cluster_idx, cluster), color in zip(clusters, colors):
            f.writelines(f"\n \n # Cluster {cluster_idx} \n \n")
            save_cluster(
                f,
                cluster,
                color,
                ndx,
            )


def create_images(output, cfg, structure, clusters, ndx):
    colors = list(itertools.islice(itertools.cycle(COLORS), len(clusters)))
    objects = setup_structures(cfg, structure)
    setup_pymol()
    cmd.color("0xdddfe5", "all")
    cmd.hide("all")
    set_view(cfg)
    for i, obj in enumerate(objects):
        # cmd.reinitialize()
        # setup_pymol()
        # setup_structures(cfg, structure)
        # cmd.color("0xdddfe5", "all")
        # cmd.hide("all")
        # set_view(cfg)

        cmd.show("cartoon", f"{obj} and polymer")
        for (cluster_idx, cluster), color in zip(clusters, colors):
            setup_cluster(obj, cluster, color, ndx)
        cmd.png(
            os.path.join(output, f"{i:02d}.png"), cfg["width"], cfg["height"], ray=1
        )
        cmd.hide("cartoon", f"{obj} and polymer")
        for (cluster_idx, cluster), color in zip(clusters, colors):
            delete_cluster(obj, cluster, color, ndx)


def parse_args():
    parser = argparse.ArgumentParser(
        prog="Create a PyMol script to draw contact clusters",
        description=(
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
        "structure_file",
        help="PDB file which may contain several models",
    )
    parser.add_argument(
        "clusters",
        help="Which clusters to draw. A list like '1-3,5,8'",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    # List of lists containing the contacts of each cluster
    clusters = []
    with open(os.path.join(cfg["source"], cfg["cluster file"]), "r") as f:
        for line in f:
            if line.startswith("#"):
                continue
            clusters.append([int(i) for i in line.split()])

    # Definition of the contacts (list of residue pairs)
    ndx = np.loadtxt(os.path.join(cfg["source"], cfg["contact index file"]), dtype=int)

    # Set of selected clusters
    cluster_selection = parse_cluster_selecton(args.clusters)

    # write_distances_script(
    #     os.path.join(args.output, "draw_distances.pml"),
    #     [(i, clusters[i - 1]) for i in cluster_selection],
    #     ndx,
    # )

    create_images(
        args.output,
        cfg,
        args.structure_file,
        [(i, clusters[i - 1]) for i in cluster_selection],
        ndx,
    )


if __name__ == "__main__":
    main()
