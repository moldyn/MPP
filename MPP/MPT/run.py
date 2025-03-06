#!/usr/bin/env python

import os
import yaml
from pathlib import Path
import argparse

import numpy as np
import MPT
import msmhelper as mh
import matplotlib.pyplot as plt
import prettypyplot as pplt
from scipy.stats import pearsonr
from itertools import combinations
from tqdm import tqdm


class Data:
    def __init__(self, yaml_file):
        with open(yaml_file, "r") as f:
            self.d = yaml.safe_load(f)

        self.source = self.d["source"]
        self.out = self.d["out"]

        self.microtraj = np.loadtxt(os.path.join(
            self.source,
            self.d["microstate trajectory"]
        ), dtype=np.uint32)
        self.mtraj_raw = np.loadtxt(os.path.join(
            self.source,
            self.d["multi feature trajectory"],
        ))
        self.limits = None if self.d["limits"] is None else np.loadtxt(
            os.path.join(self.source, self.d["limits"]),
            dtype=np.int_,
        )
        self.mfeature_traj = self.mtraj_raw < 0.45
        self.feature_traj = self.mfeature_traj.mean(axis=1)
        self.cluster = os.path.join(self.source, self.d["cluster file"])

        self.top = os.path.join(self.source, self.d["topology file"])
        self.xtc = os.path.join(self.source, self.d["xtc file"])

        self.tlag = self.d["tlag"]
        self.pop_min = self.d["pop_min"]
        self.q_min = self.d["q_min"]

        self.lumping_dir = None
        self.kernel = None
        self.feature_kernel = None

        self.n_random_frames = 20
        self.use_ref = True



### RUN ######################################################################

def process_lumpings(lumpings, data, func, mpts=None):
    """Perform lumpings"""
    if mpts is None:
        mpts = [None] * len(lumpings)
    for i, lumping in tqdm(enumerate(lumpings)):
        print(f"Processing lumping {lumping}.")
        kernel = MPT.kernel.MPTKernel(similarity=lumpings[lumping]["kernel similarity"])
        data.use_ref = True
        if lumpings[lumping]["feature kernel"] is None:
            feature_kernel = 1
            if lumpings[lumping]["kernel similarity"] == "P":
                data.use_ref = False
        elif lumpings[lumping]["feature kernel"] == "fnc":
            feature_kernel = MPT.kernel.FeatureKernel(
                data.feature_traj,
                data.microtraj,
            )
        elif lumpings[lumping]["feature kernel"] == "JS":
            feature_kernel = MPT.kernel.MultiFeatureKernel(
                data.mfeature_traj,
                data.microtraj,
            )
        else:
            raise ValueError("feature kernel must be None, fnc or JS.")

        data.lumping_dir = os.path.join(data.out, lumping)
        Path(data.lumping_dir).mkdir(parents=True, exist_ok=True)
        data.kernel = kernel
        data.feature_kernel = feature_kernel


        if mpts[i] is None:
            mpts[i] = MPT.MPT(
                data.microtraj,
                data.tlag,
                data.feature_traj,
                macrostate_thresholds=(data.pop_min, data.q_min),
                limits=data.limits,
                quiet=True,
            )
        mpts[i] = func(mpts[i], data)
    return mpts


def mpp(mpt, data):
    """Performs MPP and saves Z matrix"""
    mpt.mpt(
        data.kernel,
        feature_kernel=data.feature_kernel,
    )
    mpt.save_Z(os.path.join(data.lumping_dir, "Z.npy"))
    return mpt


def standard_plots(mpt, data):
    if mpt.Z is None:
        mpt.from_Z(os.path.join(data.lumping_dir, "Z.npy"))
    out = data.lumping_dir
    Path(out).mkdir(parents=True, exist_ok=True)
    print("Plotting dendrogram...")
    mpt.plot(os.path.join(out + "dendrogram.pdf"), scale=1)
    print("Plotting implied timescales...")
    mpt.plot_implied_timescales(os.path.join(out + "timescales.pdf"), frame_length=0.2, use_ref=data.use_ref, scale=1)
    print("Plotting Sankey diagram...")
    mpt.plot_sankey(os.path.join(out + "sankey.pdf"), scale=1)
    print("Plotting contact representation...")
    mpt.plot_contact_rep(data.mtraj_raw, data.cluster, os.path.join(out + "contact_rep.pdf"), scale=1.4)
    print("Plotting trajectory...")
    mpt.plot_macrotraj(os.path.join(out + "macrotraj.pdf"), row_length=1/3)
    print("Performing Chapman Kolmogorov test...")
    mpt.plot_ck_test(os.path.join(out + "ck_test.pdf"), frame_length=0.2)
    return mpt

def draw_random_frames(mpt, data):
    if mpt.Z is None:
        mpt.from_Z(os.path.join(data.lumping_dir, "Z.npy"))
    Path(os.path.join(data.lumping_dir + "random_frames/")).mkdir(parents=True, exist_ok=True)
    mpt.topology_file = data.top
    mpt.xtc_trajectory_file = data.xtc
    mpt.draw_random_frames(os.path.join(data.lumping_dir + "random_frames/"), n=data.n_random_frames)
    return mpt


def parse_args():
    parser = argparse.ArgumentParser(
        prog="Perform MPP on MD simulation data",
        description=(
            "This program allows for the analysis of MD data utilizing the "
            "most probable path algorithm. It allows for easy plotting of "
            "different quality measures."
        ),
    )
    parser.add_argument(
        "data_specification",
        help=(
            "yaml file containing specification of files and parameters of "
            "the simulation"
        ),
        type=argparse.FileType('r', encoding='latin-1'),
    )
    parser.add_argument(
        "lumping_grid",
        help=(
            "yaml file defining the lumpings to perform and where to store "
            "them."
        ),
        type=argparse.FileType('r', encoding='latin-1'),
    )
    parser.add_argument(
        "-Z",
        help="Perform MPP and write the Z matrix.",
        action="store_true",
    )
    parser.add_argument(
        "-p",
        "--standard-plots",
        help="Plot standard plots for specified lumpings",
        action="store_true",
    )
    parser.add_argument(
        "-r",
        "--draw-random",
        help="Draw N random frames for each macrostate",
        metavar="N",
        type=int,
    )
    return parser.parse_args()
    

def main():
    args = parse_args()

    # Parse input files
    data = Data(args.data_specification.name)
    with open(args.lumping_grid.name, "r") as f:
        lumpings = yaml.safe_load(f)

    mpts = [None] * len(lumpings)
    if args.Z:
        mpts = process_lumpings(lumpings, data, mpp, mpts)
    if args.standard_plots:
        mpts = process_lumpings(lumpings, data, standard_plots, mpts)
    if args.draw_random:
        data.n_random_frames = args.draw_random
        mpts = process_lumpings(lumpings, data, draw_random_frames, mpts)

if __name__ == "__main__":
    main()
