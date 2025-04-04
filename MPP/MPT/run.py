#!/usr/bin/env python

import os
import yaml
from pathlib import Path
import argparse

import numpy as np
import MPT
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
        self.helices = np.loadtxt(os.path.join(self.source, self.d["helices"]), dtype=int)

        self.tlag = self.d["tlag"]
        self.pop_min = self.d["pop_min"]
        self.q_min = self.d["q_min"]

        self.lumping_dir = None
        self.kernel = None
        self.feature_kernel = None
        self.mpp = None

        self.n_random_frames = 20
        self.use_ref = True

    def perform_mpp(self, out, overwrite=False):
        """out: Z.npy"""
        if os.path.exists(out) and not overwrite:
            self.mpp.from_Z(out)
        else:
            Path(os.path.dirname(out)).mkdir(parents=True, exist_ok=True)
            self.mpp.mpt(
                self.kernel,
                feature_kernel=self.feature_kernel,
            )
            self.mpp.save_Z(out)

    def get_rmsd(self, out, overwrite=False):
        """out: rmsd.npy"""
        if os.path.exists(out) and not overwrite:
            self.mpp.load_rmsd(out)
        else:
            self.mpp.save_save(out)


### RUN ######################################################################

def setup_mpp(dij, gij, data):
    kernel = MPT.kernel.MPTKernel(similarity=dij)
    if gij == "none":
        feature_kernel = None
    elif gij == "q":
        feature_kernel = MPT.kernel.FeatureKernel(
            data.feature_traj,
            data.microtraj,
        )
    elif gij == "JS":
        feature_kernel = MPT.kernel.MultiFeatureKernel(
            data.mfeature_traj,
            data.microtraj,
        )
    else:
        raise ValueError("feature kernel must be None, q or JS.")

    data.kernel = kernel
    data.feature_kernel = feature_kernel

    data.mpp = MPT.MPT(
        data.microtraj,
        data.tlag,
        data.feature_traj,
        macrostate_thresholds=(data.pop_min, data.q_min),
        limits=data.limits,
        quiet=True,
    )
    data.mpp.topology_file = data.top
    data.mpp.xtc_trajectory_file = data.xtc
    return data


# def mpp(mpt, data):
#     """Performs MPP and saves Z matrix"""
#     mpt.mpt(
#         data.kernel,
#         feature_kernel=data.feature_kernel,
#     )
#     mpt.save_Z(os.path.join(data.lumping_dir, "Z.npy"))
#     return mpt

def plot(data, out, kind="dendrogram", scale=1):
    """
    kind: dendrogram, timescales, sankey, contacts, macrotraj, ck_test, rmsd
    """
    if kind == "dendrogram":
        data.mpp.plot(out, scale=scale)
    elif kind == "timescales":
        data.mpp.plot_implied_timescales(out, scale=scale, frame_length=0.2, use_ref=data.use_ref)
    elif kind == "sankey":
        data.mpp.plot_sankey(out, scale=scale)
    elif kind == "contacts":
        data.mpp.plot_contact_rep(data.mtraj_raw, data.cluster, out, scale=scale)
    elif kind == "macrotraj":
        data.mpp.plot_macrotraj(out, row_length=1/10)
    elif kind == "ck_test":
        data.mpp.plot_ck_test(out, frame_length=0.2)
    elif kind == "rmsd":
        data.mpp.plot_rmsd(out, frame_length=0.2)


def draw_random_frames(mpt, data):
    if mpt.Z is None:
        mpt.from_Z(os.path.join(data.lumping_dir, "Z.npy"))
    Path(os.path.join(data.lumping_dir + "random_frames/")).mkdir(parents=True, exist_ok=True)
    mpt.topology_file = data.top
    mpt.xtc_trajectory_file = data.xtc
    mpt.draw_random_frames(os.path.join(data.lumping_dir + "random_frames/"), n=data.n_random_frames)
    return mpt

def write_random_frames_indices(mpt, out, n):
    Path(os.path.join(out)).mkdir(parents=True, exist_ok=True)
    mpt.draw_random_frames_indices(out, n)

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
    # parser.add_argument(
    #     "lumping_grid",
    #     help=(
    #         "yaml file defining the lumpings to perform and where to store "
    #         "them."
    #     ),
    #     type=argparse.FileType('r', encoding='latin-1'),
    # )
    parser.add_argument(
        "d",
        # "-d",
        # "--dij",
        help=(
            "dij to be used."
        )
    )
    parser.add_argument(
        "g",
        # "-g",
        # "--gij",
        help=(
            "gij to be used."
        )
    )
    parser.add_argument(
        "-o",
        "--out",
        help=(
            "Override output directory set by config file"
        ),
        # nargs="+",
    )
    parser.add_argument(
        "-Z",
        help="Perform MPP and write the Z matrix.",
        # action="store_true",
    )
    parser.add_argument(
        "-r",
        "--draw-random",
        help="Draw N random frames for each macrostate",
        metavar="N",
        type=int,
    )
    parser.add_argument(
        "-p",
        "--plot",
        # nargs="+",
        help="Generate listed plots. Possible arguments include dendrogram, contacts, sankey, rmsd, macrotraj, timescales and more. (not yet implemented)"
    )
    return parser.parse_args()
    

def main():
    args = parse_args()

    # Parse input files
    data = Data(args.data_specification.name)
    # if args.o:
    #     data.out = args.out
    data = setup_mpp(args.d, args.g, data)
    data.perform_mpp(args.Z)

    # for p in args.plot:
    if args.plot:
        plot(data, args.out, kind=args.plot)

    if args.draw_random:
        write_random_frames_indices(data.mpp, args.out, args.draw_random)



    # with open(args.lumping_grid.name, "r") as f:
    #     lumpings = yaml.safe_load(f)

    # mpts = [None] * len(lumpings)
    # if args.Z:
    #     mpts = process_lumpings(lumpings, data, mpp, mpts)
    # if args.standard_plots:
    #     mpts = process_lumpings(lumpings, data, standard_plots, mpts)
    # if args.draw_random:
    #     data.n_random_frames = args.draw_random
    #     mpts = process_lumpings(lumpings, data, draw_random_frames, mpts)

if __name__ == "__main__":
    main()
