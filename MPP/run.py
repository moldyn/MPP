#!/usr/bin/env python

import os
import warnings
import yaml
from pathlib import Path
import argparse

import numpy as np
import MPP


# Mapping from legacy space-separated keys to canonical snake_case keys.
_KEY_ALIASES = {
    "microstate trajectory": "microstate_trajectory",
    "multi feature trajectory": "multi_feature_trajectory",
    "contact threshold": "contact_threshold",
    "cluster file": "cluster_file",
    "contact index file": "contact_index_file",
    "topology file": "topology_file",
    "xtc file": "xtc_file",
    "frame length": "frame_length",
    "xtc stride": "xtc_stride",
    "n timescales": "n_timescales",
}


def _normalize_config(config):
    """Normalize legacy space-separated config keys to canonical snake_case.

    Parameters
    ----------
    config : dict
        Raw config dict loaded from YAML.

    Returns
    -------
    dict
        Config dict with all keys normalized to snake_case.

    Raises
    ------
    ValueError
        If both a legacy key and its canonical equivalent are present.
    """
    normalized = {}
    for key, value in config.items():
        canonical = _KEY_ALIASES.get(key, key)
        if canonical != key:
            if canonical in config:
                raise ValueError(
                    f"Duplicate config keys: '{key}' (legacy) and "
                    f"'{canonical}' (canonical) cannot both be specified."
                )
            warnings.warn(
                f"Config key '{key}' is deprecated; use '{canonical}' instead.",
                DeprecationWarning,
                stacklevel=3,
            )
        normalized[canonical] = value
    return normalized


OPTIONAL_PARAMS = [
    "cluster_file",
    "contact_index_file",
    "contact_threshold",
    "limits",
    "topology_file",
    "xtc_file",
    "helices",
    "frame_length",
    "view",
    "width",
    "height",
]

DEFAULTS = {k: None for k in OPTIONAL_PARAMS}
DEFAULTS["contact_threshold"] = 0.45


class Data:
    def __init__(self, yaml_file):
        with open(yaml_file, "r") as f:
            config = yaml.safe_load(f) or {}
        self.d = {**DEFAULTS, **_normalize_config(config)}

        self.source = self.d["source"]

        self.microstate_trajectory = np.loadtxt(
            os.path.join(self.source, self.d["microstate_trajectory"]), dtype=np.uint16
        )
        self.multi_state_trajectory_raw = np.loadtxt(
            os.path.join(
                self.source,
                self.d["multi_feature_trajectory"],
            ),
            ndmin=2,
        )
        self.limits = (
            None
            if self.d["limits"] is None
            else np.loadtxt(
                os.path.join(self.source, self.d["limits"]),
                dtype=np.int_,
            )
        )
        self.multi_feature_trajectory = self.multi_state_trajectory_raw < 0.45
        self.feature_trajectory = self.multi_feature_trajectory.mean(axis=1)

        self.cluster = None
        self.top = None
        self.xtc = None
        self.helices = None
        for file, param in [
            ("cluster_file", "cluster"),
            ("topology_file", "top"),
            ("xtc_file", "xtc"),
            ("helices", "helices"),
        ]:
            if self.d[file] is not None:
                setattr(self, param, os.path.join(self.source, self.d[file]))

        if self.helices is not None:
            self.helices = np.loadtxt(self.helices, dtype=int)

        self.frame_length = self.d["frame_length"]
        self.lagtime = self.d["lagtime"]
        self.pop_thr = self.d["pop_thr"]
        self.q_min = self.d["q_min"]

        self.lumping_dir = None
        self.kernel = None
        self.feature_kernel = None
        self.mpp = None

        self.n_random_frames = 20
        self.use_ref = True

    def prepare_mpp(self, dij, gij):
        if "stochastic" in self.d:
            kernel = MPP.kernel.LumpingKernel(
                method=self.d["stochastic"]["method"],
                param=self.d["stochastic"]["param"],
                similarity=dij,
            )
        else:
            kernel = MPP.kernel.LumpingKernel(
                similarity=dij,
            )

        if gij == "none":
            feature_kernel = None
        elif gij == "JS":
            feature_kernel = MPP.kernel.FeatureKernel(
                self.multi_feature_trajectory,
                self.microstate_trajectory,
            )
        else:
            raise ValueError("feature kernel must be None, q or JS.")

        if dij == "T" and gij == "none" and "stochastic" not in self.d:
            self.use_ref = False

        self.kernel = kernel
        self.feature_kernel = feature_kernel

    def setup_mpp(self, dij, gij):
        if dij != "gpcca":
            self.prepare_mpp(dij, gij)
        self.mpp = MPP.Lumping(
            self.microstate_trajectory,
            self.lagtime,
            self.multi_state_trajectory_raw,
            contact_threshold=self.d["contact_threshold"],
            pop_thr=self.pop_thr,
            q_min=self.q_min,
            limits=self.limits,
            quiet=True,
        )
        if self.top is not None and os.path.exists(self.top):
            self.mpp.topology_file = self.top
        if self.xtc is not None and os.path.exists(self.xtc):
            self.mpp.xtc_trajectory_file = self.xtc
        self.mpp.xtc_stride = self.d.get("xtc_stride", None)
        self.mpp.frame_length = self.frame_length

    def perform_mpp(self, out, overwrite=False):
        """
        Run MPP and save the Z matrix, or load it if it already exists.

        Parameters
        ----------
        out : str
            Path to save the Z matrix (e.g. ``Z.npy``).
        overwrite : bool
            If True, recompute even if the file already exists. (default False)
        """
        if os.path.exists(out) and not overwrite:
            print("Loading existing Z")
            self.mpp.load_Z(out)
        else:
            Path(os.path.dirname(out)).mkdir(parents=True, exist_ok=True)
            self.mpp.run_mpp(
                self.kernel,
                feature_kernel=self.feature_kernel,
                n=self.d["stochastic"]["n"] if "stochastic" in self.d else 1,
            )
            self.mpp.save_Z(out)

    def perform_gpcca(self, n_macrostates="ref", out=None, overwrite=False):
        """
        Run GPCCA lumping and optionally save the Z matrix.

        Parameters
        ----------
        n_macrostates : int or ``'ref'``
            Number of macrostates, or ``'ref'`` to use the count from the
            reference lumping (T). (default ``'ref'``)
        out : str, optional
            Path to save the Z matrix. If None, the result is not saved.
            (default None)
        overwrite : bool
            If True, recompute even if the file already exists. (default False)
        """
        if n_macrostates == "ref":
            n_macrostates = self.mpp.reference.n_macrostates[0]
        if out is not None and os.path.exists(out) and not overwrite:
            print("Loading existing Z")
            self.mpp.load_Z(out, gpcca=True)
        else:
            self.mpp.gpcca(n_macrostates)
            if out is not None:
                self.mpp.save_Z(out)

    def get_rmsd(self, out, overwrite=False):
        """
        Compute or load RMSD and save to file.

        Parameters
        ----------
        out : str
            Path to the RMSD file (e.g. ``rmsd.npy``).
        overwrite : bool
            If True, recompute even if the file already exists. (default False)
        """
        if not out.endswith(".npy"):
            out += ".npy"
        if os.path.exists(out) and not overwrite:
            self.mpp.load_rmsd(out)
        else:
            self.mpp.save_rmsd(out)


def plot(data, out, kind="dendrogram", scale=1):
    """
    Generate a plot of the requested kind from the given data.

    Parameters
    ----------
    data : Data
        Data object holding the MPP lumping and configuration.
    out : str
        Output file path for the plot.
    kind : str
        Type of plot to generate. One of: ``dendrogram``, ``timescales``,
        ``sankey``, ``contacts``, ``macrotraj``, ``ck_test``, ``rmsd``,
        ``delta_rmsd``, ``state_network``, ``macro_feature``,
        ``stochastic_state_similarity``, ``relative_implied_timescales``,
        ``transition_matrix``, ``transition_time``, ``macrostate_trajectory``.
        (default ``'dendrogram'``)
    scale : float
        Scaling factor for the plot. (default 1)
    """
    if kind == "dendrogram":
        data.mpp.plot.dendrogram(out, scale=scale, offset=0.0)
    elif kind == "timescales":
        if "n_timescales" in data.d:
            data.mpp.calc_timescales(data.d["n_timescales"])
        data.mpp.plot.implied_timescales(out, scale=scale, use_ref=data.use_ref)
    elif kind == "sankey":
        data.mpp.plot.sankey(out, scale=scale)
    elif kind == "contacts":
        data.mpp.plot.contact_rep(data.cluster, out, scale=scale)
    elif kind == "macrotraj":
        # trajectory_length = data.microstate_trajectory.shape[0]
        # n_macrostates = data.mpp.n_macrostates[0]
        # row_length = 1 / int(np.round(np.sqrt(trajectory_length) / (np.sqrt(n_macrostates) * 30)))
        row_length = 1 / 6
        if data.limits is not None:
            row_length = 1 / len(data.limits)
        data.mpp.plot.macrostate_trajectory(out, row_length=row_length)
    elif kind == "ck_test":
        data.mpp.plot.ck_test(out)
    elif kind == "rmsd":
        # data.get_rmsd(os.path.splitext(out)[0] + ".npy")
        data.get_rmsd(os.path.join(os.path.dirname(out), "rmsd_CA.npy"))
        data.mpp.plot.rmsd(out, helices=data.helices)
    elif kind == "delta_rmsd":
        data.get_rmsd(os.path.join(os.path.dirname(out), "rmsd_CA.npy"))
        data.mpp.plot.delta_rmsd(out, helices=data.helices)
    elif kind == "state_network":
        print("Plotting state network")
        data.mpp.plot.state_network(out)
    elif kind == "macro_feature":
        data.mpp.plot.macro_feature(out)
    elif kind == "stochastic_state_similarity":
        data.mpp.plot.stochastic_state_similarity(out)
    elif kind == "relative_implied_timescales":
        data.mpp.plot.relative_implied_timescales(out)
    elif kind == "transition_matrix":
        data.mpp.plot.transition_matrix(out)
    elif kind == "transition_time":
        data.mpp.plot.transition_time(out)
    elif kind == "macrostate_trajectory":
        data.mpp.save_macrostate_trajectory(out, one_based=True)
    else:
        raise ValueError(f"Unknown plot kind: {kind}")


def draw_random_frames(mpp, data):
    if mpp.Z is None:
        mpp.load_Z(os.path.join(data.lumping_dir, "Z.npy"))
    Path(os.path.join(data.lumping_dir + "random_frames/")).mkdir(
        parents=True, exist_ok=True
    )
    mpp.topology_file = data.top
    mpp.xtc_trajectory_file = data.xtc
    mpp.draw_random_frames(
        # os.path.join(data.lumping_dir + "random_frames/"), n=data.n_random_frames
        Path(data.lumping_dir) / "random_frames/",
        n=data.n_random_frames,
    )
    return mpp


def write_random_frames_indices(mpp, out, n):
    # Path(os.path.join(out)).mkdir(parents=True, exist_ok=True)
    mpp.draw_random_frames_indices(Path(out), n)


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
        type=argparse.FileType("r", encoding="latin-1"),
    )
    parser.add_argument("d", help=("dij to be used."))
    parser.add_argument("g", help=("gij to be used."))
    parser.add_argument(
        "-o",
        "--out",
        help=("Where to store the plot"),
    )
    parser.add_argument(
        "-Z",
        help="Perform MPP and write the Z matrix",
    )
    parser.add_argument(
        "--rmsd",
        help="Generate and write RMSD to file",
    )
    parser.add_argument(
        "--rmsd-feature",
        help="'CA' for C-alpha RMSD or 'feature' for feature RMSD (default: CA)",
        default="CA",
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
        help=(
            "Generate listed plots. Possible arguments include "
            "dendrogram, timescales, sankey, contacts, macrotraj, "
            "ck_test, rmsd, delta_rmsd, state_network, macro_feature, "
            "stochastic_state_similarity, relative_implied_timescales, "
            "transition_matrix, transition_time and "
            "macrostate_trajectory. The latter writes the macrostate "
            "trajectory to a txt file."
        ),
    )
    parser.add_argument(
        "--get-least-moving-residues",
        help="Write least moving residues for each macrostate to a file",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Parse input files
    data = Data(args.data_specification.name)
    data.setup_mpp(args.d, args.g)
    if args.d == "gpcca":
        data.perform_gpcca(args.g, args.Z)
    else:
        data.perform_mpp(args.Z)

    if args.rmsd:
        data.mpp.rmsd_feature = args.rmsd_feature
        data.mpp.rmsd_estimator = MPP.utils.argmedian
        data.get_rmsd(args.rmsd, overwrite=False)

    # for p in args.plot:
    if args.plot:
        plot(data, args.out, kind=args.plot)

    if args.draw_random:
        write_random_frames_indices(data.mpp, args.out, args.draw_random)

    if args.get_least_moving_residues:
        data.mpp.write_least_moving_residues(args.get_least_moving_residues, args.out)


if __name__ == "__main__":
    main()
