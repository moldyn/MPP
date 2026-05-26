#!/usr/bin/env python

import os
import sys
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


_VALID_D = frozenset({"T", "KL", "none", "gpcca"})
_VALID_G_MPP = frozenset({"JS", "none"})

_REQUIRED_CONFIG_KEYS = [
    "source",
    "microstate_trajectory",
    "multi_feature_trajectory",
    "lagtime",
    "pop_thr",
    "q_min",
    "frame_length",
]

OPTIONAL_PARAMS = [
    "cluster_file",
    "contact_index_file",
    "contact_threshold",
    "limits",
    "topology_file",
    "xtc_file",
    "helices",
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

        missing = [k for k in _REQUIRED_CONFIG_KEYS if k not in self.d]
        if missing:
            raise ValueError(
                "Config is missing required key(s): "
                + ", ".join(f"'{k}'" for k in missing)
                + ". See docs/usage_cli.md for the full list of required config keys."
            )

        self.source = self.d["source"]

        self.microstate_trajectory = np.loadtxt(
            os.path.join(self.source, self.d["microstate_trajectory"]), dtype=np.uint16
        )
        self.multi_feature_trajectory_raw = np.loadtxt(
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
        ct = self.d["contact_threshold"]
        if ct is not None:
            self.multi_feature_trajectory = self.multi_feature_trajectory_raw < ct
        else:
            self.multi_feature_trajectory = self.multi_feature_trajectory_raw
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

    def _prepare_kernels(self, dynamic_similarity, feature_similarity):
        if "stochastic" in self.d:
            kernel = MPP.kernel.LumpingKernel(
                method=self.d["stochastic"]["method"],
                param=self.d["stochastic"]["param"],
                similarity=dynamic_similarity,
                seed=self.d["stochastic"].get("seed", None),
            )
        else:
            kernel = MPP.kernel.LumpingKernel(
                similarity=dynamic_similarity,
            )

        if feature_similarity == "none":
            feature_kernel = None
        elif feature_similarity == "JS":
            feature_kernel = MPP.kernel.FeatureKernel(
                self.multi_feature_trajectory,
                self.microstate_trajectory,
            )
        else:
            raise ValueError("feature kernel must be 'none' or 'JS'.")

        if dynamic_similarity == "T" and feature_similarity == "none" and "stochastic" not in self.d:
            self.use_ref = False

        self.kernel = kernel
        self.feature_kernel = feature_kernel

    def setup_mpp(self, dynamic_similarity, feature_similarity):
        if dynamic_similarity != "gpcca":
            self._prepare_kernels(dynamic_similarity, feature_similarity)
        self.mpp = MPP.Lumping(
            self.microstate_trajectory,
            self.lagtime,
            self.multi_feature_trajectory_raw,
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

        Also saves ``macrostate_map.npy`` in the same directory as ``out``.
        This file maps each microstate index to its assigned macrostate index
        (integer array, shape ``(n_states,)``).

        Parameters
        ----------
        out : str
            Path to save the Z matrix (e.g. ``Z.npy``).
        overwrite : bool
            If True, recompute even if the file already exists. (default False)
        """
        if os.path.exists(out) and not overwrite:
            print(f"Loading existing Z matrix from {out}")
            self.mpp.load_Z(out)
        else:
            Path(os.path.dirname(out)).mkdir(parents=True, exist_ok=True)
            self.mpp.run_mpp(
                self.kernel,
                feature_kernel=self.feature_kernel,
                n=self.d["stochastic"]["n"] if "stochastic" in self.d else 1,
            )
            self.mpp.save_Z(out)
        macrostate_map_out = Path(out).parent / "macrostate_map.npy"
        if not macrostate_map_out.exists() or overwrite:
            np.save(macrostate_map_out, self.mpp.macrostate_map[0])

    def perform_gpcca(self, n_macrostates="reference_count", out=None, overwrite=False):
        """
        Run GPCCA lumping and optionally save the Z matrix.

        Parameters
        ----------
        n_macrostates : int or ``'reference_count'``
            Number of macrostates, or ``'reference_count'`` to use the count
            from the reference lumping (T). (default ``'reference_count'``)
        out : str, optional
            Path to save the Z matrix. If None, the result is not saved.
            (default None)
        overwrite : bool
            If True, recompute even if the file already exists. (default False)
        """
        if n_macrostates == "reference_count":
            n_macrostates = self.mpp.reference.n_macrostates[0]
        if out is not None and os.path.exists(out) and not overwrite:
            print(f"Loading existing Z matrix from {out}")
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


def _print_metrics(mpp):
    """Print quality metrics for all runs to stdout as key=value pairs.

    Parameters
    ----------
    mpp : MPP.Lumping
        Lumping object with macrostates already assigned.
    """
    def _fmt(arr):
        return ",".join(f"{v:.8g}" for v in arr)

    print(f"shannon_entropy={_fmt(mpp.shannon_entropy)}")
    print(f"davies_bouldin={_fmt(mpp.davies_bouldin_index)}")
    print(f"gmrq={_fmt(mpp.gmrq)}")
    print(f"gmrq2={_fmt(mpp.gmrq2)}")
    try:
        print(f"silhouette={_fmt(mpp.silhouette)}")
    except ValueError as exc:
        print(f"silhouette=N/A ({exc})")
    try:
        print(f"calinski_harabasz={_fmt(mpp.calinski_harabasz)}")
    except ValueError as exc:
        print(f"calinski_harabasz=N/A ({exc})")


def parse_args():
    parser = argparse.ArgumentParser(
        prog="python -m MPP.run",
        description=(
            "Run MPP (Most Probable Path) lumping on a Markov state model.\n"
            "Reads a YAML configuration file, runs or loads a lumping, "
            "and optionally generates plots or exports results."
        ),
        epilog=(
            "Examples:\n"
            "  # Run with transition-probability kernel and save Z matrix:\n"
            "  python -m MPP.run config.yml T none -Z results/t/Z.npy\n"
            "\n"
            "  # Load existing Z matrix and generate a dendrogram:\n"
            "  python -m MPP.run config.yml T none -Z results/t/Z.npy \\\n"
            "      -p dendrogram -o results/t/dendrogram.pdf\n"
            "\n"
            "  # Run with KL divergence kernel:\n"
            "  python -m MPP.run config.yml KL none -Z results/kl/Z.npy\n"
            "\n"
            "  # Run with combined transition probability + feature kernel:\n"
            "  python -m MPP.run config.yml T JS -Z results/t_js/Z.npy\n"
            "\n"
            "  # Save macrostate trajectory to text file:\n"
            "  python -m MPP.run config.yml T none -Z results/t/Z.npy \\\n"
            "      -p macrostate_trajectory -o results/t/macrostate_trajectory.txt\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "data_specification",
        metavar="config.yml",
        help=(
            "YAML configuration file specifying input paths and lumping "
            "parameters (source, microstate_trajectory, "
            "multi_feature_trajectory, lagtime, pop_thr, q_min, "
            "frame_length, and optional keys)."
        ),
        type=argparse.FileType("r", encoding="latin-1"),
    )
    parser.add_argument(
        "d",
        metavar="d",
        help=(
            "Dynamic similarity selector (lumping kernel). "
            "One of: 'T' (transition probability, recommended default), "
            "'KL' (Kullback-Leibler divergence of transition rows), "
            "'none' (feature-only mode, requires g=JS), or "
            "'gpcca' (GPCCA comparison run)."
        ),
    )
    parser.add_argument(
        "g",
        metavar="g",
        help=(
            "Feature similarity selector. "
            "One of: 'JS' (Jensen-Shannon divergence of feature distributions) "
            "or 'none' (no feature similarity). "
            "When d='gpcca': an integer number of macrostates, or "
            "'reference_count' to reuse the macrostate count from the "
            "reference T lumping."
        ),
    )
    parser.add_argument(
        "-o",
        "--out",
        metavar="PATH",
        help=(
            "Output file path for the plot or exported file "
            "(required when -p or -r is used)."
        ),
    )
    parser.add_argument(
        "-Z",
        metavar="PATH",
        help=(
            "Path to the Z matrix file (.npy). "
            "If the file does not exist, MPP is run and the result is saved here. "
            "If the file already exists, it is loaded instead of recomputed. "
            "Also writes macrostate_map.npy to the same directory."
        ),
    )
    parser.add_argument(
        "--rmsd",
        metavar="PATH",
        help=(
            "Compute per-macrostate C-alpha RMSD and write the result "
            "to this .npy file."
        ),
    )
    parser.add_argument(
        "--rmsd-feature",
        metavar="CA|feature",
        help=(
            "RMSD variant: 'CA' for C-alpha RMSD (default) or "
            "'feature' for feature-based RMSD."
        ),
        default="CA",
    )
    parser.add_argument(
        "-r",
        "--draw-random",
        metavar="N",
        help=(
            "Write N random frame indices per macrostate as .ndx files "
            "to the directory given by -o."
        ),
        type=int,
    )
    parser.add_argument(
        "-p",
        "--plot",
        metavar="PLOT",
        help=(
            "Plot type to generate and save to the path given by -o. "
            "One of: dendrogram, timescales, sankey, contacts, macrotraj, "
            "ck_test, rmsd, delta_rmsd, state_network, macro_feature, "
            "stochastic_state_similarity, relative_implied_timescales, "
            "transition_matrix, transition_time, macrostate_trajectory. "
            "The 'macrostate_trajectory' type writes a text file of "
            "macrostate assignments (one integer per line)."
        ),
    )
    parser.add_argument(
        "--scale",
        metavar="FLOAT",
        help="Scaling factor for plot size (default: 1).",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--n-timescales",
        metavar="N",
        help=(
            "Number of implied timescales to compute "
            "(overrides the n_timescales value in the config file)."
        ),
        type=int,
        dest="n_timescales",
    )
    parser.add_argument(
        "--get-least-moving-residues",
        metavar="CONTACT_INDEX_FILE",
        help=(
            "Write the least-varying residues per macrostate to the file "
            "given by -o, using CONTACT_INDEX_FILE as the contact index."
        ),
    )
    parser.add_argument(
        "--metrics",
        action="store_true",
        help=(
            "Print all available quality metrics to stdout as key=value pairs. "
            "Metrics reported: shannon_entropy, davies_bouldin, gmrq, gmrq2, "
            "silhouette, calinski_harabasz (one value per run, comma-separated "
            "for stochastic lumpings)."
        ),
    )
    return parser.parse_args()


def _arg_error(message):
    """Print a user-facing error message and exit with code 2."""
    print(f"error: {message}", file=sys.stderr)
    print("Run 'python -m MPP.run --help' for usage information.", file=sys.stderr)
    sys.exit(2)


def _validate_args(args):
    """Validate parsed CLI arguments and exit with a clear message on error."""
    if args.d not in _VALID_D:
        _arg_error(
            f"invalid dynamic similarity selector '{args.d}'. "
            f"Valid choices: {', '.join(sorted(_VALID_D))}."
        )
    if args.d != "gpcca" and args.g not in _VALID_G_MPP:
        _arg_error(
            f"invalid feature similarity selector '{args.g}'. "
            f"Valid choices: {', '.join(sorted(_VALID_G_MPP))}."
        )
    if args.d != "gpcca" and args.Z is None:
        _arg_error(
            "'-Z <path>' is required to save or load the Z matrix. "
            "Example: -Z results/t/Z.npy"
        )
    if args.plot and args.out is None:
        _arg_error(
            f"'-o <path>' is required when '-p {args.plot}' is specified."
        )


def main():
    args = parse_args()
    _validate_args(args)

    # Parse input files
    data = Data(args.data_specification.name)
    data.setup_mpp(args.d, args.g)
    if args.d == "gpcca":
        n_macrostates = args.g
        if n_macrostates != "reference_count":
            try:
                n_macrostates = int(n_macrostates)
            except ValueError:
                pass
        data.perform_gpcca(n_macrostates, args.Z)
    else:
        data.perform_mpp(args.Z)

    if args.n_timescales is not None:
        data.d["n_timescales"] = args.n_timescales

    if args.rmsd:
        data.mpp.rmsd_feature = args.rmsd_feature
        data.mpp.rmsd_estimator = MPP.utils.argmedian
        data.get_rmsd(args.rmsd, overwrite=False)

    if args.plot:
        plot(data, args.out, kind=args.plot, scale=args.scale)

    if args.draw_random:
        write_random_frames_indices(data.mpp, args.out, args.draw_random)

    if args.get_least_moving_residues:
        data.mpp.write_least_moving_residues(args.get_least_moving_residues, args.out)

    if args.metrics:
        _print_metrics(data.mpp)


if __name__ == "__main__":
    main()
