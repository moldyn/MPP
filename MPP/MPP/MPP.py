import os
import datetime
import warnings
import numpy as np
import numpy.typing as npt
import msmhelper as mh
import mdtraj as md

from typing import Literal
from pathlib import Path
from tqdm import tqdm
from collections.abc import Iterable
from sklearn.metrics import davies_bouldin_score
import pygpcca as gp

from . import core
from . import utils
from . import kernel as kernel_module
from . import plot
from .graph import draw_knetwork


class Lumping(object):
    """A class to hold all the information and methods for MPP.

    The most probable path (MPP) algorithm reduces the number of states
    in a microstate trajectory of a Markovian process in order to
    simplify its analysis. This two-step algorithm first builds a
    lumping tree by successively merging the least metastable state
    together with its most similar other state. In a second step, the
    lumping tree is parsed in reverse order (from the root) in order to
    partition the microstates into macrostates, which satisfy a minimum
    metastability and a minimum population criterion. For details, see
    the reference publication (see below).

    Attributes
    ----------
    trajectory : ndarray of int, shape (N,)
        The microstate trajectory. N is the number of frames.
    lagtime : int
        Lagtime used, in frames.
    frame_length : float
        Length of a frame in ns. (default 0.2)
    pop_thr : float
        Population threshold for macrostates. (default 0.005)
    q_min : float
        Minimum metastability of macrostates. (default 0.5)
    limits : list of int
        If the trajectory is composed of several independent
        simulations, this list contains the lengths of the individual
        trajectories. (default None)
    tmat : ndarray of float, shape (n_states, n_states)
        Transition matrix of the microstate Markov state model.
    pop : ndarray of int, shape (N,)
        Population of the microstates in the trajectory. N is the
        number of microstates.
    n_states : int
        Number of microstates.
    multi_feature_trajectory : ndarray of float, shape (N, M)
        A feature trajectory of M features and N frames. For example
        selected contact distances are passed here. Must be the same as
        for trajectory.
    mean_feature_trajectory : ndarray of float, shape (N,)
        The mean feature trajectory of N frames.
    mean_feature : ndarray of float, shape (N,)
        The mean feature for each of N microstates.
    Z : ndarray of float, shape (N-1, 4)
        The Z matrix defines how the microstates are lumped together.
    reference : Lumping
        Instance of the Lumping object with reference parameters, that
        are, the only the transition probability is utilized as
        similarity metric.
    macrostate_assignment : list of (ndarray of bool,
            shape (Lumping.n_macrostates, Lumping.n_states))
        Macrostate assignment of the lumpings. Elements, whose indices
        correspond to an assignment are True, all other elements are
        False.
    macrostate_map : list of (ndarray of int, shape (N,))
        Map arrays of the assignment. The index of the array
        corresponds to the microstate and the value is the index of the
        assigned macrostate.
    macrostate_tmat : list of (ndarray of float, shape (N, N))
        Transition matrices of the lumpings.
    macrostate_trajectory : ndarray of int, shape (N, M)
        Macrostate trajectories. N is the number of runs and M the
        length of the trajectories.
    macrostate_population : ndarray of int, shape (N, M)
        Macrostate populations in frames. N is the number of runs and
        M is the number of macrostates.
    n_macrostates : list of int
        Number of macrostates in each lumping.

    Citation
    --------
    tbd
    """

    lagtime: int
    """The lagtime used, in frames."""

    contact_threshold: float
    """Distance below which a feature (e.g. a contact) is considered
    formed. If None, directly use the feature.
    """

    pop_thr: float
    """Minimum population allowed for macrostates."""

    q_min: float
    """Minimum metastability allowed for macrostates."""

    limits: list[int]
    """When concatenated trajectories are used, the lengths of the
    individual trajectories.
    """

    n_states: int
    """The number of microstates."""

    quiet: bool
    """Suppress progress reports."""

    Z: npt.NDArray[float]
    """The Z matrix defines how the microstates are lumped together.

    A 2D array of float, shape (Lumping.n_states-1, 4). The Z matrix
    defines a lumping and is organized like the Z matrix returned from
    scipy.cluster.hierarchy.linkage. Each row describes a merging of
    two states in the first step of the MPP algorithm. The first two
    columns contain the state indices, which are merged. The third
    column holds the metastability of the less stable state (the state
    in the first column) and the last column provides the population of
    the new state, which has index Lumping.n_states + i, where i is the
    row index.
    """

    xtc_stride: float
    """Use only every Lumping.xtc_stride frame of the xtc trajectory."""

    plot: "Plotter"
    """Produce various plots of the lumping."""

    macrostate_feature: list[list[float]]
    """Mean macrostate feature."""

    macrostate_multi_feature: list[list[npt.NDArray[np.floating]]]
    """Macrostate features."""

    macrostate_assignment: list[npt.NDArray[bool]]
    """Macrostate assignments."""

    macrostate_map: list[npt.NDArray]
    # macrostate_tmat = []
    # macrostate_trajectory = np.zeros(
    # n_macrostates = []

    def __init__(
        self,
        trajectory: npt.NDArray[np.int_],
        lagtime: int,
        feature_trajectory: npt.NDArray[np.floating] | None = None,
        contact_threshold: float = 0.45,
        pop_thr: float = 0.005,
        q_min: float = 0.5,
        frame_length: float = 0.2,
        limits: list[int] | None = None,
        quiet: bool = False,
    ) -> None:
        """Initialize a Lumping object.

        Parameters
        ----------
        trajectory : ndarray of int, shape (N,)
            The microstate trajectory. N is the number of frames.
        lagtime : int
            Lag time used.
        feature_trajectory : ndarray of float, shape (N, M)
            A feature trajectory of M features and N frames. For
            example selected contact distances are passed here. Must
            be the same as for trajectory. If None, a moch array of
            ones is created. (default None)
        contact_threshold : float
            Distance in feature space below which the interaction is
            considered positive, e.g. a contact distance below which a
            contact is considered formed. If None, directly use the
            feature (default 0.45)
        pop_thr : float
            Population threshold for macrostates. (default 0.005)
        q_min : float
            Minimum metastability of macrostates. (default 0.5)
        frame_length : float
            Length of a frame in ns. (default 0.2)
        limits : list of int
            If the trajectory is composed of several independent
            simulations, this list contains the lengths of the
            individual trajectories. (default None)
        quiet : bool
            Whether to hide progress reports or not. (default False)
        """
        self.trajectory = trajectory
        self.frame_length = frame_length
        self.lagtime = lagtime
        self.pop_thr = pop_thr
        self.q_min = q_min
        self.limits = limits
        self.tmat, states = mh.msm.estimate_markov_model(
            utils.get_multi_state_trajectory(self.trajectory, self.limits),
            self.lagtime,
        )
        _, self.pop = np.unique(self.trajectory, return_counts=True)
        self.n_states = len(states)
        self.quiet = quiet
        if feature_trajectory is not None:
            self.contact_threshold = contact_threshold
            self._add_feature(feature_trajectory)
        else:
            self._add_feature(np.ones((trajectory.shape, 1)))

        self.Z = None
        self._timescales = None
        self._linkage = None
        self._macrostate_population = None
        self._tree = None
        self._shannon_entropy = None
        self._davies_bouldin_index = None
        self._gmrq = None
        self._gmrq2 = None
        self._reference = None
        self._topology_file = None
        self._xtc_trajectory_file = None
        self._rmsd = None
        self._n_i = None
        self._macro_micro_feature = None
        # Whether to use CA cartesian coordinates for RMSD calculation
        # of the feature ("feature")
        self.rmsd_feature = "CA"
        self.rmsd_estimator = np.argmin
        self._mean_frames_idx = None
        self.mean_frames = None
        self.xtc_stride = 1
        self.plot = Plotter(self)

    def _add_feature(
        self,
        feature_trajectory: npt.NDArray[np.floating],
    ):
        """Add feature data to the instance.

        feature_trajectory : ndarray of float, shape (N, M)
            A feature trajectory of M features and N frames. For
            example selected contact distances are passed here. Must
            be the same as for trajectory.
        """
        if feature_trajectory.shape[0] != self.trajectory.shape[0]:
            raise ValueError(
                (
                    "feature_trajectory must have the same length as the microstate "
                    "trajectory (mpp.trajectory)"
                )
            )
        if feature_trajectory.ndim == 2:
            self.multi_feature_trajectory = feature_trajectory
        else:
            raise ValueError("feature_trajectory must be 2 D")

        if self.contact_threshold is None:
            self.multi_feature_trajectory_bool = self.multi_feature_trajectory
        else:
            self.multi_feature_trajectory_bool = (
                self.multi_feature_trajectory < self.contact_threshold
            )
        self.mean_feature_trajectory = self.multi_feature_trajectory_bool.mean(axis=1)
        self.mean_feature = np.zeros(self.n_states)
        for i in range(self.n_states):
            self.mean_feature[i] = self.mean_feature_trajectory[
                self.trajectory == i
            ].mean()

    def run_mpp(
        self,
        kernel: kernel_module.LumpingKernel = kernel_module.LumpingKernel(),
        feature_kernel: kernel_module.FeatureKernel | None = None,
        n: int = 1,
    ) -> None:
        """Run the MPP algorithm.

        Parameters
        ----------
        kernel : LumpingKernel
            An instance of the LumpingKernel object, which determines
            the similarity metric employed.
            (default kernel_module.LumpingKernel())
        feature_kernel : FeatureKernel
            An instance of the FeatureKernel object, which determines
            the similarity metric for the feature. None to disable
            feature incorporation. (default None)
        n : int
            Count of runs for a stochastic lumping kernel. (default 1)
        """
        self.n_runs = n
        self.kernel = kernel
        self.feature_kernel = feature_kernel

        self.Z = np.zeros((self.n_runs, self.n_states - 1, 4), dtype=np.float64)
        self.full_pop = np.zeros((self.n_runs, 2 * self.n_states - 1), dtype=np.uint32)
        if self.quiet:
            iter = range(self.n_runs)
        else:
            print("Clustering ...")
            iter = tqdm(range(self.n_runs))
        for i in iter:
            self.Z[i], self.full_pop[i] = core.cluster(
                self.tmat,
                self.pop,
                kernel=self.kernel,
                feature_kernel=self.feature_kernel,
            )
        self.assign_macrostates()

    def assign_macrostates(self) -> None:
        """Assign macrostates and provide macrostate data."""
        self.macrostate_feature = []
        self.macrostate_multi_feature = []
        self.macrostate_assignment = []
        self.macrostate_map = []
        self.macrostate_tmat = []
        self.macrostate_trajectory = np.zeros(
            (self.n_runs, self.trajectory.shape[0]), dtype=self.trajectory.dtype.type
        )
        self.n_macrostates = []

        if self.quiet:
            iter = range(self.n_runs)
        else:
            print("Assigning macrostates ...")
            iter = tqdm(range(self.n_runs))
        for n_i in iter:
            self.macrostate_assignment.append(
                utils.get_macrostate_assignment_from_tree(self.tree[n_i])
            )

            # Calculate other macrostate related values
            self.macrostate_map.append(
                np.zeros(self.n_states, dtype=self.trajectory.dtype.type)
            )
            mas, mis = np.where(self.macrostate_assignment[-1] == 1)
            self.macrostate_map[-1][mis] = mas
            self.macrostate_tmat.append(
                utils.macrostate_tmat(
                    self.tmat, self.macrostate_assignment[-1], self.pop
                )
            )
            self.macrostate_trajectory[n_i] = utils.translate_trajectory(
                self.trajectory, self.macrostate_map[-1]
            )
            self.n_macrostates.append(self.macrostate_assignment[-1].shape[0])
            self.macrostate_feature.append(
                [
                    self.mean_feature_trajectory[
                        np.where(self.macrostate_trajectory[n_i] == i)
                    ].mean()
                    for i in np.arange(self.n_macrostates[-1])
                ]
            )
            self.macrostate_multi_feature.append(
                [
                    self.multi_feature_trajectory_bool[
                        np.where(self.macrostate_trajectory[n_i] == i)
                    ].mean(axis=0)
                    for i in np.arange(self.n_macrostates[-1], dtype=int)
                ]
            )

    def gpcca(self, n_macrostates: int | None = None) -> None:
        """Instead of MPP algorithm use GPCCA algorithm for lumping.

        Generalized Perron Cluster Cluster Analysis (GPCCA) is used
        instead of the MPP algorithm. GPCCA requires the definition of
        a number of macrostates to cluster the microstates into. For
        comparison, the number of macrostates in the reference lumping
        may be a good choice. This method is implemented for reference.

        Parameters
        ----------
        n_macrostates : int
            Number of macrostates. If None, employ the same number of
            macrostates as yielded from the reference lumping.
            (default None)
        """
        # Create the reference anyways as pop_thr and q_min are changed
        if n_macrostates is None:
            n_macrostates = self.reference.n_macrostates[0]
        else:
            self.reference

        # Change pop_thr and q_min in order to make sure that tiny macrostates
        # can be created as well.
        self.pop_thr = 0
        self.q_min = 0.5

        self.gpcca = gp.GPCCA(self.tmat, method="krylov")
        self.gpcca.optimize(n_macrostates)

        self.n_runs = 1
        self.n_macrostates = [n_macrostates]

        self._assign_macrostates_from_gpcca(self.gpcca.macrostate_assignment)
        self._create_mock_Z()

    def _assign_macrostates_from_gpcca(self, gma):
        gmt = np.zeros(self.trajectory.shape, dtype=self.trajectory.dtype)
        gmf = np.empty(self.n_macrostates[0])
        for i in range(self.n_macrostates[0]):
            gmt[np.where(np.isin(self.trajectory, np.where(gma == i)[0]))[0]] = i + 1
            gmf[i] = self.mean_feature_trajectory[gmt == i + 1].mean()

        order = np.argsort(gmf)[::-1]
        new_states = np.empty(self.n_macrostates[0], dtype=self.trajectory.dtype.type)
        new_states[order] = np.arange(
            self.n_macrostates[0], dtype=self.trajectory.dtype.type
        )
        self.macrostate_map = [np.empty(gma.shape, dtype=self.trajectory.dtype.type)]
        for i in range(self.n_macrostates[0]):
            self.macrostate_map[0][np.where(gma == i)] = new_states[i]

        self.macrostate_assignment = [
            np.full((self.n_macrostates[0], self.macrostate_map[0].shape[0]), False)
        ]
        self.macrostate_assignment[0][
            self.macrostate_map[0],
            np.arange(self.macrostate_map[0].shape[0], dtype=int),
        ] = True
        self.macrostate_feature = [gmf[order]]
        self.macrostate_trajectory = np.empty(
            (self.n_runs, self.trajectory.shape[0]), dtype=self.trajectory.dtype.type
        )
        self.macrostate_trajectory[0] = utils.translate_trajectory(
            self.trajectory, self.macrostate_map[0]
        )
        self.macrostate_tmat = [
            utils.macrostate_tmat(self.tmat, self.macrostate_assignment[0], self.pop)
        ]
        self.macrostate_multi_feature = [
            [
                self.multi_feature_trajectory_bool[
                    np.where(self.macrostate_trajectory[0] == i)
                ].mean(axis=0)
                for i in np.arange(self.n_macrostates[-1], dtype=int)
            ]
        ]

    def _create_mock_Z(self):
        # Create mock Z and mock full_pop for Sankey plot
        # After implementation remove mock Z.npy file in run.py
        self.Z = np.zeros((self.n_runs, self.n_states - 1, 4), dtype=np.float64)
        self.full_pop = np.zeros((self.n_runs, 2 * self.n_states - 1), dtype=np.uint32)
        self.full_pop[0, : self.n_states] = self.pop

        last_merged = self.n_states
        merge = 0
        for macrostate in range(self.n_macrostates[0]):
            microstates = np.where(self.macrostate_map[0] == macrostate)[0]
            origin = microstates[0]
            if microstates.shape[0] > 1:
                for target in microstates[1:]:
                    intermediate_state = self.n_states + merge
                    self.full_pop[0, intermediate_state] = self.full_pop[
                        0, [origin, target]
                    ].sum()
                    self.Z[0, merge] = (
                        origin,
                        target,
                        0.2,
                        self.full_pop[0, intermediate_state],
                    )
                    origin = intermediate_state
                    merge += 1

            if macrostate > 0:
                intermediate_state = self.n_states + merge
                target = last_merged
                self.full_pop[0, intermediate_state] = self.full_pop[
                    0, [origin, target]
                ].sum()
                self.Z[0, merge] = (
                    origin,
                    target,
                    0.9,
                    self.full_pop[0, intermediate_state],
                )
                last_merged = intermediate_state
                merge += 1
            else:
                last_merged = origin

    def save_macrostate_trajectory(self, out: Path, one_based: bool = False) -> None:
        """Write macrostate trajectory to a text file."""
        header = (
            f"Created by Lumping class\n"
            f"Time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
            f"Trajectory contains {self.n_macrostates[self.n_i]} states and {self.macrostate_trajectory.shape[1]} frames.\n"
            f"Trajectory index: {self.n_i}\n"
        )
        macrostate_trajectory = self.macrostate_trajectory[self.n_i]
        if one_based:
            macrostate_trajectory += 1
        np.savetxt(out, macrostate_trajectory, fmt="%.0f", header=header)

    def save_Z(self, out: Path, n_i: int | Iterable | Literal["all"] = "all") -> None:
        """Save Z matrix.

        Parameters
        ----------
        out : Path
            Where to save the Z matrix.
        n_i : int or iterable of int or {"all"}
            Which lumpings to save.
        """
        if not out.endswith(".npy"):
            out += ".npy"

        if n_i == "all":
            np.save(out, self.Z)
        elif isinstance(n_i, Iterable):
            np.save(out, self.Z[n_i])
        elif isinstance(n_i, int):
            np.save(out, self.Z[n_i : n_i + 1])
        else:
            raise ValueError("n_i must be 'all', Iterable or int.")

    def load_Z(self, Z, gpcca=False):
        """Load Z matrix."""
        if isinstance(Z, np.ndarray):
            self.Z = Z
        elif os.path.exists(Z):
            self.Z = np.load(Z)
        else:
            raise ValueError("Z must be a numpy array or a .npy file.")

        self.n_runs = self.Z.shape[0]
        # n: number of macrostates
        tmat, states = mh.msm.estimate_markov_model(
            utils.get_multi_state_trajectory(self.trajectory, self.limits),
            self.lagtime,
        )
        self.tmat = tmat.astype(float)
        _, self.pop = np.unique(self.trajectory, return_counts=True)
        self.n_states = len(states)
        self.full_pop = np.zeros((self.n_runs, 2 * self.n_states - 1), dtype=np.uint32)
        self.full_pop[:, : self.n_states] = self.pop
        self.full_pop[:, self.n_states :] = self.Z[:, :, 3]

        if gpcca:
            # First create the reference, then change pop_thr and q_min
            self.reference

            # Change pop_thr and q_min in order to make sure that tiny macrostates
            # can be created as well.
            self.pop_thr = 0
            self.q_min = 0.5

            self.n_macrostates = [((self.Z[0, :, 2] > self.q_min).sum() + 1)]
            gma = utils.get_macrostate_assignment_from_tree(self.tree[0])
            self._assign_macrostates_from_gpcca(
                np.array([np.where(i)[0][0] for i in gma.T])
            )
        else:
            self.assign_macrostates()

    def save_rmsd(self, out):
        """Save RMSD of states to numpy file."""
        np.save(out, self.rmsd)
        fname, ext = os.path.splitext(out)
        self.save_mean_frames_idx(fname + "_mean_frames.ndx")

    def load_rmsd(self, f_name):
        """Save RMSD of states from numpy file."""
        self._rmsd = np.load(f_name)
        fname, ext = os.path.splitext(f_name)
        self.load_mean_frames_idx(fname + "_mean_frames.ndx")

    def draw_random_frames_indices(
        self, out: Path | None = None, n: int = 20
    ) -> npt.NDArray[np.int_] | None:
        """Draw n random frame indices for each macrostate.

        Parameters
        ----------
        out : Path, optional
            Directory where to save the .ndx files. If None, return
            indices instead of being saved. (default None)
        n : int
            Number of random frame indices to draw. (default 20)

        Returns
        -------
        ndarray of int, shape (N,) or None
            N random frame indices if out is None, otherwise retutrns
            None.
        """
        drawn_frames = np.empty((self.n_macrostates[self.n_i], n), dtype=int)
        for state in np.arange(self.n_macrostates[self.n_i]):
            frames_in_state = np.where(self.macrostate_trajectory[self.n_i] == state)[0]
            drawn_frames[state] = np.random.choice(
                frames_in_state, size=n, replace=False
            )
        if self.xtc_stride is not None:
            drawn_frames *= self.xtc_stride
        if out is not None:
            # Path(os.path.join(out)).mkdir(parents=True, exist_ok=True)
            out.mkdir(parents=True, exist_ok=True)
            for s, i in enumerate(drawn_frames):
                np.savetxt(
                    os.path.join(out, f"{s + 1:02d}.ndx"),
                    i,
                    fmt="%.0f",
                    header="[frames]",
                )
        else:
            return drawn_frames

    def draw_random_frames(self, out: Path, n: int = 20) -> None:
        """Draw n random frames per macrostate and write pdb files.

        Parameters
        ----------
        out : Path
            Directory where to save the .pdb files.
        n : int
            Number of random frame indices to draw. (default 20)
        """
        for state in np.arange(self.n_macrostates[self.n_i]):
            frames_in_state = np.where(self.macrostate_trajectory[self.n_i] == state)[0]
            drawn_frames = np.random.choice(frames_in_state, size=n, replace=False)
            for i, frame in enumerate(drawn_frames):
                f = md.load_xtc(
                    self.xtc_trajectory_file,
                    top=self.topology_file,
                    frame=frame,
                )
                f.save_pdb(os.path.join(out, f"S{state}_{i:02d}.pdb"))

    def get_best_defined_contacts(self, n: int = 3) -> npt.NDArray[np.int_]:
        """Return the feature indices with the least variance.

        Calculate the variance for each feature for each macrostete for
        lumping n_i and return the indices for features with the least
        variance.

        Parameters
        ----------
        n : int
            Number of features to return. (default 3)

        Returns
        -------
        ndarray of int, shape (N, M)
            Indices of least varying features. N is the number of
            macrostates and M is the number of features required
            (parameter n).
        """
        contacts = np.zeros((self.n_macrostates[self.n_i], n), dtype=int)
        for i in range(self.n_macrostates[self.n_i]):
            contacts[i] = np.argsort(
                np.var(
                    self.multi_feature_trajectory[
                        self.macrostate_trajectory[self.n_i] == i
                    ],
                    axis=0,
                )
            )[:n]
        return contacts

    def get_least_moving_residues(
        self, contact_index_file: Path, n: int = 3
    ) -> list[npt.NDArray[np.int_]]:
        """Return residue indices of least varying feature.

        Parameters
        ----------
        contact_index_file : Path
            Path to contact index file. Each line contains the two
            residue indices space separated, which are part of the
            feature. The line index corresponds to the feature index.
        n : int
            Number of features to consider. (default 3)

        Returns
        -------
        list of ndarray of int
            One list entry per macrostate. The array contains the
            indices of the features with least variance for the
            respective macrostate.
        """
        contact_indices = np.loadtxt(contact_index_file, dtype=int)
        contacts = self.get_best_defined_contacts(n)
        least_moving_residues = []
        for c in contacts:
            least_moving_residues.append(np.unique(contact_indices[c].flatten()))
        return least_moving_residues

    def write_least_moving_residues(
        self, contact_index_file: Path, out: Path, n: int = 3
    ) -> None:
        """Write indices of least moving residues to an index file.

        Parameters
        ----------
        contact_index_file : Path
            Path to contact index file. Each line contains the two
            residue indices space separated, which are part of the
            feature. The line index corresponds to the feature index.
        out : Path
            Target file for the residue indices. Each line holds the
            residue indices for one macrostate in a space separated
            list
        n : int
            Number of features to consider. (default 3)
        """
        if contact_index_file != "none":
            least_moving_residues = self.get_least_moving_residues(
                contact_index_file, n=n
            )
            with open(out, "w") as f:
                for residues in least_moving_residues:
                    f.write(f"{' '.join(residues.astype(str))}\n")
        else:
            with open(out, "w") as f:
                f.write("")

    def __add__(
        self, other: "Lumping"
    ) -> tuple["Lumping", "Lumping", npt.NDArray[np.floating]]:
        """Return the state overlap between two lumpings.

        This is mostly utilized to analyze the similarity of stochastic
        lumpings to a deterministic one.

        Parameters
        ----------
        other : Lumping
            Another instance of Lumping with which to compare this
            Lumping instance.

        Returns
        -------
        tuple of (
                Lumping,
                Lumping,
                (ndarray of float, shape (3, N, M)),
        )
            The first Lumping object is the deterministic lumping to
            which the stochastic lumping (second Lumping instance) is
            compared. The array contains the results: First comes the
            union, second the fraction in reference macrostates and
            third the fraction in stochastic lumpings. N is the number
            of macrostates of the reference lumping and M the number of
            stochastic lumpings. The reference is not forcibly
            Lumping.reference, but can be any deterministic Lumping
            instance.
        """
        if self.n_runs == 1 and other.n_runs >= 1:
            # reference
            ref = self
            # stochastic lumping
            sto = other
        elif other.n_runs == 1 and self.n_runs >= 1:
            ref = other
            sto = self
        else:
            raise ValueError("The reference lumping must have exactly one run.")
        return ref, sto, utils.similarity(ref, sto)

    @property
    def reference(self) -> "Lumping":
        """The reference lupming of the system.

        Only the transition probability is utilized to estimate the
        state similarity in the lumping process.

        Returns
        -------
        Lumping
            An instance of Lumping with reference parameters.
        """
        if self._reference is None:
            k = kernel_module.LumpingKernel()
            self._reference = Lumping(
                self.trajectory,
                self.lagtime,
                self.multi_feature_trajectory,
                contact_threshold=self.contact_threshold,
                pop_thr=self.pop_thr,
                q_min=self.q_min,
                limits=self.limits,
                quiet=True,
            )
            self._reference.run_mpp(k)
        return self._reference

    @property
    def n_i(self) -> int:
        """Index of the lumping under consideration.

        0 for deterministic lumpings. Is set to the lumping with the
        longest first implied timescale, if not set manually.

        Returns
        -------
        int
            Index of the lumping under consideration.
        """
        if self._n_i is None:
            if self.n_runs > 1:
                self._n_i = np.argmax(self.timescales[:, 0])
            else:
                self._n_i = 0
        return self._n_i

    @n_i.setter
    def n_i(self, value: int) -> None:
        if not isinstance(value, int):
            raise ValueError("n_i must be an integer")
        self._n_i = value

    @property
    def timescales(self) -> npt.NDArray[np.floating]:
        """Implied timescales property, in frames.

        If more than the first 3 implied timescales are required, run
        the calc_timescales method separately.

        Returns
        -------
        ndarray of float, shape (N, M)
            First M implied timescales for all N runs.
        """
        if self._timescales is None:
            self.calc_timescales()
        return self._timescales

    def calc_timescales(self, ntimescales: int = 3) -> None:
        """Calculate implied timescales in frames.

        Parameters
        ----------
        ntimescales : int
            Number of first implied timescales to calculate.
        """
        self._timescales = np.zeros((self.n_runs, ntimescales))
        for i, trajectory in enumerate(self.macrostate_trajectory):
            self._timescales[i, :] = mh.msm.implied_timescales(
                utils.get_multi_state_trajectory(trajectory, self.limits),
                [self.lagtime],
                ntimescales=ntimescales,
            )[0]

    @property
    def linkage(self) -> npt.NDArray[np.floating]:
        """The linkage matrix.

        The linkage matrix is derived from the Z matrix and utilized in
        former implementations of the MPP algorithm.

        Returns
        -------
        ndarray of float, shape (N-1, 3)
            For a system of N microstates, each line defines a merging.
            The state in the first column is merged with the state in
            the second column and the new state has the index of the
            second state, which is the more stable one. The
            metastability of the first state is stored in the third
            column.
        """
        if self._linkage is None:
            self._linkage = utils.Z_to_linkage(self.Z[self.n_i])
        return self._linkage

    @property
    def macrostate_population(self) -> list[npt.NDArray[np.int_]]:
        """Return the macrostate populations for all runs."""
        if self._macrostate_population is None:
            self._macrostate_population = []
            for j, ma in enumerate(self.macrostate_assignment):
                self._macrostate_population.append(
                    np.zeros(ma.shape[0], dtype=self.full_pop.dtype.type)
                )
                for i, m in enumerate(ma):
                    self._macrostate_population[-1][i] = self.full_pop[
                        j, : self.n_states
                    ][m.astype(bool)].sum()
        return self._macrostate_population

    @property
    def rmsd(self) -> npt.NDArray[np.floating]:
        """Return the root mean square deviation (RMSD) of C-alphas.

        The RMSD of C-alphas is calculate so that it is minimal. The
        mean_frames attribute holds the mdtraj objects of the frames,
        which minimizes the RMSD, for each macrostate.

        Returns
        -------
        ndarray of float, shape (N, M)
            C-alpha RMSD values for N macrostates and M C-alpha atoms.
        """
        if self._rmsd is None:
            if self.rmsd_feature == "CA":
                self._rmsd, self._mean_frames_idx = utils.calc_rmsd(
                    self, estimator=self.rmsd_estimator, quiet=self.quiet
                )
            elif self.rmsd_feature == "feature":
                self._rmsd, self._mean_frames_idx = utils.calc_rmsd_feature(
                    self, estimator=self.rmsd_estimator, quiet=self.quiet
                )
        return self._rmsd

    def save_mean_frames_idx(self, out):
        """Save mean frames index to out."""
        np.savetxt(out, self._mean_frames_idx, fmt="%.0f", header="[frames]")

    def load_mean_frames_idx(self, fname):
        """Load mean frames index from fname."""
        self._mean_frames_idx = np.loadtxt(fname, dtype=int)

    # Don't use this method, rather use gmx instead.
    def load_mean_frames(self):
        """Load mean frames"""
        if self._mean_frames_idx is None:
            print("You first need to load the mean frame indices.")
        else:
            self.mean_frames = utils.load_trajectory(
                self.topology_file,
                self.xtc_trajectory_file,
                frames=self._mean_frames_idx,
                stride=self.xtc_stride,
            )

    def save_mean_frames(self, out):
        if self.mean_frames is None:
            self.load_mean_frames()
        if self._mean_frames_idx is not None:
            self.mean_frames.save(out)

    def rmsd_sharpness(self) -> float:
        """Returns the RMSD sharpness of a lumping.

        The RMSD sharpness is given by the population weighted mean of
        mean RMSDs:
        s = sum_j(mean_i(rmsd_ij) * p_j) / sum_j(p_j)
        for C-alpha i, macrostate j and population p.

        Returns
        -------
        float
            RMSD sharpness value

        See Also
        --------
        Lumping.rmsd : RMSD property.
        """
        return (
            self.rmsd.mean(axis=1) * self.macrostate_population[self.n_i]
        ).sum() / self.macrostate_population[self.n_i].sum()

    @property
    def shannon_entropy(self) -> npt.NDArray[np.floating]:
        """The Shannon entropy.

        The Shannon entropy is employed to measure how evenly the
        macrostate are populated in a lumping. The Shannon entropy
        penalizes in this case very low puplated macrostates. The lower
        the entropy, the more even the population partition of the
        macrostates.

        Returns
        -------
        ndarray of float, shape (Lumping.n_i,)
            Array contains the Shannon entropy for each run.

        References
        ----------
        [1] Claude C. Shannon, "A Mathematical Theory of Communication",
        The Bell System Technical Journal, Volume 27, Issue 3, 1948,
        DOI: 10.1002/j.1538-7305.1948.tb01338.x
        """
        if self._shannon_entropy is None:
            self._shannon_entropy = np.zeros(self.n_runs)
            for i, pop in enumerate(self.macrostate_population):
                self._shannon_entropy[i] = utils.shannon_entropy(pop)
        return self._shannon_entropy

    @property
    def davies_bouldin_index(self) -> npt.NDArray[np.floating]:
        """The davies bouldin index.

        Returns
        -------
        ndarray of float, shape (Lumping.n_i,)
            Array contains the Davies Bouldin index for each run.

        References
        ----------
        [1] David L. Davies; Donald W. Bouldin, "A Cluster Separation
        Measure", IEEE Transactions on Pattern Analysis and Machine
        Intelligence, Volume PAMI-1, Issue 2, pp. 224-227, 1979,
        DOI: 10.1109/TPAMI.1979.4766909
        """
        if self._davies_bouldin_index is None:
            self._davies_bouldin_index = np.zeros(self.n_runs)
            for i in range(self.n_runs):
                self._davies_bouldin_index[i] = davies_bouldin_score(
                    self.multi_feature_trajectory, self.macrostate_trajectory[i]
                )
        return self._davies_bouldin_index

    @property
    def gmrq(self) -> npt.NDArray[np.floating]:
        """The generalized matrix rayleigh quotient (GMRQ).

        Returns
        -------
        ndarray of float, shape (Lumping.n_i,)
            Array contains the GMRQ for each run.

        References
        ----------
        [1] Robert T. McGibbon; Vijay S. Pande, "Variational
        cross-validation of slow dynamical modes in molecular
        kinetics", The Journal of Chemical Physics, Volume 142,
        Issue 12, 2015, DOI: 10.1063/1.4916292
        """
        if self._gmrq is None:
            self._gmrq = utils.gmrq(self.macrostate_tmat)
        return self._gmrq

    @property
    def gmrq2(self) -> npt.NDArray[np.floating]:
        """Sum of the first three eigenvalue squares."""
        if self._gmrq2 is None:
            self._gmrq2 = utils.gmrq2(self.macrostate_tmat)
        return self._gmrq2

    @property
    def macro_micro_feature(self) -> npt.NDArray[np.floating]:
        """Assign macrostate feature values to corresponding microstates.

        This is useful for the analysis of stochastic lumpings.

        Returns
        -------
        ndarray of float, shape (n_states, n_runs)
            Contains macrostate feature values for each microstate.

        See Also
        --------
        MPP.plot.macro_feature
        """
        if self._macro_micro_feature is None:
            self._macro_micro_feature = np.zeros(
                (self.n_states, self.n_runs),
                dtype=self.mean_feature_trajectory.dtype.type,
            )
            for i, (ma, mf) in enumerate(
                zip(self.macrostate_assignment, self.macrostate_feature)
            ):
                for j, mb in enumerate(ma.astype(bool)):
                    self._macro_micro_feature[mb, i] = mf[j]
        return self._macro_micro_feature

    @property
    def tree(self) -> list[core.BinaryTreeNode]:
        """The lumping tree.

        This property holds the lumping trees of all lumpings performed
        by this object. The root node is stored for each lumping.


        """
        if self._tree is None:
            self._tree = []
            for z, pop in zip(self.Z, self.full_pop):
                self._tree.append(self.build_tree(z, pop))
        return self._tree

    def build_tree(self, Z, full_pop):
        """Build tree using BinaryTreeNode and return root"""
        n = Z.shape[0] + 1
        nodes = {}
        for i, (state, target_state, q, pop) in enumerate(Z):
            state = int(state)
            target_state = int(target_state)
            if state not in nodes:
                nodes[state] = core.BinaryTreeNode(
                    state,
                    self.tmat,
                    population=full_pop[state],
                    q=q,
                    pop_thr=self.pop_thr,
                    q_min=self.q_min,
                )
            if target_state not in nodes:
                nodes[target_state] = core.BinaryTreeNode(
                    target_state,
                    self.tmat,
                    population=full_pop[target_state],
                    q=q,
                    pop_thr=self.pop_thr,
                    q_min=self.q_min,
                )
            nodes[n + i] = core.BinaryTreeNode(
                n + i,
                self.tmat,
                q=q,
                pop_thr=self.pop_thr,
                q_min=self.q_min,
            )
            nodes[n + i].left = nodes[state]
            nodes[n + i].right = nodes[target_state]
        for node in nodes[n + i].leaves:
            node.feature = self.mean_feature[node.name]
        return nodes[n + i]

    @property
    def trajectory(self) -> npt.NDArray[np.int_]:
        """The microstate trajectory - 0-based."""
        return self._trajectory

    @trajectory.setter
    def trajectory(self, value):
        if value.min() == 1:
            value -= 1
            warnings.warn("1-based trajectory was shifted to 0-based.")
        if np.unique(value).shape[0] > value.max() + 1:
            raise ValueError("The state numbering in the trajectory is not continuous")
        if value.max() < 2**7:
            trajectory_type = np.uint8
        elif value.max() < 2**15:
            trajectory_type = np.uint16
        else:
            trajectory_type = np.uint32
        self._trajectory = value.astype(trajectory_type)

    @property
    def topology_file(self):
        """The topology_file property."""
        if self._topology_file is None:
            raise ValueError("No topology file set.")
        return self._topology_file

    @topology_file.setter
    def topology_file(self, value):
        if os.path.isfile(value):
            self._topology_file = value
        else:
            raise FileNotFoundError(f"No such file: {value}")

    @property
    def xtc_trajectory_file(self):
        """The xtc_trajectory_file property."""
        if self._xtc_trajectory_file is None:
            raise ValueError("No xtc trajectory file set.")
        return self._xtc_trajectory_file

    @xtc_trajectory_file.setter
    def xtc_trajectory_file(self, value):
        if os.path.isfile(value):
            self._xtc_trajectory_file = value
        else:
            raise FileNotFoundError(f"No such file: {value}")

    @property
    def frame_length(self) -> float:
        """The frame length in ns."""
        return self._frame_length

    @frame_length.setter
    def frame_length(self, value):
        self._frame_length = value


class Plotter:
    def __init__(self, obj):
        self._obj = obj

    def dendrogram(self, out: str, scale=1, offset=0):
        """Plot dendrogram"""
        plot.plot_tree(
            self._obj.tree[self._obj.n_i],
            self._obj.macrostate_assignment[self._obj.n_i],
            out,
            scale=scale,
            offset=offset,
        )

    def implied_timescales(self, out, use_ref=True, scale=1):
        """
        out: File to write plot
        use_ref: If it for reference trajectory should be plotted
        scale: scaling factor for plot
        """
        if use_ref:
            ref_trajectory = self._obj.reference.macrostate_trajectory[0]
        else:
            ref_trajectory = self._obj.trajectory

        macrostate_trajectory = utils.get_multi_state_trajectory(
            self._obj.macrostate_trajectory[self._obj.n_i], self._obj.limits
        )

        dlagtime = max(1, int(1 / self._obj.frame_length))
        plot.implied_timescales(
            [ref_trajectory, macrostate_trajectory],
            np.arange(1, 4.5 * self._obj.lagtime + dlagtime, dlagtime, dtype=int),
            out,
            frame_length=self._obj.frame_length,
            first_ref=True,
            scale=scale,
            use_ref=use_ref,
            ntimescales=self._obj.timescales.shape[1],
        )

    def macro_feature(self, out, ref=None):
        """
        Plot histogram of feature distribution.

        micro_feature (np.ndarray, NxR): N microstates, R runs, holds feature
                values of respective macrostate
        out (str): file to save the plot
        ref (list[tuple]): list of
                - macrostate_assignment
                - macrostate_feature
                - color
                - label
                of the clusterings that should be shown explicitly.
        """
        plot.macro_feature(
            self._obj.macro_micro_feature,
            out,
            self._obj.reference if ref is None else ref,
        )

    def rmsd(self, out, helices=None):
        plot.rmsd(
            self._obj.rmsd, self._obj.macrostate_population[self._obj.n_i], helices, out
        )

    def delta_rmsd(self, out, helices=None):
        plot.delta_rmsd(
            self._obj.rmsd, self._obj.macrostate_population[self._obj.n_i], helices, out
        )

    def contact_rep(self, cluster_file, out, scale=1):
        plot.contact_rep(
            self._obj.multi_feature_trajectory,
            cluster_file,
            self._obj.macrostate_trajectory[self._obj.n_i],
            out,
            utils.get_grid_format(self._obj.n_macrostates[self._obj.n_i]),
            scale=scale,
        )

    def relative_implied_timescales(self, out):
        plot.relative_implied_timescales(self._obj, out)

    def ck_test(self, out):
        plot.chapman_kolmogorov(self._obj, out, self._obj.frame_length)

    def state_network(self, out):
        plot.state_network(self._obj, out)

    def stochastic_state_similarity(self, out):
        plot.stochastic_state_similarity(self._obj, self._obj.reference, out)

    def transition_matrix(self, out):
        plot.transition_matrix(self._obj.macrostate_tmat[self._obj.n_i], out)

    def transition_time(self, out):
        plot.transition_time(
            self._obj.macrostate_tmat[self._obj.n_i],
            out,
            lagtime=self._obj.lagtime,
            frame_length=self._obj.frame_length,
        )

    def graph(self, out, u=0, f=0):
        draw_knetwork(
            self._obj.macrostate_trajectory[self._obj.n_i],
            self._obj.lagtime,
            self._obj.mean_feature_trajectory,
            out,
            u=u,
            f=f,
        )

    def sankey(self, out, ax=None, scale=1):
        plot.sankey_diagram(self._obj, self._obj.reference, out, ax=ax, scale=scale)

    def macrostate_trajectory(self, out, row_length=0.2):
        plot.state_trajectory(
            self._obj.macrostate_trajectory[self._obj.n_i],
            out,
            row_length=row_length,
            frame_length=self._obj.frame_length,
        )
