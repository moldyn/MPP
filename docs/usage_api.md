# Python API Usage Guide

This guide covers practical usage of the `MPP.Lumping` class and supporting
classes for deterministic lumping workflows.

---

## Installation

```bash
pip install -e .
```

---

## Quick Start

```python
import numpy as np
import MPP

# Load input data
trajectory = np.loadtxt("example/sample_system/input/traj", dtype=np.uint16)
feature_trajectory = np.loadtxt("example/sample_system/input/feature_traj", ndmin=2)

# Create Lumping object
mpp = MPP.Lumping(
    trajectory,
    lagtime=20,
    feature_trajectory=feature_trajectory,
    pop_thr=0.15,
    q_min=0.5,
)

# Run MPP with transition probability kernel (default)
kernel = MPP.kernel.LumpingKernel(similarity="T")
mpp.run_mpp(kernel)

# Access results
print(f"Number of macrostates: {mpp.n_macrostates[0]}")
print(f"Macrostate assignment shape: {mpp.macrostate_assignment[0].shape}")
```

---

## `MPP.Lumping`

The central class. Holds the microstate trajectory, transition matrix,
feature data, and all results.

### Constructor

```python
MPP.Lumping(
    trajectory,           # ndarray of int, shape (N,) — microstate trajectory
    lagtime,              # int — lag time in frames
    feature_trajectory=None,  # ndarray of float, shape (N, M) — optional
    contact_threshold=0.45,   # float — feature binarisation threshold (nm)
    pop_thr=0.005,        # float — minimum macrostate population
    q_min=0.5,            # float — minimum macrostate metastability
    frame_length=0.2,     # float — frame length in ns
    limits=None,          # list of int — trajectory lengths for concatenated runs
    quiet=False,          # bool — suppress progress output
)
```

The `trajectory` must be 0-based and contiguous. 1-based trajectories are
shifted automatically with a warning.

### `run_mpp`

Runs the MPP algorithm and populates all macrostate attributes.

```python
mpp.run_mpp(
    kernel=MPP.kernel.LumpingKernel(),  # LumpingKernel instance
    feature_kernel=None,                # FeatureKernel or None
    n=1,                                # int — number of runs (use 1 for deterministic runs)
)
```

### `load_Z` / `save_Z`

```python
mpp.save_Z("results/Z.npy")
mpp.load_Z("results/Z.npy")   # also calls assign_macrostates()
```

`load_Z` accepts either a path string or a NumPy array directly.

### `assign_macrostates`

Re-parse the lumping tree without re-running the algorithm. Useful after
manually changing `pop_thr` or `q_min`.

```python
mpp.pop_thr = 0.05
mpp.q_min = 0.6
mpp.assign_macrostates()
```

---

## Lumping Kernels

### `MPP.kernel.LumpingKernel`

Determines which microstate is merged and with which neighbour.

```python
# Transition probability (default, recommended)
kernel = MPP.kernel.LumpingKernel(similarity="T")

# Kullback-Leibler divergence
kernel = MPP.kernel.LumpingKernel(similarity="KL")

# Feature-only (use with FeatureKernel)
kernel = MPP.kernel.LumpingKernel(similarity="none")
```

### `MPP.kernel.FeatureKernel`

Incorporates geometric similarity via Jensen-Shannon divergence of feature
distributions. Pass alongside `LumpingKernel` to `run_mpp`.

```python
feature_kernel = MPP.kernel.FeatureKernel(
    feature_trajectory,   # binary feature trajectory, shape (N, M)
    trajectory,           # microstate trajectory, shape (N,)
)

mpp.run_mpp(kernel, feature_kernel=feature_kernel)
```

### Kernel Combinations

| `LumpingKernel(similarity=...)` | `feature_kernel` | Equivalent CLI |
|---|---|---|
| `"T"` | `None` | `T none` |
| `"KL"` | `None` | `KL none` |
| `"T"` | `FeatureKernel(...)` | `T JS` |
| `"KL"` | `FeatureKernel(...)` | `KL JS` |
| `"none"` | `FeatureKernel(...)` | `none JS` |

---

## Accessing Results

After calling `run_mpp` or `load_Z`, the following attributes are populated.
For deterministic runs, index `[0]` selects the single run.

```python
# Number of macrostates
n = mpp.n_macrostates[0]

# Macrostate assignment: bool array, shape (n_macrostates, n_states)
assignment = mpp.macrostate_assignment[0]

# Map from microstate index to macrostate index, shape (n_states,)
macro_map = mpp.macrostate_map[0]

# Macrostate trajectory, shape (n_runs, n_frames)
macrotraj = mpp.macrostate_trajectory[0]

# Macrostate transition matrix, shape (n_macrostates, n_macrostates)
macrotmat = mpp.macrostate_tmat[0]

# Macrostate populations (in frames), shape (n_macrostates,)
pop = mpp.macrostate_population[0]

# Mean feature value per macrostate, list of float
feature = mpp.macrostate_feature[0]

# Z matrix, shape (n_runs, n_states-1, 4)
Z = mpp.Z
```

---

## Config-Based Workflow

For production use, `MPP.run.Data` reads a YAML config file and orchestrates
the full workflow.

```python
from MPP.run import Data

data = Data("example/sample_system/input/config.yml")
data.setup_mpp("T", "none")          # d="T", g="none"
data.perform_mpp("results/t/Z.npy")  # run or load Z

mpp = data.mpp
print(f"Macrostates: {mpp.n_macrostates[0]}")
```

`perform_mpp` loads an existing Z matrix if the file is already present;
pass `overwrite=True` to force recomputation.

---

## Generating Plots

Plots are accessed via `mpp.plot`, an instance of `Lumping.Plotter`.

```python
mpp.plot.dendrogram("results/t/dendrogram.pdf")
mpp.plot.implied_timescales("results/t/timescales.pdf")
mpp.plot.sankey("results/t/sankey.pdf")
mpp.plot.ck_test("results/t/ck_test.pdf")
mpp.plot.macrostate_trajectory("results/t/macrotraj.pdf")
mpp.plot.state_network("results/t/state_network.pdf")
mpp.plot.transition_matrix("results/t/transition_matrix.pdf")
mpp.plot.transition_time("results/t/transition_time.pdf")
```

Contact and RMSD plots require additional files:

```python
# Contact representation (requires cluster_file)
mpp.plot.contact_rep("path/to/cluster_file", "results/t/contacts.pdf")

# RMSD plots (requires topology and XTC files)
mpp.topology_file = "path/to/structure.pdb"
mpp.xtc_trajectory_file = "path/to/trajectory.xtc"
mpp.plot.rmsd("results/t/rmsd.pdf")
mpp.plot.delta_rmsd("results/t/delta_rmsd.pdf")
```

---

## Quality Metrics

```python
# Implied timescales, shape (n_runs, n_timescales)
ts = mpp.timescales

# Shannon entropy of macrostate populations, shape (n_runs,)
h = mpp.shannon_entropy

# Davies-Bouldin index (lower = better separated), shape (n_runs,)
db = mpp.davies_bouldin_index

# GMRQ (generalized matrix Rayleigh quotient), shape (n_runs,)
gmrq = mpp.gmrq
```

---

## Saving the Macrostate Trajectory

```python
mpp.save_macrostate_trajectory("results/t/macrotraj.txt", one_based=False)
```

The output is a plain-text file with one integer per line.

---

## Concatenated Trajectories

When the microstate trajectory is composed of several independent simulations,
pass `limits` as the list of individual trajectory lengths:

```python
limits = np.loadtxt("path/to/limits", dtype=int)

mpp = MPP.Lumping(
    trajectory,
    lagtime=20,
    feature_trajectory=feature_trajectory,
    limits=limits,
)
```
