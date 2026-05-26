# Python API Usage Guide

This guide covers practical usage of the `MPP.Lumping` class and supporting
classes for deterministic lumping workflows.

---

## Installation

```bash
pip install mpp-lumping
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

All metrics are lazy-computed properties, cached on first access, and return an
`ndarray` of shape `(n_runs,)`. For a single deterministic run, index `[0]` to
get a scalar.

Print all metrics at once from the CLI with:

```bash
python -m MPP.run config.yml T none -Z results/t/Z.npy --metrics
```

### Implied Timescales

```python
# Compute implied timescales (shape: n_runs × n_timescales)
ts = mpp.timescales
# Or compute a specific number:
mpp.calc_timescales(n=5)
ts = mpp.timescales   # shape (n_runs, 5)
```

Timescales are in nanoseconds when `frame_length` is provided in ns.

### Shannon Entropy

```python
h = mpp.shannon_entropy   # shape (n_runs,), range [0, 1]
```

Measures how evenly frames are distributed across macrostates. 0 = single
dominant macrostate, 1 = perfectly uniform distribution.

### Davies-Bouldin Index

```python
db = mpp.davies_bouldin_index   # shape (n_runs,), range [0, ∞)
```

Ratio of within-cluster scatter to between-cluster separation. Lower values
indicate better-separated macrostates. Requires `multi_feature_trajectory`.

### GMRQ and GMRQ2

```python
gmrq  = mpp.gmrq    # shape (n_runs,)
gmrq2 = mpp.gmrq2   # shape (n_runs,)
```

The Generalized Matrix Rayleigh Quotient (GMRQ) is the sum of the 2nd–4th
largest eigenvalues of the macrostate transition matrix — higher values indicate
better-preserved slow dynamics. GMRQ2 uses the sum of squares of those eigenvalues.

### RMSD Sharpness

```python
sharpness = mpp.rmsd_sharpness()   # float
```

Population-weighted mean of per-macrostate mean RMSDs. Lower values indicate
more compact macrostates. Requires RMSD data (call `mpp.rmsd` or load via
`mpp.load_rmsd(path)` first).

### Silhouette Coefficient

```python
s = mpp.silhouette   # shape (n_runs,), range [-1, 1]
```

Measures how similar each frame is to its own macrostate compared to neighbouring
macrostates. Values near +1 indicate well-separated, compact macrostates; values
near −1 indicate misclassified frames. Requires `multi_feature_trajectory` and
at least 2 macrostates.

### Calinski–Harabász Index

```python
ch = mpp.calinski_harabasz   # shape (n_runs,), range [0, ∞)
```

Ratio of between-macrostate dispersion to within-macrostate dispersion. Higher
values indicate more compact, well-separated macrostates. Requires
`multi_feature_trajectory` and at least 2 macrostates.

---

## Saving the Macrostate Trajectory

```python
mpp.save_macrostate_trajectory("results/t/macrotraj.txt", one_based=False)
```

The output is a plain-text file with one integer per line.

---

## Stochastic Lumping

Pass `n > 1` to `run_mpp` to perform multiple randomised lumping runs. Use a
`seed` in the `LumpingKernel` for reproducible results.

```python
kernel = MPP.kernel.LumpingKernel(similarity="T", method="n", param=2, seed=42)
mpp.run_mpp(kernel, n=10)

# Access results per run — shape (n_runs, ...)
for i in range(mpp.n_runs):
    print(f"Run {i}: {mpp.n_macrostates[i]} macrostates")

# Stochastic-specific plots
mpp.plot.stochastic_state_similarity("results/stoch/state_similarity.pdf")
mpp.plot.relative_implied_timescales("results/stoch/rel_timescales.pdf")
mpp.plot.macro_feature("results/stoch/macro_feature.pdf")
```

When using the config-based workflow (`MPP.run.Data`), stochastic parameters
are set in the YAML `stochastic` block (see CLI guide for details). The `seed`
key is optional; omit it for a random seed.

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
