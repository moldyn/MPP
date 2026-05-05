# Product Requirements Document — MPP

## Purpose

MPP (Most Probable Path) is a Python package that coarse-grains the number of
discrete states of a Markov process derived from molecular dynamics (MD)
simulations. Given a microstate trajectory, it estimates a Markov state model
(MSM), then iteratively merges the least metastable microstate with its most
similar neighbour to build a lumping tree. The tree is parsed in reverse order
to identify macrostates that satisfy a user-defined minimum population and
minimum metastability criterion.

## Target Users

MD researchers and computational biophysicists who need to reduce the
dimensionality of microstate trajectories into interpretable macrostates and
want quantitative quality metrics for those macrostates.

## Core Functionality

### Tree Building (`core.cluster`, `MPP.Lumping.run_mpp`)
- Iteratively select the least metastable active microstate.
- Merge it with its most similar state according to a pluggable kernel:
  - `T` — transition probability (default).
  - `KL` — Kullback-Leibler divergence of transition probability rows.
  - `JS` — Jensen-Shannon divergence of a feature trajectory.
  - Stochastic variants controlled by `stochastic` config block.
- G-PCCA coarse-graining as an alternative lumping strategy (`gpcca`).
- Results are stored in a Z matrix compatible with
  `scipy.cluster.hierarchy.linkage` conventions and persisted to `.npy`.

### Macrostate Assignment (`BinaryTreeNode`, `Lumping`)
- Parse the lumping tree root-to-leaf to find nodes that satisfy
  `pop_thr` (minimum population fraction) and `q_min` (minimum metastability).
- Assign each microstate to its most probable macrostate via transition
  probabilities.
- Expose `macrostate_assignment`, `macrostate_map`, and
  `macrostate_trajectory` on the `Lumping` object.

### Quality Metrics
- Implied timescales and relative implied timescales.
- Chapman-Kolmogorov test (`ck_test`).
- RMSD per macrostate (C-alpha or feature space).
- Delta-RMSD between macrostates.
- Davies-Bouldin score for cluster quality.
- Sankey diagram of microstate-to-macrostate flows.
- Transition matrix visualisation.
- Transition times.

### Plots (`MPP.plot`)
All plots are generated via the `plot` module and invocable through all three
interfaces. Supported kinds: `dendrogram`, `timescales`, `sankey`, `contacts`,
`macrotraj`, `ck_test`, `rmsd`, `delta_rmsd`, `state_network`, `macro_feature`,
`stochastic_state_similarity`, `relative_implied_timescales`,
`transition_matrix`, `transition_time`, `macrostate_trajectory`.

## Interfaces

### CLI (Primary Interface)
Invoked as `python -m MPP.run <config.yml> <d> <g> [options]`.

| Argument | Description |
|---|---|
| `data_specification` | YAML config file |
| `d` | Dynamic similarity kernel (`T`, `KL`, `none`, `gpcca`) |
| `g` | Geometric similarity kernel (`JS`, `none`, or macrostate count for gpcca) |
| `-Z` | Path to load/save Z matrix |
| `-o` | Output file for plots |
| `-p` | Plot kind (see above) |
| `-r N` | Write N random frame indices per macrostate |
| `--rmsd` | Compute and save RMSD |
| `--rmsd-feature` | `CA` or `feature` |
| `--get-least-moving-residues` | Write least-moving residue indices |

### Python API
The primary objects are:

- `MPP.Lumping` — main analysis object.
- `MPP.run.Data` — convenience wrapper that reads a YAML config and drives
  `Lumping`.
- `MPP.kernel.LumpingKernel` / `MPP.kernel.FeatureKernel` — pluggable kernels.

All functionality available through the CLI must be reachable through the API.

### Snakemake Workflow (`workflow/`)
Provides a high-level interface for batch analyses across multiple systems.
Rules are driven by a YAML config per system and a top-level `Snakefile`.
Output files are requested by target path; Snakemake resolves dependencies
automatically. The workflow must stay in sync with the CLI arguments.

## Non-Goals

- No graphical user interface (GUI).
- No web service or REST API.
- No trajectory simulation or force-field evaluation.
- No automatic hyperparameter search for `pop_thr` / `q_min`.
- No support for non-Markovian processes.

## Quality Goals

| Goal | Description |
|---|---|
| Reproducibility | Given the same input files and config, all three interfaces must produce bit-identical Z matrices (deterministic kernels) or statistically equivalent results (stochastic kernels). |
| Scientific correctness | Algorithmic changes require tests that verify numerical outputs against known-good reference data stored in `tests/data/`. |
| Interface consistency | Every analysis capability must be accessible through CLI, API, and Snakemake with identical semantics and parameter names. |
| Test coverage | Every public function in `MPP/` must be exercised by at least one test in `tests/`. |
