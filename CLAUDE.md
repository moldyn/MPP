# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Layout

Source lives under `src/MPP/`. Key files: `src/MPP/lumping.py` (Lumping class), `src/MPP/run.py` (Data + CLI), `src/MPP/kernel.py`, `src/MPP/core.py`.

## Commands

### Environment Setup (NixOS)
On NixOS, Python is not available directly. You must enter the nix-shell and activate the prepared conda environment before running any Python commands:
```bash
nix-shell ~/Documents/shell.nix
micromamba activate mpp-test
```
After these two steps, `python`, `MPP`, and the test suite are available. The `mpp-test` environment has MPP installed in editable mode.

When running commands non-interactively (e.g. from a script or Bash tool), use the heredoc form — `nix-shell --run "..."` does not produce output reliably:
```bash
nix-shell ~/Documents/shell.nix <<'EOF'
micromamba run -n mpp-test <command>
EOF
```

Only use heredocs when explicitly required (e.g. the nix-shell pattern above). Do not use heredocs for `git commit` or other commands where a plain `-m` flag suffices.

For commits, use `git commit -am "..."`. Only use `git add` when a new (untracked) file needs to be included.

### Run all tests with coverage
```bash
bash run_all_tests.sh
# equivalent to:
coverage run --branch --source=src/MPP -m unittest_parallel --level test --coverage-branch --coverage-html htmlcov
```

### Run CORE tests only (currently in scope)
```bash
python -m pytest tests/test_properties.py tests/test_utils.py \
  tests/test_run.py::TestRunScript::test_HP35_t_ref \
  tests/test_run.py::TestRunScript::test_HP35_kl \
  tests/test_run.py::TestRunScript::test_HP35_t_js \
  tests/test_run.py::TestRunScript::test_HP35_js \
  tests/test_run.py::TestRunScript::test_PDZ3_kl \
  tests/test_run.py::TestRunScript::test_aSyn_t \
  tests/test_run.py::TestRunScript::test_aSyn_kl_js \
  tests/test_plots.py::TestPlotting::test_manual_ck_test \
  tests/test_plots.py::TestPlotting::test_manual_contacts \
  tests/test_plots.py::TestPlotting::test_manual_dendrogram \
  tests/test_plots.py::TestPlotting::test_manual_macrotraj_PDZ3 \
  tests/test_plots.py::TestPlotting::test_manual_macrotraj_ref \
  tests/test_plots.py::TestPlotting::test_manual_sankey \
  tests/test_plots.py::TestPlotting::test_manual_state_network \
  tests/test_plots.py::TestPlotting::test_manual_timescales -v
```
See `tests/TEST_CATEGORIES.md` for the full classification (CORE / OPTIONAL / DEFERRED). OPTIONAL and DEFERRED tests may fail due to missing data files (topology, xtc) — this is expected and out of scope.

### Run a single test
```bash
python -m unittest tests.test_run.TestRunScript.test_HP35_t_ref
```

### Run via CLI
```bash
python -m MPP.run <config.yml> <d> <g> -Z <Z.npy> -p <plot_type> -o <output>
# Example:
python -m MPP.run example/sample_system/input/config.yml T none -Z results/Z.npy -p dendrogram -o results/dendrogram.pdf
```

## Architecture

### Algorithm Overview
MPP is a two-step coarse-graining algorithm for Markov state models:
1. **Build lumping tree** (`core.cluster`): Iteratively merges the least metastable microstate with its most similar neighbor, recording merges in a Z matrix (scipy linkage format: `[state_a, state_b, metastability_a, joint_population]`).
2. **Assign macrostates** (`Lumping.assign_macrostates`): Parses the binary tree top-down to identify macrostates satisfying minimum metastability (`q_min`) and population (`pop_thr`) thresholds.

### Key Classes

**`MPP.Lumping`** (`src/MPP/lumping.py`) — Central user-facing class. Holds the microstate trajectory, transition matrix, feature data, and all results. Usage pattern:
```python
mpp = MPP.Lumping(trajectory, lagtime, feature_trajectory, pop_thr=0.005, q_min=0.5)
mpp.run_mpp(kernel, feature_kernel)
# Access results via mpp.macrostate_trajectory, mpp.macrostate_assignment, etc.
```
The `Plotter` inner class (accessed via `mpp.plot`) delegates to `src/MPP/plot.py`.

**`MPP.kernel.LumpingKernel`** (`src/MPP/kernel.py`) — Determines which state to merge next. Configurable with:
- `similarity`: `"T"` (transition probability), `"KL"` (Kullback-Leibler divergence), `"none"` (feature-only)
- `method`/`param`: `"n"`/int for deterministic or top-N stochastic; `"p"`/float for probability-mass-based stochastic

**`MPP.kernel.FeatureKernel`** (`src/MPP/kernel.py`) — Optional geometric similarity via Jensen-Shannon divergence of feature distributions. Tracks population-weighted mean features for merged states. Passed alongside `LumpingKernel` to incorporate both dynamic and geometric similarity.

**`core.BinaryTreeNode`** (`src/MPP/core.py`) — `anytree`-based node for the lumping tree. Properties like `is_macrostate`, `assigned_macrostate`, `macrostates` implement the tree-parsing logic of step 2.

**`MPP.run.Data`** (`src/MPP/run.py`) — High-level wrapper that reads a YAML config, instantiates `Lumping`, and orchestrates `run_mpp`/`load_Z`/`save_Z`. Entry point for the CLI (`python -m MPP.run`).

### Similarity Modes (d/g arguments)
| d | g | Description |
|---|---|---|
| `T` | `none` | Transition probability (reference/default) |
| `KL` | `none` | Kullback-Leibler divergence of row distributions |
| `T` or `KL` | `JS` | Combined dynamic + Jensen-Shannon feature similarity |
| `none` | `JS` | Feature-only (JS divergence) |
| `gpcca` | `ref`/int | Use GPCCA instead of MPP |

### Z Matrix Format
Shape `(n_runs, n_states-1, 4)`. Each row: `[state_a, state_b, metastability_a, joint_population]`. Intermediate cluster index = `n_states + i`. Saved/loaded as `.npy`.

## Documentation Hygiene

When modifying the CLI (`src/MPP/run.py`) or Python API (`src/MPP/lumping.py`, `src/MPP/kernel.py`), check whether the corresponding documentation needs updating:

- **CLI help in README.md** — the `--help` output block must match the actual output of `python -m MPP.run --help`. Regenerate it with:
  ```bash
  python -m MPP.run --help
  ```
- **`docs/usage_cli.md`** — update if argument names, defaults, or behaviour change.
- **`docs/usage_api.md`** — update if class/method signatures, parameters, or attributes change.

## Commit Messages

- Must include the issue number, e.g. `TASK-2.2 (#22): ...`
- Must NOT reference Claude Code, any AI model, or include `Co-Authored-By` lines

### Test Structure
- `tests/data/<dataset>/input/` — Input files (microstate trajectory, feature trajectory, config YAML)
- `tests/data/<dataset>/expected_output/<lumping_key>/` — Reference Z matrices and plots
- `tests/data/<dataset>/baseline/` — Baseline Z matrices and macrostate assignments for regression tests
- `tests/data/lumpings.yml` — Maps lumping keys (e.g. `t`, `kl`, `t_js`) to kernel parameters
- Three test datasets: `HP35`, `PDZ3`, `aSyn`
