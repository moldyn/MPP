# CLI Usage Guide

MPP is invoked via `python -m MPP.run`. This guide covers inputs, kernel
selection, plot generation, and output interpretation for deterministic
workflows.

---

## Prerequisites

MPP must be installed (`pip install mpp-lumping`). The
following inputs are required:

- A YAML configuration file
- A microstate trajectory file (plain-text, one integer state per line)
- A multi-feature trajectory file (plain-text, one row of floats per frame)

---

## Basic Invocation

```
python -m MPP.run <config.yml> <d> <g> -Z <Z.npy> [-p <plot>] [-o <output>]
```

**Positional arguments:**

| Argument | Description |
|---|---|
| `config.yml` | YAML configuration file |
| `d` | Dynamic similarity selector (`T`, `KL`, `none`, or `gpcca`) |
| `g` | Feature similarity selector (`JS` or `none`) |

**Options:**

| Flag | Description |
|---|---|
| `-Z <path>` | Path to save (or load) the Z matrix (`.npy`). If the file already exists, it is loaded instead of recomputed. |
| `-p <plot>` | Plot type to generate (see [Plot Types](#plot-types)) |
| `-o <output>` | Output file path for the plot or macrostate trajectory |
| `--scale <float>` | Scaling factor for plot size (default `1`) |
| `--n-timescales <N>` | Number of implied timescales to compute (overrides config value) |
| `--rmsd <path>` | Compute C-alpha RMSD and write to `.npy` file |
| `--rmsd-feature <CA\|feature>` | RMSD variant: `CA` (default) or `feature` |
| `-r <N>` | Draw N random frame indices per macrostate (writes `.ndx` files) |
| `--get-least-moving-residues <contact_index_file>` | Write least-varying residues per macrostate to file |

---

## YAML Configuration

The config file specifies input paths and lumping parameters. All keys use
`snake_case`.

```yaml
# example/sample_system/input/config.yml
source: example/sample_system/input

microstate_trajectory: traj
multi_feature_trajectory: feature_traj

lagtime: 20          # lag time in frames
pop_thr: 0.15        # minimum macrostate population (fraction)
q_min: 0.5           # minimum macrostate metastability
frame_length: 0.2    # frame length in ns

contact_threshold: 0.45  # distance threshold to binarise feature (nm)
```

**Required keys:** `source`, `microstate_trajectory`,
`multi_feature_trajectory`, `lagtime`, `pop_thr`, `q_min`, `frame_length`.

**Optional keys:**

| Key | Description |
|---|---|
| `contact_threshold` | Feature binarisation threshold (default `0.45`) |
| `cluster_file` | Contact index file for contact plots |
| `contact_index_file` | Contact pair index file for structural analysis |
| `topology_file` | PDB topology file for structural analysis |
| `xtc_file` | XTC trajectory file for structural analysis |
| `xtc_stride` | Stride for XTC reading |
| `n_timescales` | Number of implied timescales to compute |
| `helices` | Helix residue ranges for RMSD annotation |
| `limits` | Concatenated trajectory lengths (for multiple independent simulations) |

---

## Dynamic Similarity Selectors (`d`)

The `d` argument selects how microstate similarity is computed during lumping.

| `d` | Description |
|---|---|
| `T` | Transition probability (reference kernel; recommended default) |
| `KL` | Kullback-Leibler divergence of transition probability rows |
| `none` | Disable dynamic similarity; use feature kernel only |
| `gpcca` | Use GPCCA instead of MPP (comparison only) |

---

## Feature Similarity Selector (`g`)

The `g` argument optionally incorporates geometric information via feature
distributions.

| `g` | Description |
|---|---|
| `none` | No feature similarity |
| `JS` | Jensen-Shannon divergence of feature distributions |
| `reference_count` | *(gpcca mode only)* Use macrostate count from the reference `T` lumping |
| `<int>` | *(gpcca mode only)* Use a fixed number of macrostates |

---

## Kernel Combinations

| `d` | `g` | Description |
|---|---|---|
| `T` | `none` | Transition probability only (default/reference) |
| `KL` | `none` | Kullback-Leibler divergence only |
| `T` | `JS` | Combined transition probability + feature similarity |
| `KL` | `JS` | Combined KL divergence + feature similarity |
| `none` | `JS` | Feature similarity only |

---

## Examples

**Run lumping with transition probability kernel and save Z matrix:**

```bash
python -m MPP.run example/sample_system/input/config.yml T none \
    -Z results/t/Z.npy
```

**Run with KL divergence kernel:**

```bash
python -m MPP.run example/sample_system/input/config.yml KL none \
    -Z results/kl/Z.npy
```

**Run with combined transition probability + Jensen-Shannon feature kernel:**

```bash
python -m MPP.run example/sample_system/input/config.yml T JS \
    -Z results/t_js/Z.npy
```

**Load an existing Z matrix and generate a dendrogram:**

```bash
python -m MPP.run example/sample_system/input/config.yml T none \
    -Z results/t/Z.npy -p dendrogram -o results/t/dendrogram.pdf
```

**Generate a Sankey diagram:**

```bash
python -m MPP.run example/sample_system/input/config.yml T none \
    -Z results/t/Z.npy -p sankey -o results/t/sankey.pdf
```

**Save the macrostate trajectory as a text file:**

```bash
python -m MPP.run example/sample_system/input/config.yml T none \
    -Z results/t/Z.npy -p macrostate_trajectory -o results/t/macrostate_trajectory.txt
```

---

## Plot Types

Available values for `-p`:

| Plot | Description |
|---|---|
| `dendrogram` | Lumping tree with macrostate boundaries |
| `timescales` | Implied timescales of micro- and macrostate models |
| `sankey` | Sankey diagram comparing this lumping to the reference |
| `contacts` | Contact representation per macrostate |
| `macrotraj` | Macrostate trajectory as a color-coded time series |
| `ck_test` | Chapman-Kolmogorov test |
| `rmsd` | Per-macrostate C-alpha RMSD |
| `delta_rmsd` | Per-macrostate delta RMSD relative to macrostate 0 |
| `state_network` | Macrostate transition network |
| `transition_matrix` | Macrostate transition matrix heatmap |
| `transition_time` | Mean first-passage times between macrostates |
| `macrostate_trajectory` | Write macrostate trajectory to text file (use `-o <file.txt>`) |

---

## Output Interpretation

**Z matrix (`Z.npy`):** Shape `(n_runs, n_states-1, 4)`. Each row encodes one
merge step: `[state_a, state_b, metastability_a, joint_population]`. The Z
matrix is in scipy linkage format with `n_states + i` as the intermediate
cluster index. For deterministic runs, `n_runs = 1`.

**Macrostate map (`macrostate_map.npy`):** Integer array of shape `(n_states,)`.
Entry `i` gives the macrostate index assigned to microstate `i`. Written
automatically to the same directory as `Z.npy` whenever `-Z` is used.

**Macrostate trajectory (text):** One integer per line, 0-based macrostate
index. Written by `-p macrostate_trajectory -o macrostate_trajectory.txt`.

**RMSD file (`.npy`):** Shape `(n_macrostates, n_CA_atoms)`, C-alpha RMSD
values per macrostate.
