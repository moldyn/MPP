# Snakemake Workflow Usage Guide

The MPP package includes a Snakemake workflow that automates lumping and plot
generation across multiple molecular systems and kernel configurations.

---

## Prerequisites

- Snakemake installed in the environment
- MPP installed (the workflow uses the `mpp.yml` conda environment spec in
  `workflow/`)
- Input data organized as described below

---

## Directory Structure

Copy the `workflow/` directory into your working directory alongside your data:

```
<workdir>/
├── workflow/
│   ├── Snakefile
│   ├── lumpings.yml
│   ├── mpp.yml
│   └── ...
└── example/                    # data root (configurable)
    └── <SystemName>/
        ├── input/
        │   ├── config.yml
        │   ├── traj            # microstate trajectory
        │   └── feature_traj   # multi-feature trajectory
        └── results/
            └── <lumping>/
                ├── Z.npy
                └── *.pdf
```

The `data_root` variable at the top of `workflow/Snakefile` controls where
data is read from. It defaults to `"example"`.

---

## Configuration Files

### Per-system config: `<data_root>/<SystemName>/input/config.yml`

Specifies input file paths and lumping parameters. See
[`docs/usage_cli.md`](usage_cli.md) for the full key reference.

```yaml
source: example/SampleSystem/input

microstate_trajectory: traj
multi_feature_trajectory: feature_traj

lagtime: 20
pop_thr: 0.15
q_min: 0.5
```

### Lumping definitions: `workflow/lumpings.yml`

Defines named lumping configurations. Each entry maps a lumping key to
`kernel_similarity` (`d`) and `feature_kernel` (`g`) values.

```yaml
t:
  kernel_similarity: T
  feature_kernel: none

kl:
  kernel_similarity: KL
  feature_kernel: none

t_js:
  kernel_similarity: T
  feature_kernel: JS

kl_js:
  kernel_similarity: KL
  feature_kernel: JS

js:
  kernel_similarity: none
  feature_kernel: JS
```

These keys become the `{lumping}` wildcard in Snakemake rules.

---

## Workflow Rules

### `gen_Z` — Run lumping and save Z matrix

Produces `<data_root>/{system}/results/{lumping}/Z.npy`.

Invokes:
```
python -m MPP.run {config} {d} {g} -Z {output}
```

### `plot` — Generate a single plot

Produces `<data_root>/{system}/results/{lumping}/{plot}.{ext}` where
`{ext}` is `pdf` or `png`.

Invokes:
```
python -m MPP.run {config} {d} {g} -Z {Z} -p {plot} -o {output}
```

### `plot_all` — Generate all standard plots

Collects: `sankey`, `dendrogram`, `ck_test`, `timescales`, `contacts`,
`macrotraj`, `rmsd`, `delta_rmsd` in both `pdf` and `png` formats.

### `rmsd_CA` / `rmsd_feature` — Compute RMSD

Produces RMSD `.npy` and mean frame index `.ndx` files. Requires
`topology_file` and `xtc_file` to be set in the system config.

### `draw_random` (checkpoint) — Draw random frame indices

Produces per-macrostate `.ndx` files in a `random_frames/` directory.
Requires `topology_file` and `xtc_file`.

---

## Running the Workflow

**Generate Z matrices for all systems and lumpings:**

```bash
snakemake --use-conda -j 4 \
    example/SampleSystem/results/t/Z.npy \
    example/SampleSystem/results/kl/Z.npy
```

**Generate a specific plot:**

```bash
snakemake --use-conda -j 1 \
    example/SampleSystem/results/t/dendrogram.pdf
```

**Generate all plots for a system and lumping:**

```bash
snakemake --use-conda -j 4 \
    example/SampleSystem/results/t/plotted
```

The `plotted` sentinel file is created by the `plot_all` rule when all
standard plots exist.

**Dry run (preview without execution):**

```bash
snakemake -n example/SampleSystem/results/t/dendrogram.pdf
```

---

## Relationship to the CLI

Each Snakemake rule calls `python -m MPP.run` directly. The `{lumping}`
wildcard resolves to a named entry in `lumpings.yml`, which provides the
`d` and `g` arguments. This is equivalent to running the CLI commands
manually (see [`docs/usage_cli.md`](usage_cli.md)).

---

## Caching

The `gen_Z` rule is marked `cache: True`. Snakemake will reuse an existing
Z matrix if the inputs have not changed. The CLI also skips recomputation if
the output Z file already exists (unless `overwrite=True` is passed in the
Python API).

---

## Current Limitations

- **Structural rules** (`rmsd_CA`, `rmsd_feature`, `draw_random`,
  `get_random_pdb_frames`, `get_mean_frames_pdb_*`) require a topology file
  (`.pdb`) and an XTC trajectory file. These must be specified in the
  per-system `config.yml` as `topology_file` and `xtc_file`. If these are
  absent, the affected rules cannot run.
- **PyMol rendering** rules (`render_pdb_files`, `draw_cluster_in_states`)
  require PyMol and the `pymol` conda environment (`workflow/pymol.yml`).
- **GPCCA** lumping (`gpcca` entry in `lumpings.yml`) is supported but
  produces a mock Z matrix and is intended for comparison only.
- The `contacts` plot requires a `cluster_file` entry in the system config.
- Stochastic workflows (multiple runs via the `stochastic` config block) are
  not covered by this guide.

---

## Expected Outputs

After a successful `gen_Z` + `plot_all` run, the results directory contains:

```
example/<SystemName>/results/<lumping>/
├── Z.npy                  # Z matrix (lumping tree)
├── dendrogram.pdf/.png
├── sankey.pdf/.png
├── ck_test.pdf/.png
├── timescales.pdf/.png
├── contacts.pdf/.png
├── macrotraj.pdf/.png
├── rmsd.pdf/.png          # requires topology/XTC
├── delta_rmsd.pdf/.png    # requires topology/XTC
└── plotted                # sentinel file
```
