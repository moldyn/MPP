# Plot Types

MPP can generate 15 types of plots, all accessible through both the CLI (`-p <plot>`) and
the Python API (`mpp.plot.<method>(out)`).

---

## Dendrogram

**What it shows:** The lumping tree with macrostate boundaries and a contact-fraction
colorbar. A second panel shows the macrostate assignment as a color-coded grid.

**CLI:**
```bash
python -m MPP.run config.yml T none -Z Z.npy -p dendrogram -o dendrogram.pdf
```

**API:**
```python
mpp.plot.dendrogram("dendrogram.pdf")
```

---

## Implied Timescales

**What it shows:** Implied timescales of the microstate model and the macrostate model
across multiple lag times.

**CLI:**
```bash
python -m MPP.run config.yml T none -Z Z.npy -p timescales -o timescales.pdf
```

**API:**
```python
mpp.calc_timescales(n=5)
mpp.plot.implied_timescales("timescales.pdf")
```

---

## Sankey Diagram

**What it shows:** Microstate flow between the macrostates of this lumping and the
reference (`T none`) lumping.

**CLI:**
```bash
python -m MPP.run config.yml KL none -Z Z.npy -p sankey -o sankey.pdf
```

**API:**
```python
mpp.plot.sankey("sankey.pdf")
```

---

## Contact Representation

**What it shows:** Mean contact distances (or other features) per macrostate, using
a cluster file to label contacts.

**Requires:** `cluster_file` key in YAML config (contact index file).

**CLI:**
```bash
python -m MPP.run config.yml T none -Z Z.npy -p contacts -o contacts.pdf
```

**API:**
```python
mpp.plot.contact_rep("path/to/cluster_file", "contacts.pdf")
```

---

## Macrostate Trajectory

**What it shows:** The macrostate trajectory as a color-coded horizontal time series.

**CLI:**
```bash
python -m MPP.run config.yml T none -Z Z.npy -p macrotraj -o macrotraj.pdf
```

**API:**
```python
mpp.plot.macrostate_trajectory("macrotraj.pdf")
```

---

## Chapman-Kolmogorov Test

**What it shows:** Chapman-Kolmogorov test for the macrostate model. Compares
propagated macrostate populations against direct estimates at multiples of the lag time.

**CLI:**
```bash
python -m MPP.run config.yml T none -Z Z.npy -p ck_test -o ck_test.pdf
```

**API:**
```python
mpp.plot.ck_test("ck_test.pdf")
```

---

## RMSD

**What it shows:** Per-macrostate C-alpha RMSD (or feature-space RMSD) relative to the
mean structure of each macrostate.

**Requires:** Topology (`.pdb`) and trajectory (`.xtc`) files set on the `Lumping` object.

**CLI:**
```bash
python -m MPP.run config.yml T none -Z Z.npy -p rmsd -o rmsd.pdf
```

**API:**
```python
mpp.topology_file = "structure.pdb"
mpp.xtc_trajectory_file = "trajectory.xtc"
mpp.plot.rmsd("rmsd.pdf")
```

---

## Delta RMSD

**What it shows:** Per-macrostate RMSD relative to macrostate 0 (instead of each state's
own mean). Highlights structural differences between macrostates.

**Requires:** Same as RMSD above.

**CLI:**
```bash
python -m MPP.run config.yml T none -Z Z.npy -p delta_rmsd -o delta_rmsd.pdf
```

**API:**
```python
mpp.plot.delta_rmsd("delta_rmsd.pdf")
```

---

## State Network

**What it shows:** A graph of macrostates as nodes, with edge widths proportional to
transition probabilities.

**CLI:**
```bash
python -m MPP.run config.yml T none -Z Z.npy -p state_network -o state_network.pdf
```

**API:**
```python
mpp.plot.state_network("state_network.pdf")
```

---

## Transition Matrix

**What it shows:** The macrostate transition matrix as a heatmap. Entries below a
threshold are shown as zero (white).

**CLI:**
```bash
python -m MPP.run config.yml T none -Z Z.npy -p transition_matrix -o tmat.pdf
```

**API:**
```python
mpp.plot.transition_matrix("tmat.pdf")
```

---

## Transition Time

**What it shows:** Mean first-passage times between macrostates.

**CLI:**
```bash
python -m MPP.run config.yml T none -Z Z.npy -p transition_time -o transition_time.pdf
```

**API:**
```python
mpp.plot.transition_time("transition_time.pdf")
```

---

## Macrostate Trajectory (text output)

**What it produces:** Not a plot — writes the macrostate trajectory as a plain-text
file (one integer per line, 1-based macrostate indices).

**CLI:**
```bash
python -m MPP.run config.yml T none -Z Z.npy \
    -p macrostate_trajectory -o macrostate_trajectory.txt
```

**API:**
```python
mpp.save_macrostate_trajectory("macrostate_trajectory.txt", one_based=True)
```

---

## Stochastic State Similarity

**What it shows:** Overlap of macrostate assignments between stochastic lumping runs.
Only meaningful for `n_runs > 1`.

**CLI:**
```bash
python -m MPP.run config_stochastic.yml T none -Z Z.npy \
    -p stochastic_state_similarity -o state_similarity.pdf
```

**API:**
```python
mpp.plot.stochastic_state_similarity("state_similarity.pdf")
```

---

## Relative Implied Timescales

**What it shows:** Implied timescales of each stochastic run relative to the reference
`T none` lumping. Only meaningful for `n_runs > 1`.

**CLI:**
```bash
python -m MPP.run config_stochastic.yml T none -Z Z.npy \
    -p relative_implied_timescales -o rel_timescales.pdf
```

**API:**
```python
mpp.plot.relative_implied_timescales("rel_timescales.pdf")
```

---

## Macro Feature

**What it shows:** Population-weighted mean feature per macrostate across all stochastic
runs, with optional comparison to a reference lumping. Only meaningful for `n_runs > 1`.

**CLI:**
```bash
python -m MPP.run config_stochastic.yml T none -Z Z.npy \
    -p macro_feature -o macro_feature.pdf
```

**API:**
```python
mpp.plot.macro_feature("macro_feature.pdf")
```
