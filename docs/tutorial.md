# Tutorial: End-to-End MPP Analysis

This tutorial walks through a complete MPP analysis using the sample dataset
included in the repository (`example/sample_system/`). The dataset contains a
100 000-frame microstate trajectory with 6 microstates and a corresponding
single-feature trajectory.

Both the CLI and Python API approaches are shown side by side.

---

## Setup

Install the package:

```bash
pip install mpp-lumping
```

Clone the repository to access the example data:

```bash
git clone https://github.com/moldyn/MPP.git
cd MPP
```

---

## Step 1 — Inspect the input data

The example config file is `example/sample_system/input/config.yml`:

```yaml
source: example/sample_system/input

microstate_trajectory: traj
multi_feature_trajectory: feature_traj
contact_threshold: null

frame_length: 0.2   # ns per frame
lagtime: 20         # frames
pop_thr: 0.15       # minimum macrostate population
q_min: 0.5          # minimum macrostate metastability
```

**Python API:**

```python
import numpy as np

traj = np.loadtxt("example/sample_system/input/traj", dtype=int)
feat = np.loadtxt("example/sample_system/input/feature_traj", ndmin=2)

print(f"Trajectory: {traj.shape[0]} frames, {len(np.unique(traj))} unique microstates")
print(f"Features:   {feat.shape[1]} feature(s) per frame")
# Trajectory: 100000 frames, 6 unique microstates
# Features:   1 feature(s) per frame
```

---

## Step 2 — Run the MPP lumping

**CLI:**

```bash
mkdir -p results/t
python -m MPP.run example/sample_system/input/config.yml T none \
    -Z results/t/Z.npy
```

The `-Z results/t/Z.npy` flag saves the lumping tree. If the file already exists,
it is loaded instead of recomputed.

**Python API:**

```python
import MPP
import MPP.kernel

kernel = MPP.kernel.LumpingKernel(similarity="T")
mpp = MPP.Lumping(
    traj,
    lagtime=20,
    feature_trajectory=feat,
    pop_thr=0.15,
    q_min=0.5,
    frame_length=0.2,
)
mpp.run_mpp(kernel)
```

---

## Step 3 — Inspect macrostate results

**Python API:**

```python
n = mpp.n_macrostates[0]
print(f"Number of macrostates: {n}")
# Number of macrostates: 3

# Map from microstate index to macrostate index
print(f"Macrostate map: {mpp.macrostate_map[0]}")

# Population of each macrostate (number of frames)
pop = mpp.macrostate_population[0]
print(f"Populations: {pop}")
print(f"Fractions:   {pop / pop.sum():.3f}")
```

---

## Step 4 — Quality metrics

**CLI:**

```bash
python -m MPP.run example/sample_system/input/config.yml T none \
    -Z results/t/Z.npy --metrics
# shannon_entropy=0.99240...
# davies_bouldin=0.43912...
# gmrq=...
# gmrq2=...
# silhouette=...
# calinski_harabasz=...
```

**Python API:**

```python
print(f"Shannon entropy:    {mpp.shannon_entropy[0]:.4f}")   # 0.9924
print(f"Davies-Bouldin:     {mpp.davies_bouldin_index[0]:.4f}")  # 0.4391
print(f"GMRQ:               {mpp.gmrq[0]:.4f}")
print(f"Silhouette:         {mpp.silhouette[0]:.4f}")
print(f"Calinski-Harabász:  {mpp.calinski_harabasz[0]:.1f}")
```

---

## Step 5 — Generate plots

**CLI:**

```bash
# Lumping dendrogram
python -m MPP.run example/sample_system/input/config.yml T none \
    -Z results/t/Z.npy -p dendrogram -o results/t/dendrogram.pdf

# Macrostate trajectory (color-coded time series)
python -m MPP.run example/sample_system/input/config.yml T none \
    -Z results/t/Z.npy -p macrotraj -o results/t/macrotraj.pdf

# Chapman-Kolmogorov test
python -m MPP.run example/sample_system/input/config.yml T none \
    -Z results/t/Z.npy -p ck_test -o results/t/ck_test.pdf
```

**Python API:**

```python
mpp.plot.dendrogram("results/t/dendrogram.pdf")
mpp.plot.macrostate_trajectory("results/t/macrotraj.pdf")
mpp.plot.ck_test("results/t/ck_test.pdf")
mpp.plot.state_network("results/t/state_network.pdf")
mpp.plot.transition_matrix("results/t/transition_matrix.pdf")
```

---

## Step 6 — Save the macrostate trajectory

**CLI:**

```bash
python -m MPP.run example/sample_system/input/config.yml T none \
    -Z results/t/Z.npy \
    -p macrostate_trajectory -o results/t/macrostate_trajectory.txt
```

**Python API:**

```python
mpp.save_macrostate_trajectory("results/t/macrostate_trajectory.txt", one_based=True)
```

The output file contains one macrostate index per line (1-based).

---

## Step 7 — Try other kernels

Replace `T none` with any supported kernel combination to compare results:

```bash
# Kullback-Leibler divergence
python -m MPP.run example/sample_system/input/config.yml KL none \
    -Z results/kl/Z.npy --metrics

# Combined T + Jensen-Shannon feature similarity
python -m MPP.run example/sample_system/input/config.yml T JS \
    -Z results/t_js/Z.npy --metrics
```

See the [CLI Usage guide](usage_cli.md) for the full list of kernel combinations.

---

## Step 8 — Config-based Python workflow

For production workflows, `MPP.run.Data` reads the YAML config and orchestrates
the full pipeline, matching the CLI behaviour exactly:

```python
from MPP.run import Data

data = Data("example/sample_system/input/config.yml")
data.setup_mpp("T", "none")
data.perform_mpp("results/t/Z.npy")   # loads if file exists

mpp = data.mpp
print(f"Macrostates: {mpp.n_macrostates[0]}")
mpp.plot.dendrogram("results/t/dendrogram.pdf")
```

---

## Next steps

- Read the [CLI Usage guide](usage_cli.md) for the full argument reference.
- Read the [Python API guide](usage_api.md) for advanced usage (stochastic lumping,
  quality metrics, RMSD, concatenated trajectories).
- Explore all [plot types](plots.md) with CLI and API examples.
