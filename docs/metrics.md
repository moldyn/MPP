# Quality Metrics

All metrics are lazy-computed properties of `MPP.Lumping`, cached on first access,
and return an `ndarray` of shape `(n_runs,)`. For a single deterministic run,
index `[0]` to get a scalar.

Print all metrics at once from the CLI with:

```bash
python -m MPP.run config.yml T none -Z results/t/Z.npy --metrics
```

---

## Implied Timescales

```python
# Compute implied timescales (shape: n_runs × n_timescales)
ts = mpp.timescales
# Or compute a specific number:
mpp.calc_timescales(ntimescales=5)
ts = mpp.timescales   # shape (n_runs, 5)
```

Timescales are in nanoseconds when `frame_length` is provided in ns.

---

## Shannon Entropy

```python
h = mpp.shannon_entropy   # shape (n_runs,), range [0, 1]
```

Normalized Shannon entropy of the macrostate population distribution:

\[
H = -\frac{1}{\ln K} \sum_{j=1}^{K} p_j \ln p_j
\]

where \(p_j\) is the population fraction of macrostate \(j\) and \(K\) is the
number of macrostates. \(H = 0\) when all frames belong to a single macrostate;
\(H = 1\) when all macrostates are equally populated.

**Reference:** Shannon, C. E. (1948). *A Mathematical Theory of Communication.*
The Bell System Technical Journal, 27(3), 379–423.
[DOI: 10.1002/j.1538-7305.1948.tb01338.x](https://doi.org/10.1002/j.1538-7305.1948.tb01338.x)

---

## Davies-Bouldin Index

```python
db = mpp.davies_bouldin_index   # shape (n_runs,), range [0, ∞)
```

Ratio of within-cluster scatter to between-cluster separation:

\[
\text{DB} = \frac{1}{K} \sum_{i=1}^{K} \max_{j \neq i}
    \frac{s_i + s_j}{d(c_i,\, c_j)}
\]

where \(s_i\) is the mean feature distance from frames in macrostate \(i\) to
their centroid \(c_i\), and \(d(c_i, c_j)\) is the distance between centroids.
Lower values indicate better-separated macrostates. Requires
`multi_feature_trajectory`.

**Reference:** Davies, D. L. & Bouldin, D. W. (1979). *A Cluster Separation
Measure.* IEEE Trans. Pattern Anal. Mach. Intell., PAMI-1(2), 224–227.
[DOI: 10.1109/TPAMI.1979.4766909](https://doi.org/10.1109/TPAMI.1979.4766909)

---

## GMRQ and GMRQ2

```python
gmrq  = mpp.gmrq    # shape (n_runs,)
gmrq2 = mpp.gmrq2   # shape (n_runs,)
```

The Generalized Matrix Rayleigh Quotient (GMRQ) is the sum of the 2nd through 4th
largest eigenvalues of the macrostate transition matrix:

\[
\text{GMRQ} = \sum_{k=2}^{4} \lambda_k
\]

where \(\lambda_1 \geq \lambda_2 \geq \cdots\) are the eigenvalues sorted in
descending order. Higher values indicate that more slow dynamical modes are
preserved in the lumping. GMRQ2 uses the sum of squares:

\[
\text{GMRQ2} = \sum_{k=2}^{4} \lambda_k^2
\]

**Reference:** McGibbon, R. T. & Pande, V. S. (2015). *Variational
cross-validation of slow dynamical modes in molecular kinetics.*
J. Chem. Phys., 142(12), 124105.
[DOI: 10.1063/1.4916292](https://doi.org/10.1063/1.4916292)

---

## RMSD Sharpness

```python
sharpness = mpp.rmsd_sharpness()   # float
```

Population-weighted mean of per-macrostate mean C\(\alpha\) RMSDs:

\[
s = \frac{\sum_j \langle\text{RMSD}\rangle_j \cdot p_j}{\sum_j p_j}
\]

where \(\langle\text{RMSD}\rangle_j\) is the mean C\(\alpha\) RMSD of all frames
in macrostate \(j\) relative to the macrostate mean structure, and \(p_j\) is its
population in frames. Lower values indicate more structurally compact macrostates.
Requires RMSD data (access `mpp.rmsd` or load via `mpp.load_rmsd(path)` first).

---

## Silhouette Coefficient

```python
s = mpp.silhouette   # shape (n_runs,), range [-1, 1]
```

For each frame \(i\), the silhouette value is:

\[
s(i) = \frac{b(i) - a(i)}{\max\bigl(a(i),\, b(i)\bigr)}
\]

where \(a(i)\) is the mean feature distance to all other frames in the same
macrostate, and \(b(i)\) is the mean feature distance to frames in the nearest
other macrostate. The reported metric is the mean over all frames. Values near
+1 indicate well-separated, compact macrostates; values near −1 indicate
misclassified frames. Requires `multi_feature_trajectory` and at least 2
macrostates.

**Reference:** Rousseeuw, P. J. (1987). *Silhouettes: A graphical aid to the
interpretation and validation of cluster analysis.*
J. Comput. Appl. Math., 20, 53–65.
[DOI: 10.1016/0377-0427(87)90125-7](https://doi.org/10.1016/0377-0427(87)90125-7)

---

## Calinski–Harabász Index

```python
ch = mpp.calinski_harabasz   # shape (n_runs,), range [0, ∞)
```

Ratio of between-macrostate dispersion to within-macrostate dispersion,
normalised by degrees of freedom:

\[
\text{CH} = \frac{\mathrm{SS}_B}{\mathrm{SS}_W} \cdot \frac{N - K}{K - 1}
\]

where \(\mathrm{SS}_B\) is the between-cluster sum of squared distances to the
global centroid, \(\mathrm{SS}_W\) is the within-cluster sum of squared distances
to each macrostate centroid, \(N\) is the total number of frames, and \(K\) is
the number of macrostates. Higher values indicate more compact, well-separated
macrostates. Requires `multi_feature_trajectory` and at least 2 macrostates.

**Reference:** Calinski, T. & Harabasz, J. (1974). *A dendrite method for cluster
analysis.* Commun. Stat., 3(1), 1–27.
[DOI: 10.1080/03610927408827101](https://doi.org/10.1080/03610927408827101)
