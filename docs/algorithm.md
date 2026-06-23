# The MPP Algorithm

MPP (Most Probable Path) is a hierarchical lumping algorithm for Markov state models.
It reduces a large set of microstates to a compact set of macrostates by iteratively
merging the least metastable microstate with its most dynamically (and optionally
geometrically) similar neighbor, then reading off macrostates from the resulting
binary tree.

---

## Overview

Given a microstate trajectory, MPP proceeds in two stages:

1. **Build the lumping tree** (`core.cluster`): iteratively merge microstates
   bottom-up, recording every merge in a Z matrix (scipy linkage format).
2. **Assign macrostates** (`Lumping.assign_macrostates`): parse the binary tree
   top-down to identify nodes that satisfy user-defined metastability and population
   thresholds.

---

## Stage 1 — Building the lumping tree

### Transition matrix

From the microstate trajectory \(\{s_t\}\) a row-stochastic transition matrix
\(\mathbf{T}(\tau)\) is estimated at lag time \(\tau\):

\[
T_{ij} = \frac{c_{ij}}{\sum_k c_{ik}}, \qquad
c_{ij} = \#\{t : s_t = i,\; s_{t+\tau} = j\}
\]

The diagonal entry \(T_{ii}\) is the **self-transition probability**, also called
the **metastability** \(q_i\) of microstate \(i\). High \(q_i\) means the system
tends to stay in state \(i\); low \(q_i\) means the state is transient and a good
candidate for merging.

### Iterative merging

**Step 1 — Select the least metastable active state \(a\).**
Among all currently active states \(\mathcal{A}\), pick the one with the smallest
self-transition probability:

\[
a = \arg\min_{i \in \mathcal{A}} T_{ii}
\]

**Step 2 — Select the merge target \(b\).**
Find the active state most similar to \(a\) according to the chosen
[similarity metric](#similarity-metrics).

**Step 3 — Merge \(a\) into \(b\).**
Replace the two states with a single lumped state. Its outgoing transition row and
incoming transition column are population-weighted averages of the originals:

\[
T_{\text{new}, j} = \frac{n_a\, T_{aj} + n_b\, T_{bj}}{n_a + n_b}, \qquad
T_{i, \text{new}} = T_{ia} + T_{ib}
\]

**Step 4 — Record the merge in the Z matrix.**
Append one row encoding the two merged states, the metastability of \(a\) at
the time of merging, and the total population of the resulting lump:

\[
Z_i = \bigl[a,\; b,\; T_{aa},\; n_a + n_b\bigr]
\]

Steps 1–4 repeat until all \(n\) microstates have been merged into one root state,
producing \(n-1\) rows in \(\mathbf{Z}\).

### Similarity metrics

The **target state** \(b\) is chosen based on a configurable similarity function.
The three available dynamic metrics, alone or combined with a geometric term, are:

| `d` | Metric | Description |
|---|---|---|
| `T` | Transition probability | \(b = \arg\max_{j \neq a} T_{aj}\) — merge with the state that \(a\) most frequently transitions into |
| `KL` | Kullback-Leibler divergence | \(b = \arg\min_{j \neq a} D_\mathrm{KL}(T_a \| T_j)\) — merge with the state whose outgoing distribution most closely matches \(a\)'s |
| `none` | Feature-only | Dynamic similarity is uniform; only geometric similarity (`JS`) determines the target |

The **KL divergence** between two transition rows is:

\[
D_\mathrm{KL}(T_a \| T_j) = \sum_k T_{ak} \ln \frac{T_{ak}}{T_{jk}}
\]

A small value means states \(a\) and \(j\) visit the same neighbors with similar
probabilities, making them good candidates for merging.

Before selection, the KL scores are transformed by the weighting function
\(w(x) = e^{-x^2 / \sigma^2}\) to convert divergences into similarity weights.

### Geometric similarity (Jensen-Shannon, `JS`)

When a **feature trajectory** is provided, an optional geometric term is multiplied
into the dynamic similarity weights. The feature kernel computes the
Jensen-Shannon divergence between the per-state mean feature distributions:

\[
D_\mathrm{JS}(p \| q) = \frac{1}{2} D_\mathrm{KL}\!\left(p \,\Big\|\, \frac{p+q}{2}\right)
                       + \frac{1}{2} D_\mathrm{KL}\!\left(q \,\Big\|\, \frac{p+q}{2}\right)
\]

States with similar structural features (e.g. fraction of native contacts, dihedral
angles) receive higher similarity weights. The JS divergence is squared and passed
through the same weighting function before being multiplied with the dynamic
weights.

When states are merged, the feature kernel updates the population-weighted mean
feature of the resulting lumped state so that subsequent steps use the correct
aggregate feature vector.

---

## Stage 2 — Assigning macrostates

The Z matrix encodes a binary tree: each row \(i\) describes the merge of two
children into an intermediate node with index \(n + i\). Stage 2 parses this tree
top-down to find macrostates.

### When is a node a macrostate?

A node \(v\) in the lumping tree is classified as a **macrostate** if and only if
all of the following hold:

- Its parent merge happened at metastability \(q \geq q_\text{min}\) — the
  split point is dynamically meaningful.
- \(v\) itself carries at least \(p_\text{thr}\) of the total trajectory frames.
- \(v\)'s sibling also carries at least \(p_\text{thr}\) of the total frames.

The **root** is always a macrostate (fallback when no split satisfies the criteria).

Intuitively: a macrostate boundary is drawn at a merge that combines two
well-populated groups where one of them was highly metastable just before being
absorbed. The threshold \(q_\text{min}\) controls how sharp the kinetic boundary
must be; \(p_\text{thr}\) prevents macrostates with negligible population.

### Assigning microstates to macrostates

After macrostate nodes are identified, every leaf (microstate) is assigned:

- If the microstate is itself a macrostate node, it is assigned to itself.
- If the nearest ancestor macrostate has only one terminal macrostate in its
  subtree, the microstate is assigned there.
- Otherwise the microstate is assigned to the macrostate with the highest
  transition probability from that microstate (computed by temporarily merging
  each candidate macrostate's microstates with the unassigned state).

---

## Stochastic lumping

By default the algorithm is **deterministic** (`method="n"`, `param=1`): at each
step only the single most similar neighbor is considered as the merge target.

Two stochastic modes are available:

| Mode | Parameters | Description |
|---|---|---|
| Top-N | `method="n"`, `param=N` | Randomly sample the target from the `N` most similar states, weighted by their similarity scores |
| Probability-mass | `method="p"`, `param=p` | Include as many top candidates as needed so that their combined similarity mass exceeds `p`, then sample |

A `seed` can be provided to the `LumpingKernel` for reproducible results. Running
many stochastic lumpings and comparing their Z matrices or macrostate assignments
allows uncertainty quantification.

---

## Z matrix format

The Z matrix has shape `(n_runs, n_states-1, 4)` for stochastic runs (or
`(1, n_states-1, 4)` for a single deterministic run). Each row encodes one merge:

| Column | Symbol | Meaning |
|---|---|---|
| 0 | \(a\) | Index of the absorbed state (least metastable) |
| 1 | \(b\) | Index of the target state |
| 2 | \(q_a\) | Self-transition probability of \(a\) at time of merging |
| 3 | \(n_{ab}\) | Joint population of the merged state |

Intermediate cluster indices start at `n_states` and increment by one per merge,
matching the scipy `linkage` convention. The Z matrix is the only artifact that
needs to be stored; all downstream results (macrostate assignment, quality metrics,
plots) are derived from it.

---

## Complexity

| Operation | Cost |
|---|---|
| Transition matrix estimation | \(O(N)\) in trajectory length \(N\) |
| Lumping tree construction | \(O(n^2)\) in number of microstates \(n\) |
| Macrostate assignment | \(O(n)\) |

The bottleneck for large microstate counts is the \(O(n^2)\) search for the least
metastable state and its most similar neighbor at each of the \(n-1\) merge steps.

---

## References

The MPP algorithm implemented in this package:

- *Manuscript in preparation* — citation will be added upon publication.

Background and earlier version of the algorithm:

- Nagel, W. E. *et al.* (2023). *Most Probable Path — A method for lumping Markov
  state models.* Journal of Chemical Theory and Computation.
  [DOI: 10.1021/acs.jctc.3c00488](https://doi.org/10.1021/acs.jctc.3c00488)
- Jain, A. & Stock, G. (2012). *Identifying Metastable States of Folding Proteins.*
  Journal of Chemical Theory and Computation, 8(10), 3810–3819.
  [DOI: 10.1021/ct300077q](https://doi.org/10.1021/ct300077q)

Background on Markov state models:

- Pande, V. S., Beauchamp, K. & Bowman, G. R. (2010). *Everything you wanted to
  know about Markov State Models but were afraid to ask.* Methods, 52(1), 99–105.
  [DOI: 10.1016/j.ymeth.2010.06.002](https://doi.org/10.1016/j.ymeth.2010.06.002)
- Szabo, B. *et al.* (2004). *Network analysis of protein dynamics.* FEBS Letters,
  579(11), 2323–2329.
  [DOI: 10.1016/j.febslet.2005.03.052](https://doi.org/10.1016/j.febslet.2005.03.052)
