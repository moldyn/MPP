# Changelog

## v1.1.0 (since v1.0.0)

### New Features

- **`LumpingKernel` seed parameter** (TASK-7.1): `LumpingKernel` now accepts a `seed` argument (integer, passed to `numpy.random.default_rng`) for reproducible stochastic runs. In YAML configs, set `stochastic.seed: <int>`.

- **New quality metrics** (TASK-5.5/5.6/4.5.1-4.5.3): Two new properties on `Lumping`:
  - `Lumping.silhouette` — sklearn silhouette score
  - `Lumping.calinski_harabasz` — sklearn Calinski-Harabasz index

  Also adds a `--metrics` CLI flag that prints all quality metrics as `key=value` pairs.

- **`source` config key is now optional** (fix): The `source` field in YAML configs no longer needs to be set. When present, it is resolved relative to the config file. Previously it was required.

### Bug Fixes

- **Stable sort in `LumpingKernel`**: Switched from unstable `np.argsort` (quicksort) to `kind='stable'` to eliminate non-determinism when states have equal similarity scores. This affected Python 3.13/3.14 and could produce different lumping trees across runs.

- **Sankey diagram scaling**: Gap and label threshold are now relative to dataset size (previously hard-coded).

- **Transition matrix heatmap**: Removed spurious `%` suffix from annotations.

- **Plot font**: All plots now use Latin Modern Roman (via LaTeX rendering, `text.usetex=True`) for consistent typography.

### Potentially Breaking Changes

| Change | Impact |
|---|---|
| **Stable-sort algorithm** — `LumpingKernel` now uses `kind='stable'` argsort. For systems with tied similarity scores, the lumping tree will differ from v1.0.0 results. Saved Z matrices produced by v1.0.0 remain valid for loading, but recomputing from scratch may yield a different tree. | Medium — affects reproducibility comparisons against prior runs |
| **`source` key removed from example/test configs** — Paths are now resolved relative to the config file, not the working directory. Users relying on CWD-relative `source` paths must update their configs. | Low |
| **`sklearn` now required** — `silhouette_score` and `calinski_harabasz_score` require scikit-learn. If not installed, importing `MPP.lumping` will fail. | Low — sklearn is a common dependency but was not previously listed |

### Internal / Non-Breaking

- Python 3.12/3.13/3.14 CI support added; `prettypyplot` added as dependency.
- All tests reclassified as CORE; stochastic tests now use seeded RNG and are fully deterministic.
- Tutorial notebook added (`docs/tutorial/`).
- Documentation expanded: quality metrics page, stochastic workflow guide, plot types reference, implied timescales math.
