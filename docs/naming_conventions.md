# Naming Conventions — MPP

This document defines the authoritative naming standard for the MPP repository.
All Phase 2 refactors must follow these conventions.

---

## 1. Conventions by Entity Type

| Entity Type        | Convention          | Example                          |
|--------------------|---------------------|----------------------------------|
| Functions          | `snake_case`        | `run_mpp`, `assign_macrostates`  |
| Classes            | `PascalCase`        | `Lumping`, `LumpingKernel`       |
| Modules            | `snake_case`        | `core.py`, `utils.py`            |
| Variables          | `snake_case`        | `n_states`, `pop_thr`            |
| Constants          | `UPPER_SNAKE_CASE`  | `OPTIONAL_PARAMS`, `DEFAULTS`    |
| YAML config keys   | `snake_case`        | `microstate_trajectory`, `q_min` |
| CLI flags (long)   | `kebab-case`        | `--rmsd-feature`, `--draw-random`|
| CLI positional     | short lowercase     | `d`, `g` (see note below)        |

### Notes

- **Module rename completed:** `MPP/MPP.py` was renamed to `src/MPP/lumping.py`
  and the package was migrated to a `src/` layout (cleanup/src-layout-and-module-rename).
- **CLI positional args `d` / `g`:** These are established abbreviations in the
  existing CLI interface. They are clarified in help text and documentation
  but are not renamed.
- **Scientific abbreviations** in function/variable names are permitted when they
  are defined in this document's terminology policy (Section 3).

---

## 2. File and Directory Naming

| Scope          | Convention     | Example                      |
|----------------|----------------|------------------------------|
| Python modules | `snake_case`   | `kernel.py`, `run.py`        |
| YAML configs   | `snake_case`   | `config.yml`, `lumpings.yml` |
| Test files     | `test_<name>`  | `test_run.py`                |
| Docs files     | `snake_case`   | `naming_conventions.md`      |

---

## 3. Terminology Policy

### Preferred Terms

| Concept                 | Preferred Term               | Notes                                        |
|-------------------------|------------------------------|----------------------------------------------|
| Discrete MD state       | **microstate**               | Use throughout; not "state" alone if ambiguous|
| Coarse-grained state    | **macrostate**               | Use throughout                               |
| The coarse-graining     | **lumping**                  | Not "clustering" for the MPP algorithm       |
| Dynamic similarity fn.  | **lumping kernel**           | The `LumpingKernel` class                    |
| Geometric similarity fn.| **feature kernel**           | The `FeatureKernel` class                    |
| Coarse-graining step    | **merge** / **merging**      | Not "cluster" or "aggregate"                 |
| Similarity matrix       | **transition matrix**        | Abbreviated `tmat` in code                   |
| Self-transition prob.   | **metastability**            | Symbol `q`; not "stability" or "persistence" |
| Linkage tree output     | **Z matrix**                 | scipy-compatible linkage format              |
| Population threshold    | **pop_thr**                  | Parameter name; not `min_pop` or `p_min`     |
| Min. metastability      | **q_min**                    | Parameter name; not `min_q` or `q_threshold` |

### Accepted Abbreviations

| Abbreviation | Expansion                          | Permitted in         |
|--------------|------------------------------------|----------------------|
| `tmat`       | transition matrix                  | code variables       |
| `pop`        | population (count)                 | code variables       |
| `q`          | metastability (self-transition)    | code variables, docs |
| `n_states`   | number of microstates              | code variables       |
| `n_macrostates` | number of macrostates           | code variables       |
| `T`          | transition-probability similarity  | CLI arg, config, docs|
| `KL`         | Kullback-Leibler divergence        | CLI arg, config, docs|
| `JS`         | Jensen-Shannon divergence          | CLI arg, config, docs|
| `Z`          | Z matrix (linkage)                 | CLI arg, code, docs  |
| `d`          | dynamic similarity selector (CLI)  | CLI positional arg   |
| `g`          | geometric similarity selector (CLI)| CLI positional arg   |
| `CA`         | C-alpha atoms                      | code, config, docs   |
| `RMSD`       | root-mean-square deviation         | code, config, docs   |
| `GPCCA`      | Generalized PCCA (proper noun)     | code, config, docs   |

### Discouraged Abbreviations

| Abbreviation | Problem                              | Preferred Replacement   |
|--------------|--------------------------------------|-------------------------|
| `dij`        | Math-index notation leaking into code| `dynamic_similarity` or drop — use `d` at CLI boundary only |
| `gij`        | Math-index notation leaking into code| `feature_similarity` or drop — use `g` at CLI boundary only |
| `n_i`        | Ambiguous: "run index" not obvious   | `run_index` (CAUTION rename) |
| `t`          | Shadows meaning when near `tmat`     | Use `masked_tmat` for local |
| `ms`         | Used for "macrostate slice"          | `macrostate_mask`        |
| `gma`        | GPCCA macrostate assignment          | `gpcca_assignment`       |
| `gmt`        | GPCCA macrostate trajectory          | `gpcca_trajectory`       |
| `gmf`        | GPCCA macrostate feature             | `gpcca_feature`          |
| `m`          | macrostate row in loops              | context-dependent        |

### Legacy / Internal Names

These names appear in the codebase but should not be introduced in new code:

| Name                       | Status  | Notes                                       |
|----------------------------|---------|---------------------------------------------|
| `multi_state_trajectory_raw` | legacy  | Use `multi_feature_trajectory_raw` consistently |
| `full_tmat`                | internal | Acceptable as local; documents 2n-1 matrix |
| `full_pop`                 | internal | Acceptable as local/attribute               |
| `linkage`                  | legacy   | Derived from Z; prefer Z-based operations  |

---

## 4. YAML Config Key Standard

All YAML config keys use `snake_case`. The migration from space-separated keys
was completed in TASK-2.2. Legacy space-separated forms are still accepted with
a `DeprecationWarning` for one release cycle.

### Canonical YAML keys (all migrated)

| Canonical key                | Deprecated (space-separated) form |
|------------------------------|-----------------------------------|
| `microstate_trajectory`      | `microstate trajectory`           |
| `multi_feature_trajectory`   | `multi feature trajectory`        |
| `contact_threshold`          | `contact threshold`               |
| `cluster_file`               | `cluster file`                    |
| `contact_index_file`         | `contact index file`              |
| `topology_file`              | `topology file`                   |
| `xtc_file`                   | `xtc file`                        |
| `frame_length`               | `frame length`                    |
| `xtc_stride`                 | `xtc stride`                      |
| `n_timescales`               | `n timescales`                    |
| `kernel_similarity`          | `kernel similarity`               |
| `feature_kernel`             | `feature kernel`                  |

Always-conforming keys: `lagtime`, `pop_thr`, `q_min`, `source`, `stochastic`,
`method`, `param`, `n`, `seed`.

---

## 5. Cross-Interface Consistency Rule

Parameter names must be identical across CLI, Python API, and Snakemake workflow.
If a concept is named differently in any two interfaces, that is a violation.

Current cross-interface mapping:

| Concept             | CLI arg         | Python API param       | YAML key              | Snakemake param |
|---------------------|-----------------|------------------------|-----------------------|-----------------|
| Dynamic kernel      | `d`             | `similarity`           | `kernel_similarity`   | `params.d`      |
| Feature kernel      | `g`             | `feature_kernel`       | `feature_kernel`      | `params.g`      |
| Population threshold| *(via config)*  | `pop_thr`              | `pop_thr`             | *(via config)*  |
| Min. metastability  | *(via config)*  | `q_min`                | `q_min`               | *(via config)*  |
| Trajectory stride   | *(via config)*  | `xtc_stride`           | `xtc_stride`          | *(via config)*  |

The `d`/`g` CLI abbreviations map to `kernel_similarity`/`feature_kernel` in
YAML and `similarity`/`feature_kernel` in the Python API. These should be
documented consistently but are not renamed in this phase.
