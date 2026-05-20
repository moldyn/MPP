# Naming Audit — MPP

Inventory of naming violations identified across the MPP repository.
See `docs/naming_conventions.md` for the authoritative naming standard.

Each violation records: file path, current name, proposed replacement, and
risk classification (SAFE / CAUTION / HIGH-RISK).

---

## Risk Classification

| Class       | Definition                                                         |
|-------------|--------------------------------------------------------------------|
| SAFE        | Local variables, private helpers — no external consumers           |
| CAUTION     | Public methods, class attributes, config keys — has callers/users  |
| HIGH-RISK   | Core API symbols, serialized names, CLI-visible args, workflow-facing params |

---

## 1. YAML Config Key Violations (space-separated keys)

All YAML config keys should be `snake_case`. The following use spaces.

| File                                          | Current key                    | Proposed key               | Risk     |
|-----------------------------------------------|--------------------------------|----------------------------|----------|
| `tests/data/HP35/input/config.yml` (and all configs) | `microstate trajectory`   | `microstate_trajectory`    | HIGH-RISK |
| all configs                                   | `multi feature trajectory`     | `multi_feature_trajectory` | HIGH-RISK |
| all configs                                   | `contact threshold`            | `contact_threshold`        | CAUTION  |
| all configs                                   | `cluster file`                 | `cluster_file`             | CAUTION  |
| all configs                                   | `contact index file`           | `contact_index_file`       | CAUTION  |
| all configs                                   | `topology file`                | `topology_file`            | CAUTION  |
| all configs                                   | `xtc file`                     | `xtc_file`                 | CAUTION  |
| all configs                                   | `frame length`                 | `frame_length`             | CAUTION  |
| all configs                                   | `xtc stride`                   | `xtc_stride`               | CAUTION  |
| `MPP/run.py:227`                              | `"n timescales"`               | `n_timescales`             | CAUTION  |
| `workflow/lumpings.yml`, `tests/data/lumpings.yaml` | `kernel similarity`    | `kernel_similarity`        | HIGH-RISK |
| `workflow/lumpings.yml`, `tests/data/lumpings.yaml` | `feature kernel`       | `feature_kernel`           | HIGH-RISK |

**Note:** YAML key renames require simultaneous updates to every file that reads
those keys (`MPP/run.py`, `workflow/Snakefile`, all config files, and tests).
Treat the entire group as a single atomic rename per key.

---

## 2. `dij` / `gij` Internal Parameter Names

Mathematical index notation leaks from internal documentation into production code.

| File              | Location            | Current name | Proposed name          | Risk     |
|-------------------|---------------------|--------------|------------------------|----------|
| `MPP/run.py:88`   | `Data.prepare_mpp`  | `dij`        | `dynamic_similarity`   | CAUTION  |
| `MPP/run.py:88`   | `Data.prepare_mpp`  | `gij`        | `feature_similarity`   | CAUTION  |
| `MPP/run.py:116`  | `Data.setup_mpp`    | `dij`        | `dynamic_similarity`   | CAUTION  |
| `MPP/run.py:116`  | `Data.setup_mpp`    | `gij`        | `feature_similarity`   | CAUTION  |

These are method parameters (not public API attributes), so the risk is contained,
but they propagate the ambiguous `dij`/`gij` terminology through the call chain.
The CLI-facing `d` and `g` positional args (HIGH-RISK) are excluded here — they
stay as-is at the CLI boundary.

---

## 3. `n_i` — Ambiguous Public Property

`Lumping.n_i` is a public property used throughout the codebase as "the index
of the current lumping run under consideration."

| File            | Location            | Current name | Proposed name | Risk     |
|-----------------|---------------------|--------------|---------------|----------|
| `MPP/MPP.py:806`| `Lumping.n_i`       | `n_i`        | `run_index`   | HIGH-RISK |

`n_i` is referenced in: `MPP/MPP.py`, `MPP/run.py`, `MPP/plot.py`, and tests.
A rename must be project-wide. The name `n_i` is also an attribute on `Data`
(implicitly via `self.mpp.n_i`) and referenced in docstrings.

---

## 4. `MPP/MPP.py` — Module Name Violates `snake_case`

| File          | Current name | Proposed name  | Risk     |
|---------------|--------------|----------------|----------|
| `MPP/MPP.py`  | `MPP.py`     | `lumping.py`   | HIGH-RISK |

The module filename `MPP.py` matches the package directory `MPP/`, which is
confusing and violates `snake_case`. The `Lumping` class (the main content) would
be better housed in `lumping.py`. All imports (`from .MPP import Lumping`,
`from . import MPP`) and the `__init__.py` export must change together.

---

## 5. `multi_state_trajectory_raw` — Inconsistent Naming

Two names are used for the same raw contact-distance array in `run.py`.

| File           | Line | Current name                  | Proposed name                   | Risk   |
|----------------|------|-------------------------------|---------------------------------|--------|
| `MPP/run.py:41`| 41   | `multi_state_trajectory_raw`  | `multi_feature_trajectory_raw`  | SAFE   |
| `MPP/run.py:56`| 56   | `multi_feature_trajectory`    | *(already correct)*             | —      |

`multi_state_trajectory_raw` appears only in `Data.__init__` and is an instance
attribute, but it is also passed to `Lumping` via `setup_mpp`. The inconsistency
with the everywhere-else-used `multi_feature_trajectory` naming suggests this was
an oversight. Risk is SAFE because `multi_state_trajectory_raw` is only accessed
as `data.multi_state_trajectory_raw` in `run.py:122`.

---

## 6. GPCCA Internal Variable Abbreviations

Short abbreviations used in `Lumping._assign_macrostates_from_gpcca`.

| File            | Line | Current name | Proposed name        | Risk |
|-----------------|------|--------------|----------------------|------|
| `MPP/MPP.py:414`| 414  | `gma`        | `gpcca_assignment`   | SAFE |
| `MPP/MPP.py:415`| 415  | `gmt`        | `gpcca_trajectory`   | SAFE |
| `MPP/MPP.py:416`| 416  | `gmf`        | `gpcca_feature`      | SAFE |

All three are local variables inside a private method. SAFE to rename without
affecting any external interface.

---

## 7. Shadowed Python Built-ins

| File              | Line | Name  | Shadows    | Proposed name         | Risk |
|-------------------|------|-------|------------|-----------------------|------|
| `MPP/MPP.py:310`  | 310  | `iter`| `iter()`   | `run_iter`            | SAFE |
| `MPP/MPP.py:313`  | 313  | `iter`| `iter()`   | `run_iter`            | SAFE |
| `MPP/MPP.py:338`  | 338  | `iter`| `iter()`   | `run_iter`            | SAFE |
| `MPP/utils.py:22` | 22   | `map` | `map()`    | `state_map`           | SAFE |

`iter` shadows the built-in in `run_mpp` and `assign_macrostates` (used only as
a local loop variable). `map` shadows the built-in in `translate_trajectory`
(parameter name). Both are SAFE to rename — no callers pass these by name.

---

## 8. `t` — Local Variable Shadowing Concept

| File              | Line | Current name | Proposed name | Risk |
|-------------------|------|--------------|---------------|------|
| `MPP/kernel.py:98`| 98   | `t`          | `masked_tmat` | SAFE |

`t` is a local variable holding a masked copy of the transition matrix, used in
the KL-divergence branch. It is easily confused with the `T` similarity mode and
with `tmat`. Renaming to `masked_tmat` clarifies intent.

---

## 9. `ms` — Ambiguous Loop Variable

| File            | Line      | Current name | Proposed name        | Risk |
|-----------------|-----------|--------------|----------------------|------|
| `MPP/utils.py:58`| 58, 59   | `ms`         | `macrostate_mask`    | SAFE |
| `MPP/utils.py:58`| 59       | `other_ms`   | `other_macrostate_mask` | SAFE |

`ms` is used as a boolean mask slice for microstates belonging to a macrostate
inside `macrostate_tmat`. The abbreviated name does not convey whether it is an
index, a mask, or a list.

---

## 10. `lumpings.yaml` Value `"ref"` — Ambiguous Sentinel

| File                         | Key              | Current value | Proposed value     | Risk     |
|------------------------------|------------------|---------------|--------------------|----------|
| `workflow/lumpings.yml:18`   | `feature kernel` | `"ref"`       | `"reference_count"` | CAUTION  |
| `tests/data/lumpings.yaml:18`| `feature kernel` | `"ref"`       | `"reference_count"` | CAUTION  |

For the GPCCA entry, `feature kernel: ref` is a special sentinel meaning "use the
macrostate count from the reference (T) lumping." This is not obvious and is
distinct from the `JS`/`none` values used by other entries. A more descriptive
sentinel such as `"reference_count"` would clarify intent.

**Note:** This value is parsed in `MPP/run.py:174` as the `n_macrostates="ref"`
default. Renaming requires updating the parser, config files, and documentation
together.

---

## 11. `setup_mpp` vs `prepare_mpp` — Method Name Clarity

| File          | Current name    | Proposed name       | Risk    |
|---------------|-----------------|---------------------|---------|
| `MPP/run.py:88`  | `prepare_mpp`| `_prepare_kernels`  | CAUTION |
| `MPP/run.py:116` | `setup_mpp`  | `setup_mpp`         | CAUTION |

`prepare_mpp` (creates kernels) and `setup_mpp` (creates `Lumping` + calls
`prepare_mpp`) have overlapping names that do not clearly indicate their
relationship. `prepare_mpp` is only called by `setup_mpp` — it could be renamed
to `_prepare_kernels` and made private. No external callers exist in the
current codebase.

---

## 12. `BinaryTreeNode` Missing `tmat` Parameter in Docstring

| File              | Line | Issue                                             | Risk |
|-------------------|------|---------------------------------------------------|------|
| `MPP/core.py:29`  | 29   | `tmat` not documented in `__init__` Parameters   | SAFE |

The `BinaryTreeNode.__init__` docstring does not list `tmat` as a parameter even
though it is required and stored as `self.tmat`. This is a documentation gap, not
a naming violation, but noted here for completeness.

---

## Summary by Risk Class

### HIGH-RISK (do not rename without a dedicated task and full cross-repo update)

1. All space-separated YAML keys: `microstate trajectory`, `multi feature trajectory`, `kernel similarity`, `feature kernel`
2. `Lumping.n_i` property (used across MPP/, tests/, plot.py)
3. Module `MPP/MPP.py` (affects all imports)

### CAUTION (rename in a coordinated single-module task)

4. YAML keys: `contact threshold`, `cluster file`, `contact index file`, `topology file`, `xtc file`, `frame length`, `xtc stride`, `n timescales`
5. `Data.prepare_mpp(dij, gij)` / `Data.setup_mpp(dij, gij)` parameter names
6. `Data.prepare_mpp` method name (→ `_prepare_kernels`)
7. GPCCA `"ref"` sentinel value in lumpings YAML

### SAFE (rename within the containing function/method, no callers affected)

8. `iter` (loop variable) in `Lumping.run_mpp`, `Lumping.assign_macrostates`
9. `map` (parameter) in `utils.translate_trajectory`
10. `t` (local variable) in `LumpingKernel.__call__`
11. `ms`, `other_ms` in `utils.macrostate_tmat`
12. `gma`, `gmt`, `gmf` in `Lumping._assign_macrostates_from_gpcca`
13. `multi_state_trajectory_raw` in `Data.__init__`

---

## Verification

Run the following to confirm no functional changes were made:

```bash
python -m pytest tests/ --collect-only -q
```

No test should fail as a result of adding these documentation files.
