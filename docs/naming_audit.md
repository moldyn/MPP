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
| `workflow/lumpings.yml`, `tests/data/lumpings.yml` | `kernel similarity`    | `kernel_similarity` ✅ TASK-2.3 | HIGH-RISK |
| `workflow/lumpings.yml`, `tests/data/lumpings.yml` | `feature kernel`       | `feature_kernel` ✅ TASK-2.3   | HIGH-RISK |

**Note:** YAML key renames require simultaneous updates to every file that reads
those keys (`MPP/run.py`, `workflow/Snakefile`, all config files, and tests).
Treat the entire group as a single atomic rename per key.

---

## 2. `dij` / `gij` Internal Parameter Names ✅ DONE (TASK-2.3)

Mathematical index notation leaks from internal documentation into production code.

| File              | Location                   | Old name | New name               | Risk     |
|-------------------|----------------------------|----------|------------------------|----------|
| `MPP/run.py`      | `Data._prepare_kernels`    | `dij`    | `dynamic_similarity`   | CAUTION  |
| `MPP/run.py`      | `Data._prepare_kernels`    | `gij`    | `feature_similarity`   | CAUTION  |
| `MPP/run.py`      | `Data.setup_mpp`           | `dij`    | `dynamic_similarity`   | CAUTION  |
| `MPP/run.py`      | `Data.setup_mpp`           | `gij`    | `feature_similarity`   | CAUTION  |

Also renamed `Data.prepare_mpp` → `Data._prepare_kernels` (made private).
CLI-facing `d` and `g` positional args stay as-is; help text updated.

---

## 3. `n_i` — Ambiguous Public Property ✅ DONE (TASK-2.3)

`Lumping.n_i` is a public property used throughout the codebase as "the index
of the current lumping run under consideration."

| File            | Location            | Old name | New name      | Risk     |
|-----------------|---------------------|----------|---------------|----------|
| `MPP/MPP.py`    | `Lumping.run_index` | `n_i`    | `run_index`   | HIGH-RISK |

Renamed to `run_index` with `n_i` kept as a deprecated alias emitting
`DeprecationWarning`. Updated all internal uses in `MPP/MPP.py`, `MPP/plot.py`,
`MPP/utils.py`, and `tests/test_utils.py`.

---

## 4. `MPP/MPP.py` — Module Name Violates `snake_case` ✅ DONE (cleanup/src-layout-and-module-rename)

| File          | Current name | Proposed name  | Risk     |
|---------------|--------------|----------------|----------|
| `src/MPP/MPP.py`  | `MPP.py` | `lumping.py`   | HIGH-RISK |

The module filename `MPP.py` matched the package directory `MPP/`, which was
confusing and violated `snake_case`. Renamed to `lumping.py`. The package was
simultaneously migrated to a `src/` layout. `__init__.py` updated:
`from .MPP import Lumping` → `from .lumping import Lumping`.

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

## 10. `lumpings.yaml` Value `"ref"` — Ambiguous Sentinel ✅ DONE (TASK-2.3)

| File                         | Key              | Current value       | Proposed value     | Risk     |
|------------------------------|------------------|---------------------|--------------------|----------|
| `workflow/lumpings.yml`      | `feature_kernel` | `"reference_count"` | *(already done)*   | CAUTION  |
| `tests/data/lumpings.yml`    | `feature_kernel` | `"reference_count"` | *(already done)*   | CAUTION  |

Both files already use `"reference_count"` and the snake_case key `feature_kernel`.
The `perform_gpcca` default was updated to `n_macrostates="reference_count"` in
TASK-2.3. Integer parsing for the CLI `g` argument was added in TASK-4.2.

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

1. Space-separated YAML keys: `microstate trajectory`, `multi feature trajectory` — ✅ done in TASK-2.2; `kernel similarity`, `feature kernel` — ✅ done in TASK-2.3
2. `Lumping.n_i` property — ✅ renamed to `run_index` in TASK-2.3 (deprecated alias retained)
3. Module `MPP/MPP.py` (affects all imports) — ✅ renamed to `src/MPP/lumping.py`, `src/` layout adopted

### CAUTION (rename in a coordinated single-module task)

4. YAML keys: `contact threshold`, `cluster file`, `contact index file`, `topology file`, `xtc file`, `frame length`, `xtc stride`, `n timescales` — ✅ done in TASK-2.2
5. `Data.prepare_mpp(dij, gij)` / `Data.setup_mpp(dij, gij)` parameter names — ✅ renamed to `dynamic_similarity`/`feature_similarity` in TASK-2.3
6. `Data.prepare_mpp` method name — ✅ renamed to `_prepare_kernels` in TASK-2.3
7. GPCCA `"ref"` sentinel value in lumpings YAML — ✅ done in TASK-2.3 / TASK-4.2

### SAFE (rename within the containing function/method, no callers affected)

8. `iter` (loop variable) in `Lumping.run_mpp`, `Lumping.assign_macrostates` — ✅ done in TASK-2.3
9. `map` (parameter) in `utils.translate_trajectory` — ✅ done in TASK-2.3
10. `t` (local variable) in `LumpingKernel.__call__` — ✅ done in TASK-2.3
11. `ms`, `other_ms` in `utils.macrostate_tmat` — ✅ done in TASK-2.3
12. `gma`, `gmt`, `gmf` in `Lumping._assign_macrostates_from_gpcca` — ✅ done in TASK-2.3
13. `multi_state_trajectory_raw` in `Data.__init__` — ✅ done in TASK-2.3

---

## Verification

Run the following to confirm no functional changes were made:

```bash
python -m pytest tests/ --collect-only -q
```

No test should fail as a result of adding these documentation files.
