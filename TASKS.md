# Task List — MPP

Prioritized, atomic tasks for agentic development. Each task is independently
executable.

Completed work (phases 1–7, 9–11) is recorded in `CHANGELOG.md`.

---

## Phase 8 — Test Coverage

Goal: close remaining coverage gaps.

### TASK-8.1 — Test coverage audit and gap fill
- Overall branch coverage: **85%** (as of last run; see `tests/TEST_CATEGORIES.md`).
- Known gaps:
  - `random_to_png.py` — 0% (requires PyMol binary; deferred)
  - `sankey_gap.py` — 71% (edge-case layout branches)
  - `kernel.py` — 82% (stochastic `"p"`-method not tested)
  - `lumping.py` — 84% (guards and error paths)
  - `run.py` — 86% (config-loading error branches)
- Add targeted unit tests for each identified gap (except PyMol).
- Files: `tests/`, `tests/TEST_CATEGORIES.md`

---

## Phase 12 — Interface Consistency Fixes

Goal: resolve behavioral inconsistencies identified in the interface consistency
audit (documented inline below).

### TASK-12.1 — Fix `contact_threshold` ignored by `FeatureKernel` in `Data`
- **Severity:** HIGH
- `Data.__init__` binarizes the feature trajectory at a hardcoded `0.45`; the
  `contact_threshold` config value is never applied to `FeatureKernel` input.
- Fix: derive threshold from `self.d["contact_threshold"]` (fallback `0.45`).
- Files: `src/MPP/run.py`

### TASK-12.2 — Fix `frame_length` default overridden to `None` by `Data`
- **Severity:** HIGH
- `Data.setup_mpp` overwrites `Lumping.frame_length` with `None` when the key
  is absent from the YAML config, shadowing the `0.2` default in `Lumping.__init__`.
- Fix: fall back to `0.2` in `Data.__init__` when `frame_length` is absent.
- Files: `src/MPP/run.py`

### TASK-12.3 — Fix Snakemake `plot_all` structural plot handling
- **Severity:** MEDIUM
- `plot_all` unconditionally requests `contacts.pdf`, `rmsd.pdf`, `delta_rmsd.pdf`
  (fail without topology/XTC) and is missing 7 plot types available in CLI/API:
  `state_network`, `transition_matrix`, `transition_time`, `macrostate_trajectory`,
  `macro_feature`, `stochastic_state_similarity`, `relative_implied_timescales`.
- Fix: split into `plot_all_core` (always-safe) and `plot_all_structural`; add the
  three deterministic-safe missing types to `plot_all_core`.
- Files: `workflow/Snakefile`

### TASK-12.4 — Make `data_root` configurable in Snakefile
- **Severity:** MEDIUM
- Replace `data_root = "example"` with `config.get("data_root", "example")`.
- Files: `workflow/Snakefile`

### TASK-12.5 — Document stochastic workflows in Snakemake
- **Severity:** MEDIUM
- Add note in `docs/usage_snakemake.md` explaining how to trigger stochastic
  lumpings via the system config `stochastic` block.
- Consider adding example entries in `workflow/lumpings.yml`.
- Files: `docs/usage_snakemake.md`, `workflow/lumpings.yml`

### TASK-12.6 — Add `--scale` and `--n-timescales` CLI flags
- **Severity:** LOW
- `--scale` (default `1`) passed through to all plot methods.
- `--n-timescales` overrides the `n_timescales` config value per invocation.
- Files: `src/MPP/run.py`, `docs/usage_cli.md`, `README.md`

### TASK-12.7 — Document and parse `g` integer for GPCCA mode
- **Severity:** LOW
- When `d = "gpcca"`, the `g` argument is interpreted as `n_macrostates` but
  help text only mentions `JS`/`none`. Integer strings are not parsed.
- Fix: update `g` help text; add integer parsing in `main()` when `d == "gpcca"`.
- Files: `src/MPP/run.py`, `docs/usage_cli.md`
