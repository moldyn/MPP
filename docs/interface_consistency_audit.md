# Interface Consistency Audit — MPP

Audit of behavioral consistency across the three MPP user interfaces:
CLI (`MPP/run.py`), Python API (`MPP/MPP.py`, `MPP/kernel.py`), and
Snakemake workflow (`workflow/Snakefile`, `workflow/lumpings.yml`).

Scope: deterministic workflows. Stochastic, GPCCA, RMSD extras, and PDB
generation are noted where they interact with deterministic behavior but are
not the primary focus.

---

## 1. Supported Deterministic Workflow Inventory

### Kernel combinations

| Lumping key | `d` (kernel_similarity) | `g` (feature_kernel) | CLI | API | Snakemake |
|-------------|------------------------|----------------------|-----|-----|-----------|
| `t`         | `T`                    | `none`               | yes | yes | yes       |
| `kl`        | `KL`                   | `none`               | yes | yes | yes       |
| `t_js`      | `T`                    | `JS`                 | yes | yes | yes       |
| `kl_js`     | `KL`                   | `JS`                 | yes | yes | yes       |
| `js`        | `none`                 | `JS`                 | yes | yes | yes       |
| `gpcca`     | `gpcca`                | `reference_count`    | yes | yes | yes       |

All six kernel combinations are supported across all three interfaces.
No workflow gaps exist at this level.

### Plot types

| Plot type                      | CLI (`-p`) | API (`mpp.plot.*`) | Snakemake `plot_all` |
|-------------------------------|------------|-------------------|----------------------|
| `dendrogram`                  | yes        | yes               | yes                  |
| `timescales`                  | yes        | yes               | yes                  |
| `sankey`                      | yes        | yes               | yes                  |
| `contacts`                    | yes        | yes               | yes (requires `cluster_file`) |
| `macrotraj`                   | yes        | yes               | yes                  |
| `ck_test`                     | yes        | yes               | yes                  |
| `rmsd`                        | yes        | yes               | yes (requires topology/XTC) |
| `delta_rmsd`                  | yes        | yes               | yes (requires topology/XTC) |
| `state_network`               | yes        | yes               | no                   |
| `transition_matrix`           | yes        | yes               | no                   |
| `transition_time`             | yes        | yes               | no                   |
| `macro_feature`               | yes        | yes               | no                   |
| `macrostate_trajectory` (txt) | yes        | yes               | no                   |
| `stochastic_state_similarity` | yes        | yes               | no                   |
| `relative_implied_timescales` | yes        | yes               | no                   |

---

## 2. Identified Inconsistencies

---

### IC-01 — `contact_threshold` not applied to FeatureKernel in `Data`

**Severity:** HIGH

**Files:** `MPP/run.py:108`, `MPP/run.py:155–158`

**Description:**
`Data.__init__` binarizes the feature trajectory at a hardcoded threshold of
`0.45` for the `multi_feature_trajectory` attribute:

```python
self.multi_feature_trajectory = self.multi_feature_trajectory_raw < 0.45
```

This pre-binarized array is then passed to `FeatureKernel` in
`Data._prepare_kernels`. The `contact_threshold` value from the YAML config
is passed to `Lumping.__init__` (via `setup_mpp`) and applied internally
there — but it is never applied to the FeatureKernel input.

**Effect:**
When a user sets `contact_threshold` to any value other than `0.45` in the
config, the `FeatureKernel` (used for JS similarity) still binarizes at
`0.45`, while `Lumping` applies the configured threshold internally. When
`contact_threshold: null` is set (disabling binarization in `Lumping`), the
`FeatureKernel` still binarizes at `0.45`.

**Affected interfaces:** CLI, Snakemake (both go through `Data`). Direct API
usage is not affected as users control both kernel and `Lumping` inputs
directly.

**Harmonization strategy:**
`Data` should derive `multi_feature_trajectory` using `self.d["contact_threshold"]`
(with fallback to `0.45` when not `None`), not a hardcoded literal.

---

### IC-02 — `frame_length` default overridden to `None` by `Data`

**Severity:** HIGH

**Files:** `MPP/run.py:127`, `MPP/run.py:186`

**Description:**
`Lumping.__init__` has `frame_length=0.2` as its default. When using the
Python API directly, any `Lumping` instance that does not receive a
`frame_length` argument correctly defaults to `0.2` ns per frame.

When using the CLI or Snakemake (both go through `Data`), `setup_mpp`
constructs `Lumping` without `frame_length` (so it initializes to `0.2`),
then immediately overwrites it:

```python
self.mpp.frame_length = self.frame_length
```

`Data.frame_length` comes from `self.d["frame_length"]`, which defaults to
`None` when the key is absent from the YAML config. This sets
`mpp.frame_length = None`, silently overriding the `0.2` default established
in `Lumping.__init__`.

**Effect:**
Any plot that depends on `frame_length` (e.g., `timescales`, `ck_test`,
`transition_time`) will raise a `TypeError` when called without `frame_length`
in the config, because `None` is used in arithmetic. Direct API usage is
unaffected — `frame_length` stays `0.2`.

**Affected interfaces:** CLI, Snakemake.

**Harmonization strategy:**
`Data.__init__` should fall back to `0.2` when `frame_length` is absent from
the config, matching the `Lumping` default. Alternatively, `Lumping.frame_length`
setter should reject `None`.

---

### IC-03 — `plot_all` includes structurally-dependent plots unconditionally

**Severity:** MEDIUM

**Files:** `workflow/Snakefile:295–316`

**Description:**
The Snakemake `plot_all` rule unconditionally requests `contacts.pdf`,
`rmsd.pdf`, and `delta_rmsd.pdf`. These plots require optional files
(`cluster_file`, `topology_file`, `xtc_file`) in the system config. If those
files are absent, the rule will fail.

In the CLI, these plots are only generated when explicitly requested with
`-p contacts`, `-p rmsd`, or `-p delta_rmsd`, so failure occurs only on
explicit intent.

**Effect:**
A user who runs `plot_all` on a system without structural files (the common
case for pure MSM analysis) will receive a workflow error. The CLI does not
have this problem.

**Affected interfaces:** Snakemake.

**Harmonization strategy:**
Either split `plot_all` into `plot_all_core` (always-safe plots) and
`plot_all_structural` (requires topology/XTC), or make structural plots
conditional on the presence of the relevant config keys.

---

### IC-04 — Seven plot types available in CLI/API absent from Snakemake `plot_all`

**Severity:** MEDIUM

**Files:** `workflow/Snakefile:295–316`, `MPP/run.py:255–318`

**Description:**
The following plot types are implemented and exposed via the CLI (`-p`) and
API (`mpp.plot.*`) but are not collected by the Snakemake `plot_all` rule:

- `state_network`
- `transition_matrix`
- `transition_time`
- `macrostate_trajectory` (txt output)
- `macro_feature`
- `stochastic_state_similarity`
- `relative_implied_timescales`

The Snakemake `plot` rule does support generating any of these individually
(since the `{plot}` wildcard is unconstrained). The gap is only in `plot_all`.

**Affected interfaces:** Snakemake.

**Harmonization strategy:**
The three always-safe deterministic plots (`state_network`, `transition_matrix`,
`transition_time`) should be added to `plot_all`. The `macrostate_trajectory`
text output could be added with the `txt` extension, which the `plot` rule's
wildcard constraint already permits. Stochastic-only plots (`stochastic_state_similarity`,
`relative_implied_timescales`) may be deferred.

---

### IC-05 — `data_root` hardcoded in Snakefile, not configurable

**Severity:** MEDIUM

**Files:** `workflow/Snakefile:4`

**Description:**
`data_root = "example"` is set as a Python literal at the top of the
Snakefile. There is no mechanism to override this from the Snakemake command
line (e.g., via `--config data_root=<path>`) without editing the file directly.

The CLI and API have no such constraint — users specify paths directly.

**Affected interfaces:** Snakemake.

**Harmonization strategy:**
Replace the hardcoded assignment with a Snakemake `config` lookup with a
default fallback:

```python
data_root = config.get("data_root", "example")
```

This would allow `snakemake --config data_root=mydata ...` without file edits.

---

### IC-06 — Stochastic workflows not represented in Snakemake `lumpings.yml`

**Severity:** MEDIUM

**Files:** `workflow/lumpings.yml`, `MPP/run.py:141–147`, `MPP/run.py:207`

**Description:**
The CLI and `Data` support stochastic lumpings via a `stochastic` block in
the YAML config:

```yaml
stochastic:
  method: n
  param: 10
  n: 100
```

This is handled in `Data._prepare_kernels` and `Data.perform_mpp`. The
stochastic run count (`n`) and method are read directly from the config.

However, there is no stochastic lumping entry in `workflow/lumpings.yml`,
no Snakemake rule variant for stochastic runs, and no mention in any usage
guide. Stochastic support exists only implicitly via the system config.

**Affected interfaces:** Snakemake, documentation.

**Harmonization strategy:**
Add a note in `usage_snakemake.md` explaining how to trigger stochastic
lumpings. Consider adding example entries in `lumpings.yml`. No Snakemake rule
changes are strictly necessary since the stochastic config is embedded in the
system config, not in the lumping key.

---

### IC-07 — `plot scale` parameter not exposed as a CLI flag

**Severity:** LOW

**Files:** `MPP/run.py:255`, `MPP/run.py:440`

**Description:**
The `run.plot()` function accepts a `scale` parameter (default `1`) passed to
all plot methods. The API exposes `scale` on `mpp.plot.dendrogram()`,
`mpp.plot.sankey()`, etc. The CLI `parse_args()` does not expose a `--scale`
flag, so scale is always `1` when invoked from the CLI or Snakemake.

**Affected interfaces:** CLI, Snakemake.

**Harmonization strategy:**
Add an optional `--scale` flag to `parse_args()` and pass it through to
`run.plot()`.

---

### IC-08 — `n_timescales` not configurable as a CLI flag

**Severity:** LOW

**Files:** `MPP/run.py:278–280`, `MPP/MPP.py:914`

**Description:**
The number of implied timescales to compute is configurable via `n_timescales`
in the YAML config for the CLI/Snakemake path. In the API, it is controlled
by `mpp.calc_timescales(ntimescales=N)` directly. There is no `--n-timescales`
CLI flag.

The asymmetry is minor since the YAML config is always required for CLI use,
but it means timescale count cannot be varied per-invocation without editing
the config file.

**Affected interfaces:** CLI, Snakemake (minor).

**Harmonization strategy:**
Add an optional `--n-timescales` CLI flag that overrides the config value.
Low priority.

---

### IC-09 — `gpcca` `g` argument not validated or documented for CLI

**Severity:** LOW

**Files:** `MPP/run.py:366–373`, `MPP/run.py:428–431`

**Description:**
When `d = "gpcca"`, the CLI `g` argument is interpreted as `n_macrostates`
in `perform_gpcca()` rather than as a feature kernel selector. The `g`
argument help text describes only `JS` and `none` as valid values; it does
not mention the `gpcca`-specific interpretation (`reference_count` or an
integer).

`perform_gpcca()` handles `n_macrostates == "reference_count"` correctly but
does not parse integer strings. Passing `g = "5"` would send the string `"5"`
to `Lumping.gpcca()` which expects an `int`.

**Affected interfaces:** CLI.

**Harmonization strategy:**
Update the `g` argument help text to document gpcca-mode behavior. Add
integer parsing in `perform_gpcca` or in `main()` when `d == "gpcca"`.

---

### IC-10 — `naming_audit.md` has stale entries for items already completed

**Severity:** LOW

**Files:** `docs/naming_audit.md`, `workflow/lumpings.yml`, `tests/data/lumpings.yml`

**Description:**
`docs/naming_audit.md` item 10 lists the GPCCA `feature_kernel` sentinel
value `"ref"` as "still pending" (proposed replacement: `"reference_count"`).
Both `workflow/lumpings.yml` and `tests/data/lumpings.yml` already use
`"reference_count"`. The audit document is stale.

Similarly, item 10 shows `feature kernel` (space-separated) as the current
key name, but both files already use `feature_kernel` (snake_case), which
was completed in TASK-2.3.

**Affected interfaces:** documentation.

**Harmonization strategy:**
Mark item 10 in `naming_audit.md` as completed (TASK-2.3 / TASK-3.x), and
update the "current value" column to reflect the actual state.

---

### IC-11 — `lumpings.yaml` / `lumpings.yml` inconsistent file extension

**Severity:** LOW

**Files:** `workflow/lumpings.yml`, `tests/data/lumpings.yml`

**Description:**
The workflow uses `lumpings.yml` while the test fixture uses `lumpings.yaml`.
Both are syntactically valid YAML. The inconsistency is cosmetic but violates
the convention stated in `docs/naming_conventions.md` section 2 (YAML configs:
`snake_case`). The standard does not specify `.yml` vs `.yaml`, but consistency
across the repo is preferable.

**Affected interfaces:** documentation, tests.

**Harmonization strategy:**
Standardize on `.yml` (matching the workflow convention and the majority of
config files). Update `tests/data/lumpings.yml` → `tests/data/lumpings.yml`
and all references. Low breakage risk — only test code reads the test fixture.

---

## 3. Summary Table

| ID    | Description                                                      | Severity | Interfaces Affected       |
|-------|------------------------------------------------------------------|----------|---------------------------|
| IC-01 | `contact_threshold` ignored by FeatureKernel in `Data`          | HIGH     | CLI, Snakemake            |
| IC-02 | `frame_length` default overridden to `None` by `Data`           | HIGH     | CLI, Snakemake            |
| IC-03 | `plot_all` unconditionally includes structurally-dependent plots | MEDIUM   | Snakemake                 |
| IC-04 | Seven plot types missing from Snakemake `plot_all`               | MEDIUM   | Snakemake                 |
| IC-05 | `data_root` hardcoded in Snakefile                               | MEDIUM   | Snakemake                 |
| IC-06 | Stochastic workflows absent from Snakemake `lumpings.yml`        | MEDIUM   | Snakemake, docs           |
| IC-07 | `scale` not exposed as a CLI flag                                | LOW      | CLI, Snakemake            |
| IC-08 | `n_timescales` not configurable as a CLI flag                   | LOW      | CLI, Snakemake            |
| IC-09 | `g` arg undocumented for GPCCA mode; integer not parsed          | LOW      | CLI                       |
| IC-10 | `naming_audit.md` has stale entries for completed renames        | LOW      | docs                      |
| IC-11 | `lumpings.yaml` vs `lumpings.yml` inconsistent extension         | LOW      | docs, tests               |

---

## 4. Harmonization Priorities

### Immediate (HIGH)

**IC-01** and **IC-02** are behavioral bugs affecting the CLI and Snakemake
paths for any non-default `contact_threshold` and for any invocation without
`frame_length` in config respectively. Both should be fixed before any broader
interface harmonization work.

### Short-term (MEDIUM)

**IC-03** (structural plots in `plot_all`) and **IC-04** (missing plots in
`plot_all`) can be addressed together as a single Snakemake rule update.
Split `plot_all` into core and structural targets, and add the three
deterministic-safe missing plot types.

**IC-05** (`data_root` hardcoded) is a one-line Snakefile change with no
backward-compatibility concerns.

**IC-06** (stochastic in Snakemake) requires only documentation; no code
changes needed.

### Deferred (LOW)

**IC-07** through **IC-11** are minor polish items. **IC-07** and **IC-08**
add CLI flags for existing functionality. **IC-09** adds input validation.
**IC-10** and **IC-11** are documentation-only corrections.

---

## 5. Backward-Compatibility Concerns

| ID    | Concern |
|-------|---------|
| IC-01 | Fixing `contact_threshold` propagation will change FeatureKernel behavior for users with non-default thresholds. Deterministic Z matrices will differ. All existing baselines assume default `0.45`; no baseline impact if the fix applies only to non-default configs. |
| IC-02 | Fixing `frame_length` default will change timing-related plot output (axis scales, units) for users without `frame_length` in config. No impact on Z matrices or macrostate assignments. |
| IC-03 | Splitting `plot_all` is backward-compatible if the original target name is kept as an alias. |
| IC-05 | Adding `config.get("data_root", "example")` is fully backward-compatible. |
| IC-11 | Renaming `lumpings.yaml` → `lumpings.yml` requires updating all references; low risk since the file is only used in tests. |

---

## 6. Verification

Deterministic regression tests run before and after audit preparation:

```
tests/test_properties.py        PASSED
tests/test_utils.py             PASSED
tests/test_run.py::test_HP35_t_ref   PASSED
tests/test_run.py::test_HP35_kl      PASSED
tests/test_run.py::test_HP35_t_js    PASSED
tests/test_run.py::test_HP35_js      PASSED
tests/test_run.py::test_PDZ3_kl      PASSED
tests/test_run.py::test_aSyn_t       PASSED
tests/test_run.py::test_aSyn_kl_js   PASSED

15 passed, 16 warnings in 16.45s
```

No baseline outputs changed. No interface changes were made during this audit.
