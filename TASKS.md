# Task List — MPP

Prioritized, atomic tasks for agentic development. Each task is independently
executable. Complete tasks in phase order unless stated otherwise.

---

## Phase 1 — Stabilisation

Goal: make the test suite runnable and identify every broken area.

### TASK-1.1 — Audit test dependencies
- Check that every import in `tests/` (`msmhelper`, `mdtraj`, `pygpcca`,
  `anytree`, `tqdm`, etc.) is listed in `requirements.txt` and
  `environment.yml`.
- Add any missing entry; do not change versions unless a conflict exists.

### TASK-1.2 — Verify test data completeness
- Confirm that `tests/data/HP35/`, `tests/data/PDZ3/`, and `tests/data/aSyn/`
  each contain `config.yaml`, `config_stochastic.yaml` (where applicable),
  and all expected output files referenced in `test_run.py`.
- Create a plain-text manifest (`tests/data/MANIFEST.txt`) listing every
  required file per dataset.

### TASK-1.2.5 — Classify tests before fixing failures
- Classify the full test suite into three groups:
  - CORE
  - OPTIONAL
  - DEFERRED
- Record the classification before starting any failure-fixing tasks.

### TASK-1.3 — Fix broken test collection
- Run `python -m pytest tests/ --collect-only` and resolve any import errors
  or missing fixtures so that all test cases are collected without error.
- Do not skip tests; fix the root cause.

### TASK-1.4 — Fix failing unit tests in `test_properties.py` and `test_utils.py`
- Run `python -m pytest tests/test_properties.py tests/test_utils.py -v`.
- For each failure: fix the production code if it is a bug, or update the
  expected value with justification if the test expectation is wrong.

### TASK-1.5 — Fix failing unit tests in `test_plots.py` and `test_rmsd.py`
- Run `python -m pytest tests/test_plots.py tests/test_rmsd.py -v`.
- Fix each failure following the same policy as TASK-1.4.

### TASK-1.6 — Fix failing integration tests in `test_run.py`
- Run `python -m pytest tests/test_run.py -v`.
- Fix each failure; do not mark tests as expected failures unless the feature
  is genuinely broken and tracked by a separate issue.

---

## Phase 1.5 — Scientific Baseline Definition

Goal: define what "correct behaviour" means before refactoring.

### TASK-1.7 — Capture reference outputs
- Run full pipeline on example datasets
- Store:
  - Z matrices
  - macrostate assignments
- Save under `tests/data/*/baseline/`

### TASK-1.8 — Add regression tests
- Add tests that compare current outputs against baseline
- Use tolerances where necessary (floating point)

---

## Phase 1.6 — Code Hygiene & Formatting

Goal: clean the codebase without changing behaviour.

### TASK-1.9 — Remove commented-out code
- Scan all files in `MPP/`, `workflow/`, and `tests/`
- Remove:
  - commented-out functions
  - commented-out code blocks
- Keep only meaningful explanatory comments

### TASK-1.10 — Remove dead code
- Identify unused:
  - functions
  - imports
  - variables
- Remove them after verifying they are not used anywhere

### TASK-1.11 — Enforce line length and formatting
- Wrap lines > 100 characters:
  - code
  - docstrings
- Ensure consistent indentation and spacing

### TASK-1.12 — Standardise docstring formatting
- Convert all docstrings to NumPy style
- Ensure:
  - Parameters section exists where needed
  - No overly long lines
  - Consistent formatting across modules

---

## Phase 2 — Naming Refactor

Goal: establish consistent naming conventions and apply them project-wide.

### TASK-2.1 — Document naming conventions
- Create `docs/naming_conventions.md` with the agreed rules:
  - Functions: `snake_case`.
  - Classes: `PascalCase`.
  - Modules: `snake_case`.
  - Config YAML keys: `snake_case` with words separated by underscores
    (e.g. `pop_thr`, not `pop thr` — migrate space-separated keys).
  - Public constants: `UPPER_SNAKE_CASE`.
- List current violations found in the codebase.

### TASK-2.2 — Rename space-separated YAML config keys
- In `run.py` (`OPTIONAL_PARAMS`, `Data.__init__`) replace all space-separated
  config keys (`"microstate trajectory"`, `"multi feature trajectory"`,
  `"cluster file"`, etc.) with underscore-separated equivalents.
- Update `workflow/mpp.yml` and all config files under `example/` and
  `tests/data/`.
- Keep backward compatibility by accepting both forms with a deprecation
  warning for one release cycle, then remove the old form.

### TASK-2.3 — Standardise kernel parameter names
- `LumpingKernel` and `FeatureKernel` use `similarity`, `method`, `param`.
  Verify these match the CLI flags `d` and `g` conceptually; rename if not.
- Update all call sites in `run.py`, `MPP.py`, and tests.

### TASK-2.4 — Rename internal `dij` / `gij` to descriptive names
- `dij` → `dynamic_similarity` (or `kernel_similarity`).
- `gij` → `feature_similarity` (or `feature_kernel`).
- Apply consistently in `run.py`, `MPP.py`, `workflow/Snakefile`, and tests.

---

## Phase 3 — Documentation

Goal: make every public symbol self-documenting.

### TASK-3.1 — Add NumPy-style docstrings to `core.py`
- `BinaryTreeNode.__init__` already has a partial docstring; complete it.
- Add docstrings to every property and method that lacks one.
- Conform to NumPy docstring style (Parameters / Returns / Raises sections).

### TASK-3.2 — Add NumPy-style docstrings to `MPP.py` (`Lumping`)
- The class docstring exists; add method-level docstrings for every public
  method that is missing one.

### TASK-3.3 — Add docstrings to `kernel.py` and `utils.py`
- Every public function and class must have a docstring with at minimum a one-
  line summary and a Parameters section.

### TASK-3.4 — Add docstrings to `plot.py`
- Each plot function must document its parameters, especially `out` (output
  path) and any optional kwargs.

### TASK-3.5 — Expand `docs/` with usage examples
- Add `docs/usage_cli.md` with worked CLI examples matching the `example/`
  directory.
- Add `docs/usage_api.md` showing how to use `MPP.Lumping` directly.
- Add `docs/usage_snakemake.md` summarising the Snakemake workflow.

---

## Phase 3.5 — Documentation Sanitisation

Goal: ensure documentation is accurate and consistent.

### TASK-3.6 — Remove outdated or incorrect docstrings
- Identify docstrings that contradict actual behaviour
- Fix or remove them

### TASK-3.7 — Align terminology with PRD
- Ensure consistent use of:
  - microstate
  - macrostate
  - kernel
  - lumping

---

## Phase 4 — Interface Consistency

Goal: CLI, API, and Snakemake must expose the same capabilities with the same
parameter names.

### TASK-4.1 — Audit CLI vs API parity
- List every option in `run.py` `parse_args()` and verify it is accessible
  via `MPP.Lumping` or `MPP.run.Data` without going through the CLI.
- For each gap: add the missing parameter to the API.

### TASK-4.2 — Audit Snakemake vs CLI parity
- List every `rule` in `workflow/Snakefile` and verify the shell command it
  invokes matches the current CLI argument names.
- Fix any stale flag names resulting from prior renames.

### TASK-4.3 — Unify `rmsd_feature` handling
- In `run.py`, `--rmsd-feature` defaults to `"CA"` and is set on
  `data.mpp.rmsd_feature`.
- Ensure `Lumping` exposes `rmsd_feature` as a settable attribute with the
  same default and validates the value (`"CA"` or `"feature"` only).

### TASK-4.4 — Validate plot kind names between CLI and `plot` module
- The `plot()` function in `run.py` and the `Lumping.plot` namespace must
  support exactly the same set of kind strings.
- Add a test that calls `plot()` with an unknown kind and asserts `ValueError`.

---

## Phase 4.5 — New Metrics Implementation

Goal: implement planned quality metrics.

### TASK-4.5.1 — Implement Silhouette Coefficient
- Use `sklearn.metrics.silhouette_score`
- Ensure compatibility with macrostate assignments

### TASK-4.5.2 — Implement Calinski–Harabasz index
- Use `sklearn.metrics.calinski_harabasz_score`

### TASK-4.5.3 — Add tests for both metrics
- Validate against known small datasets

### TASK-4.5.4 — Expose metrics in CLI/API/Snakemake
- Add CLI flags
- Add API accessors

---

## Phase 5 — UI Polish

Goal: improve CLI usability and error reporting.

### TASK-5.1 — Improve `d` and `g` argument help text
- Replace `"dij to be used."` and `"gij to be used."` with text that lists
  the accepted values and their meaning, e.g.:
  `"Dynamic similarity kernel. One of: T (transition probability), KL (Kullback-Leibler), none, gpcca."`.

### TASK-5.2 — Add validation of `d` / `g` combinations in `parse_args`
- Raise `argparse.ArgumentTypeError` for invalid combinations (e.g. `gpcca`
  with `JS`) before any file I/O occurs.
- Add a test that passes an invalid combination and asserts a non-zero exit
  code with a descriptive message.

### TASK-5.3 — Improve file-not-found error messages
- In `Data.__init__`, when `np.loadtxt` fails because a file does not exist,
  catch the `OSError` and re-raise with the full resolved path and which config
  key referred to it.

### TASK-5.4 — Add `--version` flag to CLI
- Read the version from `pyproject.toml` (or `MPP.__version__` if defined)
  and expose it via `parser.add_argument("--version", action="version")`.
- Add a test that invokes `python -m MPP.run --version` and checks the output
  is non-empty.

### TASK-5.5 — Add group link to docs site
- Add an `extra.social` block in `mkdocs.yml` with a globe icon linking to
  `https://www.moldyn.uni-freiburg.de/` and label "Molecular Dynamics Group Freiburg".
- Add a brief "Developed by" line in `docs/index.md` crediting the group.
- Files: `mkdocs.yml`, `docs/index.md`

### TASK-5.6 — Add `--metrics` flag to CLI
- Add `--metrics` (boolean flag, `store_true`) to `parse_args()` in `src/MPP/run.py`.
- In `main()`, after macrostate assignment, if `--metrics` is set, compute and print
  all available metrics to stdout as `key=value` pairs:
  `shannon_entropy`, `davies_bouldin`, `gmrq`, `gmrq2`.
  Include `silhouette` and `calinski_harabasz` once TASK-4.5.1–4.5.2 are done.
- Update CLI help in README and `docs/usage_cli.md`.
- Add a test that invokes the CLI with `--metrics` and checks the output keys.
- Files: `src/MPP/run.py`, `docs/usage_cli.md`, `README.md`, `tests/test_run.py`

---

## Phase 3 (Continued) — Documentation Additions

### TASK-3.8 — Document all quality metrics
- Add a "Quality Metrics" section to `docs/usage_api.md` covering:
  Shannon entropy, Davies-Bouldin, GMRQ, GMRQ2, RMSD sharpness, Silhouette,
  Calinski-Harabász. For each: brief explanation, property name, example usage.
- Update `docs/usage_cli.md` to describe `--metrics` output format.
- Files: `docs/usage_api.md`, `docs/usage_cli.md`

### TASK-3.9 — Document all plot types with examples
- Add `docs/plots.md` documenting each of the 15 plot types:
  name, what it shows, minimal CLI invocation, minimal Python API call,
  required inputs (note which require XTC/topology).
- Add `plots.md` to `mkdocs.yml` nav.
- Files: `docs/plots.md`, `mkdocs.yml`

---

## Phase 6 — RMSD Sanitisation

Goal: clean up RMSD code and promote RMSD tests.

### TASK-6.1 — RMSD code and API audit
- Review `src/MPP/lumping.py` (RMSD property, `calc_rmsd*` calls) and
  `src/MPP/utils.py` (`calc_rmsd`, `calc_rmsd_feature`, helpers) for:
  - API inconsistencies (e.g., `rmsd_feature` attr vs. `--rmsd-feature` CLI flag)
  - Dead or commented-out code
- Fix identified issues without changing behaviour.
- Files: `src/MPP/lumping.py`, `src/MPP/utils.py`

### TASK-6.2 — Promote RMSD tests to CORE
- Verify `TestRMSD_PDZ3::test_rmsd_property` and `test_rmsd_sharpness` pass with
  existing test data (no XTC required for these).
- Move them from OPTIONAL to CORE in `tests/TEST_CATEGORIES.md` and update `CLAUDE.md`.
- Files: `tests/TEST_CATEGORIES.md`, `CLAUDE.md`

---

## Phase 7 — Stochastic Workflow Sanitisation

Goal: make stochastic tests deterministic and well-documented.

### TASK-7.1 — Fix stochastic test infrastructure with seeded RNG
- Add a `seed` parameter to `LumpingKernel.__init__` that seeds
  `numpy.random.default_rng` for reproducible stochastic runs.
- Propagate `seed` through `Lumping.run_mpp()` and `Data` / CLI (via YAML
  `stochastic.seed` key or a `--seed` CLI flag).
- Generate reference Z matrices using a fixed seed and save to
  `tests/data/HP35/expected_output/t_stochastic/Z_stochastic.npy` (and aSyn equivalent).
- Update stochastic tests to pass the fixed seed and compare against reference.
- Move `test_HP35_t_stoch` and `test_aSyn_t_stoch` from DEFERRED to CORE.
- Files: `src/MPP/kernel.py`, `src/MPP/lumping.py`, `src/MPP/run.py`,
  `tests/test_run.py`, `tests/TEST_CATEGORIES.md`, `CLAUDE.md`

### TASK-7.2 — Document stochastic workflow
- Add a "Stochastic Lumping" section to `docs/usage_api.md` and `docs/usage_cli.md`.
- Cover: `stochastic` YAML block, `method`/`param`/`n`/`seed` options, output format
  (Z shape `(n_runs, n_states-1, 4)`), seeded reproducibility.
- Files: `docs/usage_api.md`, `docs/usage_cli.md`

---

## Phase 8 — Test Coverage

Goal: close remaining coverage gaps after phases 6–7.

### TASK-8.1 — Test coverage audit and gap fill
- After TASK-6.2 and TASK-7.1, re-classify all remaining OPTIONAL/DEFERRED tests.
- Identify public functions in `src/MPP/` with zero test coverage.
- Add targeted unit tests for the identified gaps.
- Files: `tests/`, `tests/TEST_CATEGORIES.md`

---

## Phase 9 — CI Python Version Matrix

### TASK-9.1 — Expand CI to Python 3.13 and 3.14
- Add `3.13` and `3.14` to the `python-version` matrix in the test workflow.
- Update `pyproject.toml` classifiers to list `3.12`, `3.13`, `3.14` as supported.
- Update `docs/index.md` tested-version note.
- Files: `.github/workflows/<test-workflow>.yml`, `pyproject.toml`, `docs/index.md`

---

## Phase 10 — Tutorials

### TASK-10.1 — End-to-end tutorial
- Create `tutorials/tutorial_basic.ipynb`: walk-through using `example/sample_system`.
  Covers: loading trajectory, building kernels, running MPP, assigning macrostates,
  generating plots — shown in both CLI and Python API form.
- Create `docs/tutorial.md` as a rendered markdown version (code blocks, no execution).
- Add both to `mkdocs.yml` nav.
- Files: `tutorials/tutorial_basic.ipynb`, `docs/tutorial.md`, `mkdocs.yml`

---

## Phase 11 — Scientific Documentation (Deferred)

Goal: mathematical documentation of the MPP algorithm. To be done after publication.

### TASK-11.1 — Scientific algorithm documentation
- Add `docs/algorithm.md` with:
  - Metastability criterion and merge selection (step 1)
  - Macrostate assignment / tree parsing (step 2, pop_thr, q_min)
  - Z matrix format and scipy compatibility
  - Placeholder for publication reference
- Add to `mkdocs.yml` nav.
- Files: `docs/algorithm.md`, `mkdocs.yml`
