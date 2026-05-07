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
