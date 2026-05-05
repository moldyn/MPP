# Agent Guidelines — MPP

Guidelines for AI agents (e.g. GitHub Copilot) working on this repository.

---

## Core Rules

1. **NEVER change algorithmic logic without tests.**
   Any change to `core.py`, `MPP.py`, `kernel.py`, or `utils.py` that alters
   numerical output must be accompanied by a test that verifies the new
   behaviour against a known-good reference (stored in `tests/data/`).

2. **ALWAYS update tests when behaviour changes intentionally.**
   If a change is expected to alter outputs, update the expected reference data
   in `tests/data/` and document why in the commit message.

3. **NEVER introduce silent behaviour changes in the CLI or API.**
   Removed or renamed arguments must raise a clear error or keep backward
   compatibility. New required parameters must be gated behind a version bump.

4. **KEEP CLI, API, and Snakemake consistent.**
   A capability added to one interface must be reflected in all three. Parameter
   names must match across `run.py`, `MPP.Lumping`, and `workflow/Snakefile`.

---

## Refactor Rules

5. **Naming changes must be project-wide and consistent.**
   Renaming a function, class, parameter, or config key requires updating every
   call site in `MPP/`, `tests/`, `workflow/`, `docs/`, and `README.md` in the
   same commit or PR. Partial renames are not acceptable.

6. **Large refactors must be split into atomic tasks.**
   A refactor touching more than one module or more than ~150 lines of
   production code must be decomposed into separate tasks (see `TASKS.md`),
   each independently reviewable and testable.

---

## Code Quality Rules

7. **Prefer type hints for all public function signatures.**
   Follow the patterns already established in `core.py` and `MPP.py`
   (`NDArray`, `Literal`, etc. from `numpy.typing` and `typing`).

8. **Avoid deep nesting.**
   Maximum cyclomatic complexity per function is 10. Extract helper functions
   rather than adding another level of `if`/`for`.

9. **No hidden side effects.**
   Functions must not modify global state, write files, or print to stdout
   unless their name or docstring explicitly states this (e.g.
   `save_Z`, `write_least_moving_residues`).

---

## Testing Rules

10. **Every change must pass all existing tests, or update tests intentionally.**
    Run `python -m pytest tests/` (or `bash run_all_tests.sh`) before
    submitting. A failing test that is not related to your change must be
    noted, not silently ignored.

11. **New behaviour requires a new test.**
    A new function, class, or CLI flag must have a corresponding test in the
    appropriate file under `tests/`.

12. **Reference data updates require explicit justification.**
    When expected output files under `tests/data/*/expected_output/` are
    changed, the commit message must state what changed and why the new values
    are correct.

---

## Allowed Scope

| Scope | Definition | Constraints |
|---|---|---|
| **Small** | Single-function or single-file change, ≤ 50 lines of production code | May be done in one task |
| **Medium** | Single-module cleanup (e.g. all of `utils.py`) | Must include tests; document intent |
| **Large** | Cross-module or cross-interface change | Must be split into small/medium tasks in `TASKS.md` before starting |

---

## What Agents Must NOT Do

- Do not invent new features not described in `PRD.md`.
- Do not add new third-party dependencies without updating
  `requirements.txt`, `environment.yml`, and `pyproject.toml`.
- Do not modify `tests/data/` reference files without a matching test update.
- Do not reformat unrelated code in the same commit as a logic change.
- Do not merge failing tests.
