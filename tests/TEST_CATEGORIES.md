# Test Suite Categories

Based on `python -m pytest tests/ --collect-only -q` (39 tests collected at initial classification; new tests added since).

## Classification summary

- **CORE**: 29 tests (includes stochastic, metrics, CLI-metrics tests)
- **OPTIONAL**: 6 tests
- **DEFERRED**: 4 tests

## Per-test classification

### `tests/test_properties.py`

| Test case | Category | Rationale |
|---|---|---|
| `test_properties.TestProperties::test_shannon_entropy` | CORE | Deterministic lumping quality metric on core HP35 T workflow. |
| `test_properties.TestProperties::test_gmrq` | CORE | Deterministic lumping quality metric in core pipeline. |
| `test_properties.TestProperties::test_davies_bouldin_index` | CORE | Deterministic macrostate quality metric tied to core reproducibility. |
| `test_properties.TestProperties::test_silhouette` | CORE | Silhouette coefficient regression test (TASK-4.5.1). |
| `test_properties.TestProperties::test_calinski_harabasz` | CORE | Calinski-Harabász index regression test (TASK-4.5.2). |
| `test_properties.TestProperties::test_silhouette_single_macrostate` | CORE | Guard: ValueError when only 1 macrostate exists. |
| `test_properties.TestProperties::test_calinski_harabasz_single_macrostate` | CORE | Guard: ValueError when only 1 macrostate exists. |

### `tests/test_utils.py`

| Test case | Category | Rationale |
|---|---|---|
| `test_utils.TestProperties::test_Z_to_linkage` | CORE | Validates deterministic Z-matrix to linkage conversion. |
| `test_utils.TestProperties::test_linkage_to_Z` | CORE | Validates deterministic macrostate assignment reconstruction from linkage. |
| `test_utils.TestProperties::test_calc_full_tmat` | CORE | Validates deterministic transition matrix/population reconstruction. |
| `test_utils.TestProperties::test_Z_to_mask` | CORE | Validates deterministic macrostate mask generation from Z. |
| `test_utils.TestFullFeature::test_full_feature_from_Z` | CORE | Validates deterministic feature-kernel transformation from Z. |

### `tests/test_run.py`

| Test case | Category | Rationale |
|---|---|---|
| `TestRunScript::test_HP35_t_ref` | CORE | CLI/API deterministic lumping run (T kernel). |
| `TestRunScript::test_HP35_t_stoch` | CORE | Stochastic lumping with seeded RNG — reproducible, regression-testable. |
| `TestRunScript::test_HP35_kl` | CORE | CLI/API deterministic lumping run (KL kernel). |
| `TestRunScript::test_HP35_t_js` | CORE | CLI/API deterministic lumping run (T + JS kernels). |
| `TestRunScript::test_HP35_js` | CORE | CLI/API deterministic lumping run (JS feature-kernel path). |
| `TestRunScript::test_HP35_gpcca` | DEFERRED | GPCCA output path is explicitly lower priority. |
| `TestRunScript::test_PDZ3_kl` | CORE | Deterministic KL run on PDZ3 dataset; reproducibility-critical. |
| `TestRunScript::test_aSyn_t` | CORE | Deterministic T run on aSyn dataset; reproducibility-critical. |
| `TestRunScript::test_aSyn_kl_js` | CORE | Deterministic mixed-kernel run in core pipeline. |
| `TestRunScript::test_aSyn_t_stoch` | CORE | Stochastic lumping with seeded RNG — reproducible, regression-testable. |
| `TestRunScript::test_random_frames_indices_aSyn_t_ref` | DEFERRED | Random frame generation is explicitly lower priority. |
| `TestRunScript::test_macrostate_map_saved_alongside_z` | CORE | Verifies macrostate_map.npy is written alongside Z.npy. |
| `TestCLIValidation::test_invalid_d_exits_with_error` | CORE | CLI argument validation. |
| `TestCLIValidation::test_invalid_g_exits_with_error` | CORE | CLI argument validation. |
| `TestCLIValidation::test_missing_z_for_mpp_exits_with_error` | CORE | CLI argument validation. |
| `TestCLIValidation::test_plot_without_out_exits_with_error` | CORE | CLI argument validation. |
| `TestCLIValidation::test_metrics_flag_prints_all_keys` | CORE | --metrics flag prints all quality metric keys (TASK-5.6). |

### `tests/test_rmsd.py`

| Test case | Category | Rationale |
|---|---|---|
| `TestRMSD_HP35::test_draw_random_indices` | DEFERRED | Random frame index generation is explicitly deferred. |
| `TestRMSD_HP35::test_draw_random_frames` | DEFERRED | Random frame export / pdb-related output is deferred. |
| `TestRMSD_aSyn::test_write_least_moving_residues` | DEFERRED | Least-moving-residue export is explicitly deferred. |
| `TestRMSD_PDZ3::test_rmsd_property` | OPTIONAL | Requires XTC+topology for PDZ3 (not in test data); cannot be CORE. |
| `TestRMSD_PDZ3::test_rmsd_sharpness` | OPTIONAL | Requires PDZ3 rmsd.npy from RMSD computation; cannot be CORE without XTC. |

### `tests/test_plots.py`

| Test case | Category | Rationale |
|---|---|---|
| `TestPlotting::test_manual_ck_test` | CORE | Deterministic diagnostic plotting in core analysis path. |
| `TestPlotting::test_manual_contacts` | CORE | Deterministic core plot output coverage. |
| `TestPlotting::test_manual_delta_rmsd` | OPTIONAL | RMSD-related plotting is lower-priority optional scope. |
| `TestPlotting::test_manual_dendrogram` | CORE | Deterministic dendrogram plotting is explicitly core. |
| `TestPlotting::test_manual_macro_feature` | DEFERRED | Stochastic-only plotting output. |
| `TestPlotting::test_manual_macrotraj_PDZ3` | CORE | Deterministic macrostate trajectory plotting for core datasets. |
| `TestPlotting::test_manual_macrotraj_ref` | CORE | Deterministic macrostate trajectory plotting in core path. |
| `TestPlotting::test_manual_relative_implied_timescales` | DEFERRED | Stochastic-analysis plotting output. |
| `TestPlotting::test_manual_rmsd` | OPTIONAL | RMSD plotting is lower-priority optional scope. |
| `TestPlotting::test_manual_sankey` | CORE | Deterministic sankey plotting in core deterministic analysis. |
| `TestPlotting::test_manual_state_network` | CORE | Deterministic state-network plotting in core deterministic analysis. |
| `TestPlotting::test_manual_stochastic_state_similarity` | DEFERRED | Stochastic-analysis plotting output. |
| `TestPlotting::test_manual_timescales` | CORE | Deterministic timescales plotting is explicitly core. |
| `TestPlotting::test_manual_transition_matrix` | OPTIONAL | Plotting extra beyond core deterministic priorities. |
| `TestPlotting::test_manual_transition_time` | OPTIONAL | Plotting extra beyond core deterministic priorities. |

## Deferred areas identified

- GPCCA output validation.
- Random frame index/frame export.
- PDB generation checks.
- Least-moving-residue export.
- Stochastic-only plots (`macro_feature`, `relative_implied_timescales`, `stochastic_state_similarity`).

## Stale aliases / obsolete paths / deprecated dataset names (document only)

### Stale dataset aliases

- `PDZ3_7` referenced in:
  - `tests/test_properties.py`
  - `tests/test_utils.py`
  - `tests/test_rmsd.py`
- `aSyn_rdc_200ns` referenced in:
  - `tests/test_properties.py`
  - `tests/test_utils.py`
  - `tests/test_rmsd.py`
- `HP35_stoch` appears in `SYSTEMS` lists in:
  - `tests/test_properties.py`
  - `tests/test_utils.py`
  - `tests/test_rmsd.py`

### Obsolete/missing dataset paths currently referenced by tests

- `tests/data/PDZ3_7/input/config.yml`
- `tests/data/aSyn_rdc_200ns/input/config.yml`
- `tests/data/HP35_stoch/input/config.yml`
- `tests/data/<dataset>/config.yml` root-level config references in `test_run.py` (`tests/data/MANIFEST.txt` indicates only `input/config.yml` exists for HP35/PDZ3/aSyn).

### Deprecated name/path indicators already captured in manifest

- `tests/data/MANIFEST.txt` documents these stale aliases and missing paths under:
  - "Additional dataset names referenced by tests"
  - "REQUIRED (missing)" entries per dataset.
