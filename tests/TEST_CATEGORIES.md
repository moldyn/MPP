# Test Suite Categories

All 67 tests collected by `python -m pytest tests/ --collect-only -q` pass.
All tests have the required data files and run as part of the normal suite.

The OPTIONAL / DEFERRED classification below is **historical** — it reflects the state
before the full test dataset (XTC trajectories, topology files) was committed.
All tests are now effectively CORE.

---

## Per-test classification (historical)

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
| `test_utils.TestPureUtils::test_argmedian_odd` | CORE | Unit test for argmedian utility (odd-length array). |
| `test_utils.TestPureUtils::test_argmedian_even` | CORE | Unit test for argmedian utility (even-length array). |
| `test_utils.TestPureUtils::test_weighting_function_single` | CORE | Unit test for weighting_function (scalar case). |
| `test_utils.TestPureUtils::test_weighting_function_multi` | CORE | Unit test for weighting_function (vector case). |
| `test_utils.TestPureUtils::test_find_state_lengths_simple` | CORE | Unit test for run-length encoding utility. |
| `test_utils.TestPureUtils::test_find_state_lengths_single` | CORE | Unit test for run-length encoding (single element). |
| `test_utils.TestPureUtils::test_get_multi_state_trajectory_none_limits` | CORE | Unit test for trajectory split (no limits). |
| `test_utils.TestPureUtils::test_get_multi_state_trajectory_splits` | CORE | Unit test for trajectory split (with limits). |

### `tests/test_run.py`

| Test case | Category | Rationale |
|---|---|---|
| `TestRunScript::test_HP35_t_ref` | CORE | CLI/API deterministic lumping run (T kernel). |
| `TestRunScript::test_HP35_t_stoch` | CORE | Stochastic lumping with seeded RNG — reproducible, regression-testable. |
| `TestRunScript::test_HP35_kl` | CORE | CLI/API deterministic lumping run (KL kernel). |
| `TestRunScript::test_HP35_t_js` | CORE | CLI/API deterministic lumping run (T + JS kernels). |
| `TestRunScript::test_HP35_js` | CORE | CLI/API deterministic lumping run (JS feature-kernel path). |
| `TestRunScript::test_HP35_gpcca` | CORE | GPCCA output path — previously deferred, now passing with full data. |
| `TestRunScript::test_PDZ3_kl` | CORE | Deterministic KL run on PDZ3 dataset; reproducibility-critical. |
| `TestRunScript::test_aSyn_t` | CORE | Deterministic T run on aSyn dataset; reproducibility-critical. |
| `TestRunScript::test_aSyn_kl_js` | CORE | Deterministic mixed-kernel run in core pipeline. |
| `TestRunScript::test_aSyn_t_stoch` | CORE | Stochastic lumping with seeded RNG — reproducible, regression-testable. |
| `TestRunScript::test_random_frames_indices_aSyn_t_ref` | CORE | Random frame index generation — previously deferred, now passing. |
| `TestRunScript::test_macrostate_map_saved_alongside_z` | CORE | Verifies macrostate_map.npy is written alongside Z.npy. |
| `TestRunScript::test_macrostate_map_reloaded_on_load` | CORE | Verifies macrostate_map.npy is (re-)written when loading an existing Z. |
| `TestConfigNormalization::test_canonical_keys_load_without_warning` | CORE | New snake_case keys load silently. |
| `TestConfigNormalization::test_legacy_keys_emit_deprecation_warning` | CORE | Old space-separated keys emit DeprecationWarning. |
| `TestConfigNormalization::test_duplicate_legacy_and_canonical_keys_raises` | CORE | Duplicate key conflict raises ValueError. |
| `TestConfigNormalization::test_normalize_config_canonical_keys_unchanged` | CORE | Canonical keys pass through _normalize_config unchanged. |
| `TestConfigNormalization::test_normalize_config_all_aliases` | CORE | All legacy aliases renamed to canonical form. |
| `TestCLIValidation::test_invalid_d_exits_with_error` | CORE | CLI argument validation. |
| `TestCLIValidation::test_invalid_g_exits_with_error` | CORE | CLI argument validation. |
| `TestCLIValidation::test_missing_z_for_mpp_exits_with_error` | CORE | CLI argument validation. |
| `TestCLIValidation::test_plot_without_out_exits_with_error` | CORE | CLI argument validation. |
| `TestCLIValidation::test_missing_required_config_key_raises` | CORE | Missing required config key raises ValueError. |
| `TestCLIValidation::test_nonexistent_config_file_gives_argparse_error` | CORE | Non-existent config path gives non-zero exit. |
| `TestCLIValidation::test_metrics_flag_prints_all_keys` | CORE | --metrics flag prints all quality metric keys (TASK-5.6). |

### `tests/test_regression_baseline.py`

| Test case | Category | Rationale |
|---|---|---|
| `test_generated_z_matches_baseline` | CORE | Regenerates Z from config and compares to committed baseline. |
| `test_generated_macrostate_assignment_matches_baseline` | CORE | Regenerates macrostate assignment and compares to committed baseline. |

### `tests/test_rmsd.py`

| Test case | Category | Rationale |
|---|---|---|
| `TestRMSD_HP35::test_draw_random_indices` | CORE | Random frame index generation — previously deferred, now passing. |
| `TestRMSD_HP35::test_draw_random_frames` | CORE | Random frame export — previously deferred, now passing with HP35 XTC. |
| `TestRMSD_aSyn::test_write_least_moving_residues` | CORE | Least-moving-residue export — previously deferred, now passing. |
| `TestRMSD_PDZ3::test_rmsd_property` | CORE | RMSD computation and save/load roundtrip — previously optional, now passing with PDZ3 XTC. |
| `TestRMSD_PDZ3::test_rmsd_sharpness` | CORE | RMSD sharpness metric regression — previously optional, now passing. |

### `tests/test_plots.py`

| Test case | Category | Rationale |
|---|---|---|
| `TestPlotting::test_manual_ck_test` | CORE | Deterministic diagnostic plotting in core analysis path. |
| `TestPlotting::test_manual_contacts` | CORE | Deterministic core plot output coverage. |
| `TestPlotting::test_manual_delta_rmsd` | CORE | RMSD delta plot — previously optional, now passing with PDZ3 XTC. |
| `TestPlotting::test_manual_dendrogram` | CORE | Deterministic dendrogram plotting is explicitly core. |
| `TestPlotting::test_manual_macro_feature` | CORE | Stochastic macro-feature plot — previously deferred, now passing. |
| `TestPlotting::test_manual_macrotraj_PDZ3` | CORE | Deterministic macrostate trajectory plotting for core datasets. |
| `TestPlotting::test_manual_macrotraj_ref` | CORE | Deterministic macrostate trajectory plotting in core path. |
| `TestPlotting::test_manual_relative_implied_timescales` | CORE | Stochastic timescales plot — previously deferred, now passing. |
| `TestPlotting::test_manual_rmsd` | CORE | RMSD plot — previously optional, now passing with PDZ3 XTC. |
| `TestPlotting::test_manual_sankey` | CORE | Deterministic sankey plotting in core deterministic analysis. |
| `TestPlotting::test_manual_state_network` | CORE | Deterministic state-network plotting in core deterministic analysis. |
| `TestPlotting::test_manual_stochastic_state_similarity` | CORE | Stochastic similarity plot — previously deferred, now passing. |
| `TestPlotting::test_manual_timescales` | CORE | Deterministic timescales plotting is explicitly core. |
| `TestPlotting::test_manual_transition_matrix` | CORE | Transition matrix plot — previously optional, now passing. |
| `TestPlotting::test_manual_transition_time` | CORE | Transition time plot — previously optional, now passing. |

---

## Coverage summary (as of 2026-05-27, 67 tests)

Overall: **85%** branch coverage. Key remaining gaps:

| File | Coverage | Main uncovered area |
|------|----------|---------------------|
| `random_to_png.py` | 0% | PyMol PNG rendering — requires PyMol binary, not available in CI |
| `sankey_gap.py` | 71% | Edge-case layout branches (single-state, no-gap paths) |
| `kernel.py` | 82% | Stochastic `"p"`-method (probability-mass selection) not tested |
| `lumping.py` | 84% | Scattered guards and error paths |
| `run.py` | 86% | A few config-loading error branches |
| `core.py` | 89% | Rare tree-traversal edge cases |
| `utils.py` | 89% | `calc_rmsd`/`calc_rmsd_feature` error paths |
| `plot.py` | 90% | Conditional plot branches |
