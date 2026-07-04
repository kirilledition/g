# Python Product Test Suite Audit

Date: 2026-06-14

Scope: product behavior tests under `tests/` for `src/g`. Excluded from this audit: tooling and Symphony automation tests, benchmark/debug harness tests, and Rust-only coverage unless it replaces stale Python product coverage.

Validation baseline: `uv run pytest --collect-only -q tests -m "not phase0_data and not phase1_parity"` collected `767/776` tests after cleanup. The remaining deselections are marker-gated parity/data tests.

## Audit Table

| File | Status | Reason | Current Coverage Value | Replacement Coverage If Removed | Risk |
| --- | --- | --- | --- | --- | --- |
| `tests/test_api.py` | keep | Protects public Python API, lazy import boundary, runtime policy, telemetry events, pipeline dispatch, and manifest metadata. Existing user edits were left untouched. | 43 collected tests. | N/A | High if removed. |
| `tests/test_backend_planner.py` | keep | Covers backend/device/genotype format planning used by public execution paths. | 6 collected tests. | N/A | Medium. |
| `tests/test_callback_lifecycle.py` | keep | Covers native callback lifecycle, progress, writer, and binary correction summary contracts. | 9 collected tests. | N/A | High. |
| `tests/test_cli.py` | delete | Fully dead Click-era module skip after Rust CLI frontend migration. | 0 active tests; module-level skip only. | `tests/test_cli_bridge.py`, Rust `dispatch_cli` coverage, and unskipped CLI smoke. | Low. |
| `tests/test_cli_bridge.py` | keep | Covers native CLI output bridging, configless help/error paths, runtime startup ownership, and concise failure reporting. | 7 collected tests. | N/A | High. |
| `tests/test_cli_smoke.py` | rewrite | Valuable installed-console smoke; stale output assertion and skip removed. | 1 collected test, now active. | N/A | Medium, because it exercises real CLI execution. |
| `tests/test_core.py` | keep | Covers native bridge behavior, BGEN chunk delivery, and prediction source alignment contracts. | 16 collected tests. | N/A | High. |
| `tests/test_interface.py` | rewrite | Removed old dataclass helper skips and rewrote stale TOML/schema placeholders against native config APIs. | 103 collected tests. | N/A | High. |
| `tests/test_io_output.py` | keep | Owns output schema, native writer, manifest, resume, strict validation, and finalization contracts. | 113 collected tests. | N/A | High. |
| `tests/test_io_sample.py` | delete | Covered only removed module-level sample-file PyO3 wrappers. | 0 collected tests after deletion. | `cargo test -p g-input`, engine-owned alignment coverage in `tests/test_regenie2_pipeline.py`. | Low. |
| `tests/test_execution_plan_source.py` | keep | Covers execution-plan BGEN source-path contract helpers. | 4 collected tests. | N/A | Medium. |
| `tests/test_jax_runtime.py` | keep | Protects JAX runtime policy resolution and diagnostics. | 17 collected tests. | N/A | High. |
| `tests/test_preflight.py` | keep | Covers execution preflight validation and native chromosome API use. | 17 collected tests. | N/A | High. |
| `tests/test_regenie2_binary.py` | keep | Large but high-value numerical binary kernel and fallback coverage, including p-threshold, sparse/Firth, packed8, and CPU/GPU parity skipif. | 66 collected tests. | N/A | High. |
| `tests/test_regenie2_binary_config.py` | keep | Focused dataclass validation for binary kernel configs built from native defaults. | 28 collected tests. | N/A | Medium. |
| `tests/test_regenie2_binary_diagnostics.py` | keep | Covers binary diagnostics summarization contracts. | 3 collected tests. | N/A | Medium. |
| `tests/test_regenie2_binary_firth_null.py` | keep | Protects null Firth fitting behavior. | 3 collected tests. | N/A | High. |
| `tests/test_regenie2_binary_full_model.py` | keep | Covers full-model adjusted binary computations. | 10 collected tests. | N/A | High. |
| `tests/test_regenie2_binary_scalar_firth.py` | keep | Covers scalar approximate Firth behavior and sparse correction flags. | 7 collected tests. | N/A | High. |
| `tests/test_regenie2_linear.py` | keep | Protects quantitative kernels, packed8/dosage paths, and numerical edge cases. | 24 collected tests. | N/A | High. |
| `tests/test_regenie2_parity.py` | keep | Marker-gated data parity checks remain valid external baseline coverage. | Deselected under audit marker filter. | N/A | Medium. |
| `tests/parity/test_regenie_parity_harness.py` | keep | Validates parity metadata and harness drift detection without external data. | 5 collected tests. | N/A | Medium. |
| `tests/test_regenie2_pipeline.py` | keep | Large but current: native engine dispatch, callbacks, grouping, resume, shutdown, and trusted BGEN validation. | 92 collected tests. | N/A | High. |
| `tests/test_regenie_binary_correction_contract.py` | rewrite | Rebuilt old dataclass-era skipped contract around `RegenieConfig.from_options` and current binary correction normalization. | 8 collected tests after removing obsolete SPA-only duplicate cases. | API/interface tests cover broader config validation. | Medium. |
| `tests/test_tabular.py` | delete | Covered removed module-level sample-alignment PyO3 wrappers. | 0 collected tests after deletion. | `cargo test -p g-input`, prediction-source aligned-handle coverage in `tests/test_core.py`, grouped/complete-case alignment coverage in `tests/test_regenie2_pipeline.py`. | Medium. |
| `tests/test_telemetry.py` | keep | Covers telemetry path resolution, stream ownership, event schema, caps, and close counters. | 15 collected tests. | N/A | High. |
| `tests/test_timing.py` | keep | Covers stage timing aggregation and persistence contracts. | 10 collected tests. | N/A | Medium. |
| `tests/test_warm_cache.py` | keep | Covers warm-cache planning and native-dispatch cache execution paths. | 8 collected tests. | N/A | Medium. |

## Out Of Scope

The following collected Python files are intentionally excluded from product cleanup status because they test tooling, Symphony automation, architecture/debug checkers, or benchmark/report harnesses: `tests/test_internal_defaults_checker.py`, `tests/test_internal_init_exports_checker.py`, `tests/test_nsight_tool_installer.py`, `tests/test_performance_compare.py`, `tests/test_regenie_comparison_scripts.py`, `tests/test_rust_architecture.py`, `tests/test_symphony_cleanup.py`, `tests/test_symphony_sync_main.py`, and `tests/test_tooling_architecture.py`.

This audit did not require Rust-only coverage changes. Those tests remain relevant replacement coverage for native CLI/config metadata and output manifest checks.

## Function-Level Cleanup Notes

This pass did not produce a full per-function table for every product test file. Function-level review was targeted at stale skipped blocks, dead Click-era coverage, duplicate binary correction contracts, and the large files called out in the cleanup plan. The concrete function-level actions were:

| File | Test or Helper | Action | Reason or Replacement |
| --- | --- | --- | --- |
| `tests/test_cli.py` | module-level skip | Deleted | No active tests remained after the Rust CLI frontend replaced Click internals. Covered by `tests/test_cli_bridge.py`, Rust `dispatch_cli` tests, and active CLI smoke. |
| `tests/test_cli_smoke.py` | `test_installed_cli_runs_regenie2_linear_smoke` | Rewritten | Removed stale skip and changed the old `"Parquet dataset saved"` assertion to the current completion line contract. |
| `tests/test_regenie_binary_correction_contract.py` | `build_binary_config` | Rewritten | Replaced skipped dataclass helper with native-backed `RegenieConfig.from_options` construction. |
| `tests/test_regenie_binary_correction_contract.py` | `test_default_binary_config_normalizes_to_score_only` | Rewritten | Keeps score-only binary correction contract; now tolerates native float32 threshold representation. |
| `tests/test_regenie_binary_correction_contract.py` | `test_firth_approx_maps_to_approximate_firth_plan` | Rewritten | Keeps approximate Firth plan contract; now uses native config parsing. |
| `tests/test_regenie_binary_correction_contract.py` | `test_approx_without_firth_raises` | Rewritten | Current validation occurs at native config construction. |
| `tests/test_regenie_binary_correction_contract.py` | `test_firth_and_spa_raises_for_spa` | Deleted | `spa` is not a supported option in the current native Python option surface; unsupported option coverage exists in `tests/test_interface.py`. |
| `tests/test_regenie_binary_correction_contract.py` | `test_invalid_p_threshold_values_raise` | Rewritten | Keeps threshold validation contract across native/config and execution-plan validation boundaries. |
| `tests/test_regenie_binary_correction_contract.py` | `test_spa_raises_until_implemented` | Deleted | Duplicate obsolete SPA contract; `spa` is now rejected as unknown before binary plan normalization. |
| `tests/test_regenie_binary_correction_contract.py` | `test_exact_firth_raises_until_parity_proven` | Rewritten | Current validation occurs at native config construction. |
| `tests/test_interface.py` | `build_input_config`, `build_trait_config`, `build_binary_config`, `build_output_config` | Deleted | Dead skipped dataclass-era helpers, no current callers. |
| `tests/test_interface.py` | `test_every_supported_option_has_explain_metadata` | Rewritten | Replaced stale explain-metadata placeholder with native option surface metadata assertions. |
| `tests/test_interface.py` | `test_packaged_default_catalog_matches_option_policies` | Rewritten | Replaced stale default-catalog placeholder with `config.default.toml` versus native option registry coverage. |
| `tests/test_interface.py` | `test_packaged_default_hash_uses_raw_toml_payload` | Rewritten | Verifies emitted effective TOML metadata uses the raw packaged default TOML hash. |
| `tests/test_interface.py` | `test_typed_toml_schema_matches_option_registry` | Rewritten | Verifies resolved TOML sections stay inside the native option registry, including normalized column-list fields. |
| `tests/test_interface.py` | `test_packaged_default_toml_decodes_to_typed_config` | Rewritten | Verifies packaged defaults serialize to typed native config values. |
| `tests/test_interface.py` | `test_msgspec_toml_schema_rejects_unknown_keys_and_wrong_types` | Rewritten | Replaced msgspec-era placeholder with native TOML schema rejection checks. |
| `tests/test_interface.py` | `test_msgspec_toml_schema_rejects_removed_jax_x64_option` | Rewritten | Replaced msgspec-era placeholder with native rejection of removed `jax_x64`. |
| `tests/test_interface.py` | `test_toml_metadata_is_accepted_but_not_an_option` | Rewritten | Verifies effective TOML metadata can be read but is not exposed as a flat Python option. |
| `tests/test_interface.py` | `test_quantitative_execution_plan_rejects_direct_binary_only_config` | Deleted | Dead direct-config-object placeholder; binary-only option rejection is already covered by native Python option tests in the same file. |
| `tests/test_interface.py` | `test_toml_serialization_emits_multi_column_and_binary_sections` | Rewritten | Verifies current TOML emission for multi-column input and binary sections. |

The oversized/high-coupling files were reviewed at file/section level rather than every individual test function. No individual tests in `tests/test_regenie2_pipeline.py`, `tests/test_regenie2_binary.py`, `tests/test_io_output.py`, or `tests/test_api.py` were deleted or rewritten in this pass because the collected tests protect current public contracts: native dispatch and callback behavior, binary numerical/fallback behavior, output/resume manifest contracts, and Python API/runtime contracts.
