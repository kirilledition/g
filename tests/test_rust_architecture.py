"""Architecture tests for Rust core and Python binding boundaries."""

from __future__ import annotations

import typing
from pathlib import Path

from tooling.debug import check_rust_architecture

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
RUST_SOURCE_DIRECTORY = REPOSITORY_ROOT / "src"
RUST_CRATE_DIRECTORY = REPOSITORY_ROOT / "crates"
PYTHON_BINDING_DIRECTORY = RUST_SOURCE_DIRECTORY / "python"
PYTHON_BINDING_MARKERS = (
    "pyo3",
    "numpy::",
    "#[pyclass",
    "#[pymethods",
    "#[pyfunction",
    "PyArray",
    "PyReadonly",
    "PyReadwrite",
    "PyResult",
    "PyRef",
)


def iter_rust_source_paths() -> typing.Iterator[Path]:
    """Yield Rust source files that should satisfy binding-isolation policy."""
    for rust_source_directory in (RUST_SOURCE_DIRECTORY, RUST_CRATE_DIRECTORY):
        if not rust_source_directory.exists():
            continue
        yield from sorted(rust_source_directory.rglob("*.rs"))


def test_python_binding_markers_are_isolated_to_python_modules() -> None:
    violations: list[str] = []
    for rust_source_path in iter_rust_source_paths():
        if rust_source_path == RUST_SOURCE_DIRECTORY / "lib.rs":
            continue
        if rust_source_path.is_relative_to(PYTHON_BINDING_DIRECTORY):
            continue
        rust_source_text = rust_source_path.read_text(encoding="utf-8")
        observed_markers = [marker for marker in PYTHON_BINDING_MARKERS if marker in rust_source_text]
        if observed_markers:
            relative_source_path = rust_source_path.relative_to(REPOSITORY_ROOT)
            violations.append(f"{relative_source_path}: {', '.join(observed_markers)}")

    assert violations == []


def test_root_crate_boundary_policy_allows_current_private_adapter() -> None:
    assert check_rust_architecture.collect_root_crate_boundary_violations(REPOSITORY_ROOT) == ()


def test_python_telemetry_fallback_policy_allows_current_adapter() -> None:
    assert check_rust_architecture.collect_python_telemetry_fallback_violations(REPOSITORY_ROOT) == ()


def test_python_association_effect_fallback_policy_allows_current_adapter() -> None:
    assert check_rust_architecture.collect_python_association_effect_fallback_violations(REPOSITORY_ROOT) == ()


def test_runtime_telemetry_legacy_close_policy_allows_current_runtime() -> None:
    assert check_rust_architecture.collect_runtime_telemetry_legacy_close_violations(REPOSITORY_ROOT) == ()


def test_root_pyo3_removed_export_policy_allows_current_adapter() -> None:
    assert check_rust_architecture.collect_root_pyo3_export_violations(REPOSITORY_ROOT) == ()


def test_root_crate_boundary_policy_rejects_public_domain_reexports(tmp_path: Path) -> None:
    root_source_directory = tmp_path / "src"
    python_source_directory = root_source_directory / "python"
    python_source_directory.mkdir(parents=True)
    (root_source_directory / "lib.rs").write_text(
        "\n".join(
            (
                "pub use g_engine as engine;",
                "pub mod python;",
                "fn _core() {}",
            )
        ),
        encoding="utf-8",
    )
    (python_source_directory / "mod.rs").write_text("pub fn register_module() {}\n", encoding="utf-8")

    violations = check_rust_architecture.collect_root_crate_boundary_violations(tmp_path)

    assert violations == (
        check_rust_architecture.RootCrateBoundaryViolation(
            source_path=Path("src/lib.rs"),
            marker="pub use g_",
            message="root crate must not re-export internal domain crates as public Rust aliases",
        ),
        check_rust_architecture.RootCrateBoundaryViolation(
            source_path=Path("src/lib.rs"),
            marker="pub mod python;",
            message="root crate must keep its internal PyO3 adapter module private",
        ),
        check_rust_architecture.RootCrateBoundaryViolation(
            source_path=Path("src/lib.rs"),
            marker="mod python;",
            message="root crate must declare the internal PyO3 adapter module privately",
        ),
        check_rust_architecture.RootCrateBoundaryViolation(
            source_path=Path("src/python/mod.rs"),
            marker="pub(crate) fn register_module",
            message="root PyO3 adapter registration must be crate-private",
        ),
    )


def test_python_telemetry_fallback_policy_rejects_rust_to_python_dispatch(tmp_path: Path) -> None:
    python_source_directory = tmp_path / "src" / "python"
    python_source_directory.mkdir(parents=True)
    (python_source_directory / "telemetry.rs").write_text(
        "\n".join(
            (
                'session.call_method1("log_run_failed", ());',
                'session.call_method0("close_with_event");',
                'session.call_method("log_jax_runtime_diagnostic_event", ());',
            )
        ),
        encoding="utf-8",
    )

    violations = check_rust_architecture.collect_python_telemetry_fallback_violations(tmp_path)

    assert violations == (
        check_rust_architecture.PythonTelemetryFallbackViolation(
            source_path=Path("src/python/telemetry.rs"),
            method_name="log_run_failed",
            line_number=1,
            message=check_rust_architecture.PYTHON_TELEMETRY_FALLBACK_MESSAGE,
        ),
        check_rust_architecture.PythonTelemetryFallbackViolation(
            source_path=Path("src/python/telemetry.rs"),
            method_name="close_with_event",
            line_number=2,
            message=check_rust_architecture.PYTHON_TELEMETRY_FALLBACK_MESSAGE,
        ),
        check_rust_architecture.PythonTelemetryFallbackViolation(
            source_path=Path("src/python/telemetry.rs"),
            method_name="log_jax_runtime_diagnostic_event",
            line_number=3,
            message=check_rust_architecture.PYTHON_TELEMETRY_FALLBACK_MESSAGE,
        ),
    )


def test_python_association_effect_fallback_policy_rejects_optional_effect_dispatch(tmp_path: Path) -> None:
    python_source_directory = tmp_path / "src" / "python"
    python_source_directory.mkdir(parents=True)
    (python_source_directory / "association_backend.rs").write_text(
        "\n".join(
            (
                'self.call_optional_effect_method0("open_inputs")?;',
                'self.call_optional_effect_method1("abort_outputs", phase.to_string());',
                'if effects.hasattr("write_batch_result")? { effects.call_method0("write_batch_result")?; }',
            )
        ),
        encoding="utf-8",
    )

    violations = check_rust_architecture.collect_python_association_effect_fallback_violations(tmp_path)

    assert violations == (
        check_rust_architecture.PythonAssociationEffectFallbackViolation(
            source_path=Path("src/python/association_backend.rs"),
            marker="call_optional_effect_method0",
            line_number=1,
            message=check_rust_architecture.PYTHON_ASSOCIATION_EFFECT_FALLBACK_MESSAGE,
        ),
        check_rust_architecture.PythonAssociationEffectFallbackViolation(
            source_path=Path("src/python/association_backend.rs"),
            marker="call_optional_effect_method1",
            line_number=2,
            message=check_rust_architecture.PYTHON_ASSOCIATION_EFFECT_FALLBACK_MESSAGE,
        ),
        check_rust_architecture.PythonAssociationEffectFallbackViolation(
            source_path=Path("src/python/association_backend.rs"),
            marker='hasattr("write_batch_result")',
            line_number=3,
            message=check_rust_architecture.PYTHON_ASSOCIATION_EFFECT_FALLBACK_MESSAGE,
        ),
    )


def test_runtime_telemetry_legacy_close_policy_rejects_legacy_close_fields(tmp_path: Path) -> None:
    runtime_source_directory = tmp_path / "crates" / "runtime" / "src"
    runtime_source_directory.mkdir(parents=True)
    (runtime_source_directory / "telemetry_session.rs").write_text(
        "\n".join(
            (
                "should_emit_legacy_close_event: false,",
                'legacy_close_event_name: "telemetry_session_closed".to_string(),',
                'legacy_close_event_level: "debug".to_string(),',
            )
        ),
        encoding="utf-8",
    )

    violations = check_rust_architecture.collect_runtime_telemetry_legacy_close_violations(tmp_path)

    assert violations == (
        check_rust_architecture.RuntimeTelemetryLegacyCloseViolation(
            source_path=Path("crates/runtime/src/telemetry_session.rs"),
            marker="should_emit_legacy_close_event",
            line_number=1,
            message=check_rust_architecture.RUNTIME_TELEMETRY_LEGACY_CLOSE_MESSAGE,
        ),
        check_rust_architecture.RuntimeTelemetryLegacyCloseViolation(
            source_path=Path("crates/runtime/src/telemetry_session.rs"),
            marker="legacy_close_event_name",
            line_number=2,
            message=check_rust_architecture.RUNTIME_TELEMETRY_LEGACY_CLOSE_MESSAGE,
        ),
        check_rust_architecture.RuntimeTelemetryLegacyCloseViolation(
            source_path=Path("crates/runtime/src/telemetry_session.rs"),
            marker="legacy_close_event_level",
            line_number=3,
            message=check_rust_architecture.RUNTIME_TELEMETRY_LEGACY_CLOSE_MESSAGE,
        ),
    )


def test_root_pyo3_removed_export_policy_rejects_detached_helper_exports(tmp_path: Path) -> None:
    python_source_directory = tmp_path / "src" / "python"
    python_source_directory.mkdir(parents=True)
    (python_source_directory / "shutdown.rs").write_text(
        "\n".join(
            (
                "module.add_function(wrap_pyfunction!(build_shutdown_signal_payload, module)?)?;",
                "module.add_class::<NativeSecondSignalExceptionPlan>()?;",
                "module.add_function(wrap_pyfunction!(emit_diagnostic_event, module)?)?;",
                "module.add_function(wrap_pyfunction!(compile_run_request_json, module)?)?;",
                "module.add_function(wrap_pyfunction!(initialize_pipeline_output_runs, module)?)?;",
                "module.add_function(wrap_pyfunction!(default_nvidia_driver_probe_paths_payload, module)?)?;",
                "module.add_function(wrap_pyfunction!(nvidia_driver_files_are_visible_value, module)?)?;",
                "module.add_function(wrap_pyfunction!(build_jax_runtime_policy_payload, module)?)?;",
                "module.add_function(wrap_pyfunction!(build_logging_runtime_policy_payload, module)?)?;",
                "module.add_function(wrap_pyfunction!(build_process_runtime_state_handle, module)?)?;",
                "module.add_function(wrap_pyfunction!(build_runtime_policy_handle, module)?)?;",
                "module.add_function(wrap_pyfunction!(default_local_cache_directory_value, module)?)?;",
                "module.add_function(wrap_pyfunction!(describe_logging_runtime_policy_value, module)?)?;",
                "module.add_function(wrap_pyfunction!(emit_cli_run_failed_telemetry_event, module)?)?;",
                "module.add_function(wrap_pyfunction!(plan_cli_telemetry_close_failure, module)?)?;",
                "module.add_class::<NativeCliRunFailureTelemetryPlan>()?;",
                "module.add_function(wrap_pyfunction!(resolve_telemetry_output_run_root_value, module)?)?;",
                "module.add_function(wrap_pyfunction!(resolve_telemetry_paths_payload, module)?)?;",
                "module.add_function(wrap_pyfunction!(build_execution_run_artifacts_payload, module)?)?;",
                "module.add_function(wrap_pyfunction!(extend_run_manifest_metadata, module)?)?;",
                "module.add_function(wrap_pyfunction!(build_phenotype_compute_group_id_value, module)?)?;",
                "module.add_function(wrap_pyfunction!(build_phenotype_compute_groups_payload, module)?)?;",
                "module.add_function(wrap_pyfunction!(build_phenotype_output_directory_name, module)?)?;",
                "module.add_function(wrap_pyfunction!(normalize_binary_correction_payload, module)?)?;",
                "module.add_function(wrap_pyfunction!(plan_association_backend_payload, module)?)?;",
                "module.add_function(wrap_pyfunction!(resolve_association_mode_value, module)?)?;",
                "module.add_function(wrap_pyfunction!(plan_bgen_delivery_invocation, module)?)?;",
                "module.add_function(wrap_pyfunction!(resolve_manifest_gpu_genotype_format, module)?)?;",
                "module.add_function(wrap_pyfunction!(build_preflight_report_payload, module)?)?;",
                "module.add_function(wrap_pyfunction!(validate_finite_array_values, module)?)?;",
                "module.add_function(wrap_pyfunction!(build_callback_chunk_identity, module)?)?;",
                "module.add_function(wrap_pyfunction!(emit_callback_progress_update_telemetry, module)?)?;",
                "module.add_function(wrap_pyfunction!(record_runner_run_started_telemetry_event, module)?)?;",
                "module.add_function(wrap_pyfunction!(record_bgen_engine_opened_telemetry_event, module)?)?;",
                "module.add_function(wrap_pyfunction!(build_run_completed_event_payload, module)?)?;",
                "module.add_function(wrap_pyfunction!(render_run_failed_lines, module)?)?;",
                "module.add_function(wrap_pyfunction!(record_native_cli_stdout_diagnostic_event, module)?)?;",
                "module.add_function(wrap_pyfunction!(record_native_cli_completed_line_diagnostic_event, module)?)?;",
                "module.add_function(wrap_pyfunction!(record_runner_run_started_diagnostic_event, module)?)?;",
                (
                    "module.add_function("
                    "wrap_pyfunction!(record_runner_metadata_artifacts_finalized_diagnostic_event, module)?)?;"
                ),
                "module.add_function(wrap_pyfunction!(record_preflight_warning_diagnostic_event, module)?)?;",
                "module.add_function(wrap_pyfunction!(record_pipeline_bgen_engine_opened_diagnostic_event, module)?)?;",
                (
                    "module.add_function("
                    "wrap_pyfunction!("
                    "record_callback_null_logistic_nonconvergence_warning_diagnostic_event, module)?)?;"
                ),
                (
                    "module.add_function("
                    "wrap_pyfunction!(record_native_dispatch_delivery_started_diagnostic_event, module)?)?;"
                ),
                "module.add_function(wrap_pyfunction!(record_jax_runtime_diagnostic_event, module)?)?;",
                (
                    "module.add_function("
                    "wrap_pyfunction!(record_final_timing_outputs_write_started_diagnostic_event, module)?)?;"
                ),
                "module.add_function(wrap_pyfunction!(resolve_final_timing_output_context, module)?)?;",
                "module.add_function(wrap_pyfunction!(build_pipeline_output_preparation_batch_from_values, module)?)?;",
                "module.add_function(wrap_pyfunction!(resolve_output_run_paths, module)?)?;",
                "module.add_function(wrap_pyfunction!(scan_committed_chunk_identifiers, module)?)?;",
                "module.add_function(wrap_pyfunction!(load_run_manifest_payload, module)?)?;",
                "module.add_function(wrap_pyfunction!(write_run_manifest, module)?)?;",
                "module.add_function(wrap_pyfunction!(build_manifest_json_sha256_from_value, module)?)?;",
                "module.add_function(wrap_pyfunction!(build_prepared_run_plan_json_from_current_header, module)?)?;",
                "module.add_function(wrap_pyfunction!(validate_run_manifest_compatibility_from_values, module)?)?;",
                (
                    "module.add_function("
                    "wrap_pyfunction!(read_manifest_committed_chunk_identifiers_from_value, module)?)?;"
                ),
                "module.add_function(wrap_pyfunction!(validate_strict_manifest_chunks_from_value, module)?)?;",
                "module.add_function(wrap_pyfunction!(repair_strict_manifest_chunk_commits_from_value, module)?)?;",
                "module.add_function(wrap_pyfunction!(initialize_output_run_from_values, module)?)?;",
                "module.add_function(wrap_pyfunction!(prepare_output_run, module)?)?;",
                "module.add_function(wrap_pyfunction!(finalize_output_run_chunks, module)?)?;",
                "module.add_function(wrap_pyfunction!(finish_output_writer_session, module)?)?;",
                "module.add_function(wrap_pyfunction!(finish_output_writer_session_interrupted, module)?)?;",
                "module.add_function(wrap_pyfunction!(abort_output_writer_session, module)?)?;",
                "module.add_function(wrap_pyfunction!(write_regenie2_multi_native_chunk, module)?)?;",
                "module.add_function(wrap_pyfunction!(write_regenie2_multi_native_chunk_f64, module)?)?;",
                "module.add_function(wrap_pyfunction!(global_process_runtime_state, module)?)?;",
                "module.add_function(wrap_pyfunction!(close_telemetry_session_with_event, module)?)?;",
            )
        ),
        encoding="utf-8",
    )

    violations = check_rust_architecture.collect_root_pyo3_export_violations(tmp_path)

    assert violations == (
        check_rust_architecture.RootPyO3ExportViolation(
            source_path=Path("src/python/shutdown.rs"),
            export_name="build_shutdown_signal_payload",
            line_number=1,
            message=check_rust_architecture.ROOT_PYO3_REMOVED_EXPORT_MESSAGE,
        ),
        check_rust_architecture.RootPyO3ExportViolation(
            source_path=Path("src/python/shutdown.rs"),
            export_name="NativeSecondSignalExceptionPlan",
            line_number=2,
            message=check_rust_architecture.ROOT_PYO3_REMOVED_EXPORT_MESSAGE,
        ),
        check_rust_architecture.RootPyO3ExportViolation(
            source_path=Path("src/python/shutdown.rs"),
            export_name="emit_diagnostic_event",
            line_number=3,
            message=check_rust_architecture.ROOT_PYO3_REMOVED_EXPORT_MESSAGE,
        ),
        check_rust_architecture.RootPyO3ExportViolation(
            source_path=Path("src/python/shutdown.rs"),
            export_name="compile_run_request_json",
            line_number=4,
            message=check_rust_architecture.ROOT_PYO3_REMOVED_EXPORT_MESSAGE,
        ),
        check_rust_architecture.RootPyO3ExportViolation(
            source_path=Path("src/python/shutdown.rs"),
            export_name="initialize_pipeline_output_runs",
            line_number=5,
            message=check_rust_architecture.ROOT_PYO3_REMOVED_EXPORT_MESSAGE,
        ),
        check_rust_architecture.RootPyO3ExportViolation(
            source_path=Path("src/python/shutdown.rs"),
            export_name="default_nvidia_driver_probe_paths_payload",
            line_number=6,
            message=check_rust_architecture.ROOT_PYO3_REMOVED_EXPORT_MESSAGE,
        ),
        check_rust_architecture.RootPyO3ExportViolation(
            source_path=Path("src/python/shutdown.rs"),
            export_name="nvidia_driver_files_are_visible_value",
            line_number=7,
            message=check_rust_architecture.ROOT_PYO3_REMOVED_EXPORT_MESSAGE,
        ),
        check_rust_architecture.RootPyO3ExportViolation(
            source_path=Path("src/python/shutdown.rs"),
            export_name="build_jax_runtime_policy_payload",
            line_number=8,
            message=check_rust_architecture.ROOT_PYO3_REMOVED_EXPORT_MESSAGE,
        ),
        check_rust_architecture.RootPyO3ExportViolation(
            source_path=Path("src/python/shutdown.rs"),
            export_name="build_logging_runtime_policy_payload",
            line_number=9,
            message=check_rust_architecture.ROOT_PYO3_REMOVED_EXPORT_MESSAGE,
        ),
        check_rust_architecture.RootPyO3ExportViolation(
            source_path=Path("src/python/shutdown.rs"),
            export_name="build_process_runtime_state_handle",
            line_number=10,
            message=check_rust_architecture.ROOT_PYO3_REMOVED_EXPORT_MESSAGE,
        ),
        check_rust_architecture.RootPyO3ExportViolation(
            source_path=Path("src/python/shutdown.rs"),
            export_name="build_runtime_policy_handle",
            line_number=11,
            message=check_rust_architecture.ROOT_PYO3_REMOVED_EXPORT_MESSAGE,
        ),
        check_rust_architecture.RootPyO3ExportViolation(
            source_path=Path("src/python/shutdown.rs"),
            export_name="default_local_cache_directory_value",
            line_number=12,
            message=check_rust_architecture.ROOT_PYO3_REMOVED_EXPORT_MESSAGE,
        ),
        check_rust_architecture.RootPyO3ExportViolation(
            source_path=Path("src/python/shutdown.rs"),
            export_name="describe_logging_runtime_policy_value",
            line_number=13,
            message=check_rust_architecture.ROOT_PYO3_REMOVED_EXPORT_MESSAGE,
        ),
        check_rust_architecture.RootPyO3ExportViolation(
            source_path=Path("src/python/shutdown.rs"),
            export_name="emit_cli_run_failed_telemetry_event",
            line_number=14,
            message=check_rust_architecture.ROOT_PYO3_REMOVED_EXPORT_MESSAGE,
        ),
        check_rust_architecture.RootPyO3ExportViolation(
            source_path=Path("src/python/shutdown.rs"),
            export_name="plan_cli_telemetry_close_failure",
            line_number=15,
            message=check_rust_architecture.ROOT_PYO3_REMOVED_EXPORT_MESSAGE,
        ),
        check_rust_architecture.RootPyO3ExportViolation(
            source_path=Path("src/python/shutdown.rs"),
            export_name="NativeCliRunFailureTelemetryPlan",
            line_number=16,
            message=check_rust_architecture.ROOT_PYO3_REMOVED_EXPORT_MESSAGE,
        ),
        check_rust_architecture.RootPyO3ExportViolation(
            source_path=Path("src/python/shutdown.rs"),
            export_name="resolve_telemetry_output_run_root_value",
            line_number=17,
            message=check_rust_architecture.ROOT_PYO3_REMOVED_EXPORT_MESSAGE,
        ),
        check_rust_architecture.RootPyO3ExportViolation(
            source_path=Path("src/python/shutdown.rs"),
            export_name="resolve_telemetry_paths_payload",
            line_number=18,
            message=check_rust_architecture.ROOT_PYO3_REMOVED_EXPORT_MESSAGE,
        ),
        check_rust_architecture.RootPyO3ExportViolation(
            source_path=Path("src/python/shutdown.rs"),
            export_name="build_execution_run_artifacts_payload",
            line_number=19,
            message=check_rust_architecture.ROOT_PYO3_REMOVED_EXPORT_MESSAGE,
        ),
        check_rust_architecture.RootPyO3ExportViolation(
            source_path=Path("src/python/shutdown.rs"),
            export_name="extend_run_manifest_metadata",
            line_number=20,
            message=check_rust_architecture.ROOT_PYO3_REMOVED_EXPORT_MESSAGE,
        ),
        check_rust_architecture.RootPyO3ExportViolation(
            source_path=Path("src/python/shutdown.rs"),
            export_name="build_phenotype_compute_group_id_value",
            line_number=21,
            message=check_rust_architecture.ROOT_PYO3_REMOVED_EXPORT_MESSAGE,
        ),
        check_rust_architecture.RootPyO3ExportViolation(
            source_path=Path("src/python/shutdown.rs"),
            export_name="build_phenotype_compute_groups_payload",
            line_number=22,
            message=check_rust_architecture.ROOT_PYO3_REMOVED_EXPORT_MESSAGE,
        ),
        check_rust_architecture.RootPyO3ExportViolation(
            source_path=Path("src/python/shutdown.rs"),
            export_name="build_phenotype_output_directory_name",
            line_number=23,
            message=check_rust_architecture.ROOT_PYO3_REMOVED_EXPORT_MESSAGE,
        ),
        check_rust_architecture.RootPyO3ExportViolation(
            source_path=Path("src/python/shutdown.rs"),
            export_name="normalize_binary_correction_payload",
            line_number=24,
            message=check_rust_architecture.ROOT_PYO3_REMOVED_EXPORT_MESSAGE,
        ),
        check_rust_architecture.RootPyO3ExportViolation(
            source_path=Path("src/python/shutdown.rs"),
            export_name="plan_association_backend_payload",
            line_number=25,
            message=check_rust_architecture.ROOT_PYO3_REMOVED_EXPORT_MESSAGE,
        ),
        check_rust_architecture.RootPyO3ExportViolation(
            source_path=Path("src/python/shutdown.rs"),
            export_name="resolve_association_mode_value",
            line_number=26,
            message=check_rust_architecture.ROOT_PYO3_REMOVED_EXPORT_MESSAGE,
        ),
        check_rust_architecture.RootPyO3ExportViolation(
            source_path=Path("src/python/shutdown.rs"),
            export_name="plan_bgen_delivery_invocation",
            line_number=27,
            message=check_rust_architecture.ROOT_PYO3_REMOVED_EXPORT_MESSAGE,
        ),
        check_rust_architecture.RootPyO3ExportViolation(
            source_path=Path("src/python/shutdown.rs"),
            export_name="resolve_manifest_gpu_genotype_format",
            line_number=28,
            message=check_rust_architecture.ROOT_PYO3_REMOVED_EXPORT_MESSAGE,
        ),
        check_rust_architecture.RootPyO3ExportViolation(
            source_path=Path("src/python/shutdown.rs"),
            export_name="build_preflight_report_payload",
            line_number=29,
            message=check_rust_architecture.ROOT_PYO3_REMOVED_EXPORT_MESSAGE,
        ),
        check_rust_architecture.RootPyO3ExportViolation(
            source_path=Path("src/python/shutdown.rs"),
            export_name="validate_finite_array_values",
            line_number=30,
            message=check_rust_architecture.ROOT_PYO3_REMOVED_EXPORT_MESSAGE,
        ),
        check_rust_architecture.RootPyO3ExportViolation(
            source_path=Path("src/python/shutdown.rs"),
            export_name="build_callback_chunk_identity",
            line_number=31,
            message=check_rust_architecture.ROOT_PYO3_REMOVED_EXPORT_MESSAGE,
        ),
        check_rust_architecture.RootPyO3ExportViolation(
            source_path=Path("src/python/shutdown.rs"),
            export_name="emit_callback_progress_update_telemetry",
            line_number=32,
            message=check_rust_architecture.ROOT_PYO3_REMOVED_EXPORT_MESSAGE,
        ),
        check_rust_architecture.RootPyO3ExportViolation(
            source_path=Path("src/python/shutdown.rs"),
            export_name="record_runner_run_started_telemetry_event",
            line_number=33,
            message=check_rust_architecture.ROOT_PYO3_REMOVED_EXPORT_MESSAGE,
        ),
        check_rust_architecture.RootPyO3ExportViolation(
            source_path=Path("src/python/shutdown.rs"),
            export_name="record_bgen_engine_opened_telemetry_event",
            line_number=34,
            message=check_rust_architecture.ROOT_PYO3_REMOVED_EXPORT_MESSAGE,
        ),
        check_rust_architecture.RootPyO3ExportViolation(
            source_path=Path("src/python/shutdown.rs"),
            export_name="build_run_completed_event_payload",
            line_number=35,
            message=check_rust_architecture.ROOT_PYO3_REMOVED_EXPORT_MESSAGE,
        ),
        check_rust_architecture.RootPyO3ExportViolation(
            source_path=Path("src/python/shutdown.rs"),
            export_name="render_run_failed_lines",
            line_number=36,
            message=check_rust_architecture.ROOT_PYO3_REMOVED_EXPORT_MESSAGE,
        ),
        check_rust_architecture.RootPyO3ExportViolation(
            source_path=Path("src/python/shutdown.rs"),
            export_name="record_native_cli_stdout_diagnostic_event",
            line_number=37,
            message=check_rust_architecture.ROOT_PYO3_REMOVED_EXPORT_MESSAGE,
        ),
        check_rust_architecture.RootPyO3ExportViolation(
            source_path=Path("src/python/shutdown.rs"),
            export_name="record_native_cli_completed_line_diagnostic_event",
            line_number=38,
            message=check_rust_architecture.ROOT_PYO3_REMOVED_EXPORT_MESSAGE,
        ),
        check_rust_architecture.RootPyO3ExportViolation(
            source_path=Path("src/python/shutdown.rs"),
            export_name="record_runner_run_started_diagnostic_event",
            line_number=39,
            message=check_rust_architecture.ROOT_PYO3_REMOVED_EXPORT_MESSAGE,
        ),
        check_rust_architecture.RootPyO3ExportViolation(
            source_path=Path("src/python/shutdown.rs"),
            export_name="record_runner_metadata_artifacts_finalized_diagnostic_event",
            line_number=40,
            message=check_rust_architecture.ROOT_PYO3_REMOVED_EXPORT_MESSAGE,
        ),
        check_rust_architecture.RootPyO3ExportViolation(
            source_path=Path("src/python/shutdown.rs"),
            export_name="record_preflight_warning_diagnostic_event",
            line_number=41,
            message=check_rust_architecture.ROOT_PYO3_REMOVED_EXPORT_MESSAGE,
        ),
        check_rust_architecture.RootPyO3ExportViolation(
            source_path=Path("src/python/shutdown.rs"),
            export_name="record_pipeline_bgen_engine_opened_diagnostic_event",
            line_number=42,
            message=check_rust_architecture.ROOT_PYO3_REMOVED_EXPORT_MESSAGE,
        ),
        check_rust_architecture.RootPyO3ExportViolation(
            source_path=Path("src/python/shutdown.rs"),
            export_name="record_callback_null_logistic_nonconvergence_warning_diagnostic_event",
            line_number=43,
            message=check_rust_architecture.ROOT_PYO3_REMOVED_EXPORT_MESSAGE,
        ),
        check_rust_architecture.RootPyO3ExportViolation(
            source_path=Path("src/python/shutdown.rs"),
            export_name="record_native_dispatch_delivery_started_diagnostic_event",
            line_number=44,
            message=check_rust_architecture.ROOT_PYO3_REMOVED_EXPORT_MESSAGE,
        ),
        check_rust_architecture.RootPyO3ExportViolation(
            source_path=Path("src/python/shutdown.rs"),
            export_name="record_jax_runtime_diagnostic_event",
            line_number=45,
            message=check_rust_architecture.ROOT_PYO3_REMOVED_EXPORT_MESSAGE,
        ),
        check_rust_architecture.RootPyO3ExportViolation(
            source_path=Path("src/python/shutdown.rs"),
            export_name="record_final_timing_outputs_write_started_diagnostic_event",
            line_number=46,
            message=check_rust_architecture.ROOT_PYO3_REMOVED_EXPORT_MESSAGE,
        ),
        check_rust_architecture.RootPyO3ExportViolation(
            source_path=Path("src/python/shutdown.rs"),
            export_name="resolve_final_timing_output_context",
            line_number=47,
            message=check_rust_architecture.ROOT_PYO3_REMOVED_EXPORT_MESSAGE,
        ),
        check_rust_architecture.RootPyO3ExportViolation(
            source_path=Path("src/python/shutdown.rs"),
            export_name="build_pipeline_output_preparation_batch_from_values",
            line_number=48,
            message=check_rust_architecture.ROOT_PYO3_REMOVED_EXPORT_MESSAGE,
        ),
        check_rust_architecture.RootPyO3ExportViolation(
            source_path=Path("src/python/shutdown.rs"),
            export_name="resolve_output_run_paths",
            line_number=49,
            message=check_rust_architecture.ROOT_PYO3_REMOVED_EXPORT_MESSAGE,
        ),
        check_rust_architecture.RootPyO3ExportViolation(
            source_path=Path("src/python/shutdown.rs"),
            export_name="scan_committed_chunk_identifiers",
            line_number=50,
            message=check_rust_architecture.ROOT_PYO3_REMOVED_EXPORT_MESSAGE,
        ),
        check_rust_architecture.RootPyO3ExportViolation(
            source_path=Path("src/python/shutdown.rs"),
            export_name="load_run_manifest_payload",
            line_number=51,
            message=check_rust_architecture.ROOT_PYO3_REMOVED_EXPORT_MESSAGE,
        ),
        check_rust_architecture.RootPyO3ExportViolation(
            source_path=Path("src/python/shutdown.rs"),
            export_name="write_run_manifest",
            line_number=52,
            message=check_rust_architecture.ROOT_PYO3_REMOVED_EXPORT_MESSAGE,
        ),
        check_rust_architecture.RootPyO3ExportViolation(
            source_path=Path("src/python/shutdown.rs"),
            export_name="build_manifest_json_sha256_from_value",
            line_number=53,
            message=check_rust_architecture.ROOT_PYO3_REMOVED_EXPORT_MESSAGE,
        ),
        check_rust_architecture.RootPyO3ExportViolation(
            source_path=Path("src/python/shutdown.rs"),
            export_name="build_prepared_run_plan_json_from_current_header",
            line_number=54,
            message=check_rust_architecture.ROOT_PYO3_REMOVED_EXPORT_MESSAGE,
        ),
        check_rust_architecture.RootPyO3ExportViolation(
            source_path=Path("src/python/shutdown.rs"),
            export_name="validate_run_manifest_compatibility_from_values",
            line_number=55,
            message=check_rust_architecture.ROOT_PYO3_REMOVED_EXPORT_MESSAGE,
        ),
        check_rust_architecture.RootPyO3ExportViolation(
            source_path=Path("src/python/shutdown.rs"),
            export_name="read_manifest_committed_chunk_identifiers_from_value",
            line_number=56,
            message=check_rust_architecture.ROOT_PYO3_REMOVED_EXPORT_MESSAGE,
        ),
        check_rust_architecture.RootPyO3ExportViolation(
            source_path=Path("src/python/shutdown.rs"),
            export_name="validate_strict_manifest_chunks_from_value",
            line_number=57,
            message=check_rust_architecture.ROOT_PYO3_REMOVED_EXPORT_MESSAGE,
        ),
        check_rust_architecture.RootPyO3ExportViolation(
            source_path=Path("src/python/shutdown.rs"),
            export_name="repair_strict_manifest_chunk_commits_from_value",
            line_number=58,
            message=check_rust_architecture.ROOT_PYO3_REMOVED_EXPORT_MESSAGE,
        ),
        check_rust_architecture.RootPyO3ExportViolation(
            source_path=Path("src/python/shutdown.rs"),
            export_name="initialize_output_run_from_values",
            line_number=59,
            message=check_rust_architecture.ROOT_PYO3_REMOVED_EXPORT_MESSAGE,
        ),
        check_rust_architecture.RootPyO3ExportViolation(
            source_path=Path("src/python/shutdown.rs"),
            export_name="prepare_output_run",
            line_number=60,
            message=check_rust_architecture.ROOT_PYO3_REMOVED_EXPORT_MESSAGE,
        ),
        check_rust_architecture.RootPyO3ExportViolation(
            source_path=Path("src/python/shutdown.rs"),
            export_name="finalize_output_run_chunks",
            line_number=61,
            message=check_rust_architecture.ROOT_PYO3_REMOVED_EXPORT_MESSAGE,
        ),
        check_rust_architecture.RootPyO3ExportViolation(
            source_path=Path("src/python/shutdown.rs"),
            export_name="finish_output_writer_session",
            line_number=62,
            message=check_rust_architecture.ROOT_PYO3_REMOVED_EXPORT_MESSAGE,
        ),
        check_rust_architecture.RootPyO3ExportViolation(
            source_path=Path("src/python/shutdown.rs"),
            export_name="finish_output_writer_session_interrupted",
            line_number=63,
            message=check_rust_architecture.ROOT_PYO3_REMOVED_EXPORT_MESSAGE,
        ),
        check_rust_architecture.RootPyO3ExportViolation(
            source_path=Path("src/python/shutdown.rs"),
            export_name="abort_output_writer_session",
            line_number=64,
            message=check_rust_architecture.ROOT_PYO3_REMOVED_EXPORT_MESSAGE,
        ),
        check_rust_architecture.RootPyO3ExportViolation(
            source_path=Path("src/python/shutdown.rs"),
            export_name="write_regenie2_multi_native_chunk",
            line_number=65,
            message=check_rust_architecture.ROOT_PYO3_REMOVED_EXPORT_MESSAGE,
        ),
        check_rust_architecture.RootPyO3ExportViolation(
            source_path=Path("src/python/shutdown.rs"),
            export_name="write_regenie2_multi_native_chunk_f64",
            line_number=66,
            message=check_rust_architecture.ROOT_PYO3_REMOVED_EXPORT_MESSAGE,
        ),
        check_rust_architecture.RootPyO3ExportViolation(
            source_path=Path("src/python/shutdown.rs"),
            export_name="global_process_runtime_state",
            line_number=67,
            message=check_rust_architecture.ROOT_PYO3_REMOVED_EXPORT_MESSAGE,
        ),
        check_rust_architecture.RootPyO3ExportViolation(
            source_path=Path("src/python/shutdown.rs"),
            export_name="close_telemetry_session_with_event",
            line_number=68,
            message=check_rust_architecture.ROOT_PYO3_REMOVED_EXPORT_MESSAGE,
        ),
    )


def test_root_pyo3_removed_export_policy_rejects_class_method_exports(tmp_path: Path) -> None:
    python_source_directory = tmp_path / "src" / "python"
    python_source_directory.mkdir(parents=True)
    (python_source_directory / "output.rs").write_text(
        "\n".join(
            (
                "struct NativePreparedOutputRun {",
                "    #[pyo3(get)]",
                "    existing_manifest_json: Option<String>,",
                "}",
                "impl NativeManifestFileFingerprintCache {",
                "    fn build_prediction_loco_file_fingerprints_json(&self) {}",
                "    fn build_current_run_manifest_header_json_from_input_json(&self) {}",
                "}",
                "impl NativeOutputLifecyclePolicy {",
                "    fn build_prepared_run_plan_json_from_current_header(&self) {}",
                "}",
            )
        ),
        encoding="utf-8",
    )

    violations = check_rust_architecture.collect_root_pyo3_export_violations(tmp_path)

    assert violations == (
        check_rust_architecture.RootPyO3ExportViolation(
            source_path=Path("src/python/output.rs"),
            export_name="existing_manifest_json",
            line_number=2,
            message=check_rust_architecture.ROOT_PYO3_REMOVED_EXPORT_MESSAGE,
        ),
        check_rust_architecture.RootPyO3ExportViolation(
            source_path=Path("src/python/output.rs"),
            export_name="build_prediction_loco_file_fingerprints_json",
            line_number=6,
            message=check_rust_architecture.ROOT_PYO3_REMOVED_EXPORT_MESSAGE,
        ),
        check_rust_architecture.RootPyO3ExportViolation(
            source_path=Path("src/python/output.rs"),
            export_name="build_current_run_manifest_header_json_from_input_json",
            line_number=7,
            message=check_rust_architecture.ROOT_PYO3_REMOVED_EXPORT_MESSAGE,
        ),
        check_rust_architecture.RootPyO3ExportViolation(
            source_path=Path("src/python/output.rs"),
            export_name="build_prepared_run_plan_json_from_current_header",
            line_number=10,
            message=check_rust_architecture.ROOT_PYO3_REMOVED_EXPORT_MESSAGE,
        ),
    )


def build_package_payload(package_name: str, dependency_names: tuple[str, ...]) -> dict[str, typing.Any]:
    """Build a small Cargo metadata package payload for architecture tests."""
    return {
        "id": f"path+file:///test/{package_name}#0.1.0",
        "name": package_name,
        "dependencies": [{"name": dependency_name} for dependency_name in dependency_names],
    }


def build_metadata_payload(packages: tuple[dict[str, typing.Any], ...]) -> dict[str, typing.Any]:
    """Build a small Cargo metadata payload for architecture tests."""
    return {
        "packages": list(packages),
        "workspace_members": [str(package_payload["id"]) for package_payload in packages],
    }


def test_rust_architecture_policy_allows_current_single_package_workspace() -> None:
    metadata_payload = build_metadata_payload((build_package_payload("g", ("pyo3", "numpy", "arrow")),))

    assert check_rust_architecture.collect_rust_architecture_violations(metadata_payload) == ()


def test_rust_architecture_policy_rejects_python_binding_dependencies_outside_root() -> None:
    metadata_payload = build_metadata_payload(
        (
            build_package_payload("g", ("g-genotype", "pyo3")),
            build_package_payload("g-genotype", ("pyo3",)),
        )
    )

    violations = check_rust_architecture.collect_rust_architecture_violations(metadata_payload)

    assert violations == (
        check_rust_architecture.RustArchitectureViolation(
            package_name="g-genotype",
            dependency_name="pyo3",
            message="only the root `g` package may depend on PyO3 or NumPy crates",
        ),
    )


def test_rust_architecture_policy_rejects_forbidden_internal_dependencies() -> None:
    metadata_payload = build_metadata_payload(
        (
            build_package_payload("g", ("g-interface",)),
            build_package_payload("g-interface", ("g-genotype", "g-plan")),
            build_package_payload("g-genotype", ()),
            build_package_payload("g-plan", ()),
        )
    )

    violations = check_rust_architecture.collect_rust_architecture_violations(metadata_payload)

    assert violations == (
        check_rust_architecture.RustArchitectureViolation(
            package_name="g-interface",
            dependency_name="g-genotype",
            message="workspace package depends on a forbidden internal crate",
        ),
    )


def test_rust_architecture_policy_restricts_native_cli_frontend_dependencies() -> None:
    metadata_payload = build_metadata_payload(
        (
            build_package_payload("g-cli", ("g-engine", "g-interface")),
            build_package_payload("g-engine", ()),
            build_package_payload("g-interface", ()),
        )
    )

    violations = check_rust_architecture.collect_rust_architecture_violations(metadata_payload)

    assert violations == (
        check_rust_architecture.RustArchitectureViolation(
            package_name="g-cli",
            dependency_name="g-engine",
            message="workspace package depends on a forbidden internal crate",
        ),
    )


def test_rust_architecture_policy_requires_new_internal_crates_to_declare_policy() -> None:
    metadata_payload = build_metadata_payload(
        (
            build_package_payload("g", ("g-surprise",)),
            build_package_payload("g-surprise", ()),
        )
    )

    violations = check_rust_architecture.collect_rust_architecture_violations(metadata_payload)

    assert violations == (
        check_rust_architecture.RustArchitectureViolation(
            package_name="g-surprise",
            dependency_name="*",
            message="workspace package has no declared Rust architecture policy",
        ),
    )
