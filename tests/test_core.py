from __future__ import annotations

import json
import subprocess
import sys
import typing
import unittest.mock
from pathlib import Path

import numpy as np
import pytest

from g import _core

TEST_DATA_DIRECTORY = Path(__file__).parent / "data" / "bgen"
HAPLOTYPES_BGEN_PATH = TEST_DATA_DIRECTORY / "haplotypes.bgen"
TRUSTED_PACKED8_BGEN_PATH = Path(__file__).parents[1] / "reference" / "regenie-patched" / "example" / "example.bgen"


def run_logging_subprocess(script: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
    )


def build_native_runtime_compatibility_token() -> _core.NativeRuntimeCompatibilityToken:
    runtime_state = _core.NativeRuntimeState()
    logging_policy_payload = runtime_state.build_logging_runtime_policy_payload(
        log_filter="info",
        log_file=None,
        log_stderr=False,
        log_queue_size=1024,
        log_lossy=True,
        include_source_location=False,
        include_span_events=False,
        trace_file=None,
        trace_filter="info",
        trace_event_cap=None,
        telemetry_mode="off",
        telemetry_stream_file=None,
    )
    jax_policy_payload: dict[str, object] = {
        "device": "cpu",
        "cache_directory": None,
        "matmul_precision": None,
        "persistent_cache": True,
        "persistent_cache_min_entry_size_bytes": 0,
        "persistent_cache_min_compile_time_seconds": 0,
        "xla_autotune_cache": False,
        "transfer_guard": False,
    }
    return runtime_state.require_compatible_runtime_policy(
        logging_policy_payload,
        None,
        jax_policy_payload,
    )


def build_native_jax_runtime_setup_session(
    *,
    requested_device: str,
    cache_directory: str,
    matmul_precision: str | None = None,
    persistent_cache: bool = True,
    persistent_cache_min_entry_size_bytes: int = 0,
    persistent_cache_min_compile_time_seconds: int = 0,
    xla_autotune_cache: bool = False,
    transfer_guard: bool = False,
) -> _core.NativeJaxRuntimeSetupSession:
    """Build a native setup session through the runtime-state boundary."""
    runtime_state = _core.NativeRuntimeState()
    jax_policy_payload = runtime_state.build_jax_runtime_policy_payload(
        device=requested_device,
        cache_directory=cache_directory,
        matmul_precision=matmul_precision,
        persistent_cache=persistent_cache,
        persistent_cache_min_entry_size_bytes=persistent_cache_min_entry_size_bytes,
        persistent_cache_min_compile_time_seconds=persistent_cache_min_compile_time_seconds,
        xla_autotune_cache=xla_autotune_cache,
        transfer_guard=transfer_guard,
    )
    return runtime_state.build_jax_runtime_setup_session(jax_policy_payload, cache_directory)


class RecordingNativeCallbackTelemetrySession:
    """Telemetry double for native callback emission helpers."""

    def __init__(self) -> None:
        """Initialize captured telemetry calls."""
        self.native_telemetry_session = self
        self.progress_events: list[tuple[str, str, str, int]] = []
        self.progress_records: list[dict[str, object]] = []
        self.binary_summaries: list[dict[str, int]] = []

    def emit_callback_progress_event(self, progress_event: _core.NativeCallbackProgressTelemetryEvent) -> None:
        """Record one native callback progress event."""
        self.progress_events.append(
            (
                progress_event.event_name,
                progress_event.level,
                progress_event.chromosome,
                progress_event.processed_chunk_count,
            )
        )

    def emit_progress(self, processed_chunk_count: int, fields: dict[str, object]) -> None:
        """Record one native callback progress record."""
        self.progress_records.append({"processed_chunk_count": processed_chunk_count, **fields})

    def log_binary_correction_summary(self, summary_payload: dict[str, int]) -> None:
        """Record one binary correction summary payload."""
        self.emit_binary_correction_summary_event(summary_payload)

    def emit_binary_correction_summary_event(self, summary_payload: dict[str, int]) -> None:
        """Record one binary correction summary payload through the handle."""
        self.binary_summaries.append(dict(summary_payload))


def test_initialize_logging_is_idempotent_and_writes_python_and_rust_jsonl(tmp_path: Path) -> None:
    log_path = tmp_path / "g.jsonl"

    completed_process = run_logging_subprocess(
        "\n".join(
            [
                "import logging",
                "from g import _core",
                f"log_path = {str(log_path)!r}",
                'first_result = _core.initialize_logging(log_filter="info", log_file=log_path, log_stderr=False)',
                'second_result = _core.initialize_logging(log_filter="debug", log_file=log_path, log_stderr=False)',
                'logging.warning("python warning reaches tracing")',
                "_core.shutdown_logging()",
                "print(first_result, second_result)",
            ]
        )
    )

    log_text = log_path.read_text(encoding="utf-8")
    records = [json.loads(line) for line in log_text.splitlines() if line]

    assert completed_process.stdout.strip() == "True False"
    assert records
    assert "python warning reaches tracing" in log_text
    assert "logging initialized" in log_text


def test_initialize_logging_defaults_to_info_filter(tmp_path: Path) -> None:
    log_path = tmp_path / "g-default.jsonl"

    run_logging_subprocess(
        "\n".join(
            [
                "import logging",
                "from g import _core",
                f"log_path = {str(log_path)!r}",
                "_core.initialize_logging(log_file=log_path, log_stderr=False)",
                'logging.warning("default warning is visible")',
                "_core.shutdown_logging()",
            ]
        )
    )

    log_text = log_path.read_text(encoding="utf-8")
    records = [json.loads(line) for line in log_text.splitlines() if line]

    assert records
    assert "default warning is visible" in log_text
    assert "logging initialized" in log_text


def test_raw_diagnostic_emitters_are_not_exported() -> None:
    assert not hasattr(_core, "emit_diagnostic_event")
    assert not hasattr(_core, "emit_diagnostic_event_fields")


def test_unused_raw_payload_builders_are_not_exported() -> None:
    assert not hasattr(_core, "configure_bgen_decode_tile_variant_count")
    assert not hasattr(_core, "configure_rayon_global_thread_pool")
    assert not hasattr(_core, "build_default_local_cache_directory_value")
    assert not hasattr(_core, "default_local_cache_directory_value")
    assert not hasattr(_core, "build_file_content_sha256_value")
    assert not hasattr(_core, "build_current_run_manifest_header_json_from_input_json")
    assert not hasattr(_core, "build_manifest_file_fingerprint_payload")
    assert not hasattr(_core, "build_manifest_file_fingerprint_mapping_payload")
    assert not hasattr(_core, "build_multi_run_artifacts_payload")
    assert not hasattr(_core, "build_phenotype_run_artifacts_payload")
    assert not hasattr(_core, "build_prepared_run_manifest_header_json")
    assert not hasattr(_core, "build_prepared_run_plan_json")
    assert not hasattr(_core, "build_prediction_loco_file_fingerprints_json")
    assert not hasattr(_core, "build_run_manifest_extension_payload")
    assert not hasattr(_core, "build_trusted_bgen_validation_cache_payload")
    assert not hasattr(_core, "compile_run_request_json")
    assert not hasattr(_core, "initialize_pipeline_output_run_batch")
    assert not hasattr(_core, "initialize_pipeline_output_runs")
    assert not hasattr(_core, "build_trusted_bgen_validation_cache_path_value")
    assert not hasattr(_core, "build_trusted_bgen_validation_fingerprint_value")
    assert not hasattr(_core, "default_trusted_bgen_validation_cache_directory_value")
    assert not hasattr(_core, "default_local_temporary_root_value")
    assert not hasattr(_core, "format_rayon_thread_pool_configuration_error_value")
    assert not hasattr(_core, "NativeStageTimingRecorderPlan")
    assert not hasattr(_core, "NativeTelemetryClosePlan")
    assert not hasattr(_core, "NativeTelemetryEventEmissionPlan")
    assert not hasattr(_core, "NativeTelemetryProgressEmissionPlan")
    assert not hasattr(_core, "NativeTimingFileWritePlan")
    assert not hasattr(_core, "plan_stage_timing_recorder")
    assert not hasattr(_core, "plan_telemetry_close")
    assert not hasattr(_core, "plan_telemetry_event_emission")
    assert not hasattr(_core, "plan_telemetry_progress_emission")
    assert not hasattr(_core.NativePreparedOutputRun, "existing_manifest_json")
    assert not hasattr(
        _core.NativeManifestFileFingerprintCache,
        "build_current_run_manifest_header_json_from_input_json",
    )
    assert not hasattr(
        _core.NativeManifestFileFingerprintCache,
        "build_prediction_loco_file_fingerprints_json",
    )
    native_pipeline_output_preparation_batch_type = typing.cast(
        "typing.Any",
        _core.NativePipelineOutputPreparationBatch,
    )
    with pytest.raises(TypeError):
        native_pipeline_output_preparation_batch_type(
            (),
            (),
            (),
            (),
            resume=False,
            resume_mode="fast",
        )
    assert not hasattr(_core, "plan_timing_file_write")
    assert not hasattr(_core, "plan_null_logistic_nonconvergence")
    for removed_scheduler_export_name in (
        "format_dosage_callback_worker_error_message",
        "format_result_callback_worker_error_message",
        "plan_callback_queue_backpressure_observation",
        "plan_callback_queue_operation_observation",
        "plan_callback_queue_stage_backpressure_observation",
        "plan_callback_queue_stage_observation",
        "plan_callback_worker_abort",
        "plan_callback_worker_finish",
        "plan_callback_worker_start",
        "plan_callback_worker_stop_poll",
        "plan_dosage_buffer_reuse",
        "plan_dosage_callback_worker_join",
        "plan_dosage_callback_worker_stop",
        "plan_dosage_work_handoff",
        "plan_dosage_work_item_dispatch",
        "plan_dosage_work_item_stage_duration",
        "plan_result_callback_worker_join",
        "plan_result_callback_worker_stop",
        "plan_result_write_handoff",
        "plan_result_write_item_dispatch",
        "plan_variant_major_dosage_batch_handoff",
        "resolve_bgen_delivery_method_value",
        "resolve_callback_worker_backpressure_poll_timeout_seconds",
        "resolve_callback_worker_stop_poll_timeout_seconds",
        "resolve_native_callback_queue_limits",
        "resolve_native_callback_worker_shutdown_timeouts",
        "should_attempt_callback_worker_stop",
        "validate_pipeline_resume_compatibility",
    ):
        assert not hasattr(_core, removed_scheduler_export_name)
    assert not hasattr(_core, "plan_trusted_bgen_validation_cache_lookup")
    assert not hasattr(_core, "validate_binary_phenotype_case_control_counts")
    assert not hasattr(_core, "validate_binary_phenotype_coding")
    assert not hasattr(_core, "validate_finite_array")
    assert not hasattr(_core, "write_trusted_bgen_validation_cache_payload")
    assert not hasattr(_core.Regenie2RunEngine, "validate_trusted_no_missing_diploid_with_cache")


def test_plan_genotype_chunks_splits_by_boundaries_and_resume_state() -> None:
    """Ensure the native chunk planner returns chromosome-homogeneous work units."""
    chunks = _core.plan_genotype_chunks(
        variant_count=12,
        chunk_size=5,
        chromosome_boundary_indices=[0, 3, 9, 12],
        committed_chunk_identifiers=[5],
    )

    assert [(chunk.variant_start_index, chunk.variant_stop_index) for chunk in chunks] == [
        (0, 3),
        (3, 5),
        (9, 10),
        (10, 12),
    ]


def test_intersect_committed_chunk_identifier_sets_returns_sorted_shared_identifiers() -> None:
    shared_chunk_identifiers = _core.intersect_committed_chunk_identifier_sets(((64, 0, 32), (32, 64, 96), (32, 128)))

    assert shared_chunk_identifiers == [32]
    assert _core.intersect_committed_chunk_identifier_sets(()) == []


def test_plan_multi_trait_chunk_write_uses_native_committed_chunk_policy() -> None:
    write_plan = _core.plan_multi_trait_chunk_write(
        writer_session_count=3,
        chunk_identifier=32,
        committed_chunk_identifier_sets=((0,), (32,), (64,)),
    )
    assert write_plan.active_trait_indices == [0, 2]
    assert write_plan.total_trait_count == 3
    assert write_plan.active_trait_count == 2
    assert write_plan.all_traits_committed is False

    committed_write_plan = _core.plan_multi_trait_chunk_write(
        writer_session_count=2,
        chunk_identifier=32,
        committed_chunk_identifier_sets=((32,), (0, 32)),
    )
    assert committed_write_plan.active_trait_indices == []
    assert committed_write_plan.active_trait_count == 0
    assert committed_write_plan.all_traits_committed is True

    with pytest.raises(ValueError, match="Committed chunk identifier set count"):
        _core.plan_multi_trait_chunk_write(
            writer_session_count=2,
            chunk_identifier=32,
            committed_chunk_identifier_sets=((32,),),
        )


def test_native_python_association_backend_uses_coarse_typed_calls() -> None:
    class RecordingPythonAssociationBackend:
        def __init__(self) -> None:
            self.group_identifier: str | None = None
            self.phenotype_count: int | None = None
            self.chromosome: str | None = None
            self.prediction_chromosome: str | None = None
            self.prediction_row_count: int | None = None
            self.batch_chromosome: str | None = None
            self.batch_variant_count: int | None = None
            self.batch_variant_offset: int | None = None

        def prepare_group(self, group_input: _core.NativePreparedGroupInput) -> dict[str, object]:
            self.group_identifier = group_input.group_identifier
            self.phenotype_count = group_input.phenotype_count
            return {"group_identifier": group_input.group_identifier, "phenotype_count": group_input.phenotype_count}

        def prepare_chromosome(
            self,
            group_state: object,
            chromosome: str,
            predictions: _core.NativePredictionView,
        ) -> dict[str, object]:
            group_state_mapping = typing.cast("dict[str, object]", group_state)
            self.chromosome = chromosome
            self.prediction_chromosome = predictions.chromosome
            self.prediction_row_count = predictions.row_count
            return {
                "group_identifier": group_state_mapping["group_identifier"],
                "chromosome": chromosome,
                "prediction_row_count": predictions.row_count,
            }

        def compute_batch(
            self,
            chromosome_state: object,
            batch: _core.NativeGenotypeBatchView,
        ) -> _core.NativeAssociationBatchResult:
            chromosome_state_mapping = typing.cast("dict[str, object]", chromosome_state)
            self.batch_chromosome = batch.chromosome
            self.batch_variant_count = batch.variant_count
            self.batch_variant_offset = batch.variant_offset
            statistic_sum = typing.cast("int", chromosome_state_mapping["prediction_row_count"])
            statistic_sum += batch.variant_count + batch.variant_offset
            return _core.NativeAssociationBatchResult(batch.chromosome, batch.variant_count, float(statistic_sum))

    python_backend = RecordingPythonAssociationBackend()
    native_backend = _core.NativePythonAssociationBackend(python_backend)

    group_state = native_backend.prepare_group("binary", 2)
    chromosome_state = native_backend.prepare_chromosome(group_state, "chr2", "chr2", 5)
    result = native_backend.compute_batch(chromosome_state, "chr2", 4, 3)

    assert python_backend.group_identifier == "binary"
    assert python_backend.phenotype_count == 2
    assert python_backend.chromosome == "chr2"
    assert python_backend.prediction_chromosome == "chr2"
    assert python_backend.prediction_row_count == 5
    assert python_backend.batch_chromosome == "chr2"
    assert python_backend.batch_variant_count == 4
    assert python_backend.batch_variant_offset == 3
    assert result.chromosome == "chr2"
    assert result.variant_count == 4
    assert result.statistic_sum == 12.0


def test_native_python_association_backend_runs_native_coordinator_single_batch() -> None:
    class CoordinatorPythonAssociationBackend:
        def __init__(self) -> None:
            self.call_names: list[str] = []

        def prepare_group(self, group_input: _core.NativePreparedGroupInput) -> dict[str, object]:
            self.call_names.append("prepare_group")
            return {"group_identifier": group_input.group_identifier, "phenotype_count": group_input.phenotype_count}

        def prepare_chromosome(
            self,
            group_state: object,
            chromosome: str,
            predictions: _core.NativePredictionView,
        ) -> dict[str, object]:
            self.call_names.append("prepare_chromosome")
            group_state_mapping = typing.cast("dict[str, object]", group_state)
            return {
                "group_identifier": group_state_mapping["group_identifier"],
                "phenotype_count": group_state_mapping["phenotype_count"],
                "chromosome": chromosome,
                "prediction_row_count": predictions.row_count,
            }

        def compute_batch(
            self,
            chromosome_state: object,
            batch: _core.NativeGenotypeBatchView,
        ) -> _core.NativeAssociationBatchResult:
            self.call_names.append("compute_batch")
            chromosome_state_mapping = typing.cast("dict[str, object]", chromosome_state)
            statistic_sum = typing.cast("int", chromosome_state_mapping["phenotype_count"])
            statistic_sum *= batch.variant_count
            statistic_sum += typing.cast("int", chromosome_state_mapping["prediction_row_count"])
            statistic_sum += batch.variant_offset
            return _core.NativeAssociationBatchResult(batch.chromosome, batch.variant_count, float(statistic_sum))

    python_backend = CoordinatorPythonAssociationBackend()
    native_backend = _core.NativePythonAssociationBackend(python_backend)

    report = native_backend.run_single_batch("binary", 2, "chr2", "chr2", 5, "chr2", 4, 3)

    assert python_backend.call_names == ["prepare_group", "prepare_chromosome", "compute_batch"]
    assert report.phase_history == [
        "planned",
        "inputs_opened",
        "inputs_aligned",
        "preflight_validated",
        "outputs_initialized",
        "running",
        "draining",
        "finalizing",
        "completed",
    ]
    assert report.result.chromosome == "chr2"
    assert report.result.variant_count == 4
    assert report.result.statistic_sum == 16.0


def test_native_python_association_backend_runs_single_batch_with_effects() -> None:
    class CoordinatorPythonAssociationBackend:
        def prepare_group(self, group_input: _core.NativePreparedGroupInput) -> dict[str, object]:
            return {"phenotype_count": group_input.phenotype_count}

        def prepare_chromosome(
            self,
            group_state: object,
            chromosome: str,
            predictions: _core.NativePredictionView,
        ) -> dict[str, object]:
            group_state_mapping = typing.cast("dict[str, object]", group_state)
            return {
                "phenotype_count": group_state_mapping["phenotype_count"],
                "chromosome": chromosome,
                "prediction_row_count": predictions.row_count,
            }

        def compute_batch(
            self,
            chromosome_state: object,
            batch: _core.NativeGenotypeBatchView,
        ) -> _core.NativeAssociationBatchResult:
            chromosome_state_mapping = typing.cast("dict[str, object]", chromosome_state)
            statistic_sum = typing.cast("int", chromosome_state_mapping["phenotype_count"])
            statistic_sum *= batch.variant_count
            statistic_sum += typing.cast("int", chromosome_state_mapping["prediction_row_count"])
            statistic_sum += batch.variant_offset
            return _core.NativeAssociationBatchResult(batch.chromosome, batch.variant_count, float(statistic_sum))

    class RecordingEngineRunEffects:
        def __init__(self) -> None:
            self.calls: list[str] = []
            self.written_results: list[tuple[str, int, float]] = []

        def emit_phase_event(self, phase: str) -> None:
            self.calls.append(f"phase:{phase}")

        def write_batch_result(self, result: _core.NativeAssociationBatchResult) -> None:
            self.calls.append("write_batch_result")
            self.written_results.append((result.chromosome, result.variant_count, result.statistic_sum))

        def drain_writers(self) -> None:
            self.calls.append("drain_writers")

        def finalize_outputs(self) -> None:
            self.calls.append("finalize_outputs")

    python_effects = RecordingEngineRunEffects()
    native_backend = _core.NativePythonAssociationBackend(CoordinatorPythonAssociationBackend())
    native_effects = _core.NativePythonEngineRunEffects(python_effects)

    report = native_backend.run_single_batch_with_effects(
        "binary",
        2,
        "chr2",
        "chr2",
        5,
        "chr2",
        4,
        3,
        native_effects,
    )

    assert report.result.statistic_sum == 16.0
    assert python_effects.calls == [
        "phase:inputs_opened",
        "phase:inputs_aligned",
        "phase:preflight_validated",
        "phase:outputs_initialized",
        "phase:running",
        "write_batch_result",
        "phase:draining",
        "drain_writers",
        "phase:finalizing",
        "finalize_outputs",
        "phase:completed",
    ]
    assert python_effects.written_results == [("chr2", 4, 16.0)]


def test_native_python_association_backend_runs_native_coordinator_chromosome_batches() -> None:
    class CoordinatorPythonAssociationBackend:
        def __init__(self) -> None:
            self.call_names: list[str] = []
            self.batch_offsets: list[int] = []

        def prepare_group(self, group_input: _core.NativePreparedGroupInput) -> dict[str, object]:
            self.call_names.append("prepare_group")
            return {"group_identifier": group_input.group_identifier, "phenotype_count": group_input.phenotype_count}

        def prepare_chromosome(
            self,
            group_state: object,
            chromosome: str,
            predictions: _core.NativePredictionView,
        ) -> dict[str, object]:
            self.call_names.append("prepare_chromosome")
            group_state_mapping = typing.cast("dict[str, object]", group_state)
            return {
                "group_identifier": group_state_mapping["group_identifier"],
                "phenotype_count": group_state_mapping["phenotype_count"],
                "chromosome": chromosome,
                "prediction_row_count": predictions.row_count,
            }

        def compute_batch(
            self,
            chromosome_state: object,
            batch: _core.NativeGenotypeBatchView,
        ) -> _core.NativeAssociationBatchResult:
            self.call_names.append("compute_batch")
            self.batch_offsets.append(batch.variant_offset)
            chromosome_state_mapping = typing.cast("dict[str, object]", chromosome_state)
            statistic_sum = typing.cast("int", chromosome_state_mapping["phenotype_count"])
            statistic_sum *= batch.variant_count
            statistic_sum += typing.cast("int", chromosome_state_mapping["prediction_row_count"])
            statistic_sum += batch.variant_offset
            return _core.NativeAssociationBatchResult(batch.chromosome, batch.variant_count, float(statistic_sum))

    python_backend = CoordinatorPythonAssociationBackend()
    native_backend = _core.NativePythonAssociationBackend(python_backend)

    report = native_backend.run_chromosome_batches(
        "binary",
        2,
        "chr2",
        "chr2",
        5,
        [_core.NativeGenotypeBatchView("chr2", 4, 3), _core.NativeGenotypeBatchView("chr2", 2, 7)],
    )

    assert python_backend.call_names == ["prepare_group", "prepare_chromosome", "compute_batch", "compute_batch"]
    assert python_backend.batch_offsets == [3, 7]
    assert report.phase_history == [
        "planned",
        "inputs_opened",
        "inputs_aligned",
        "preflight_validated",
        "outputs_initialized",
        "running",
        "draining",
        "finalizing",
        "completed",
    ]
    assert [(result.chromosome, result.variant_count, result.statistic_sum) for result in report.results] == [
        ("chr2", 4, 16.0),
        ("chr2", 2, 16.0),
    ]


def test_native_python_association_backend_runs_chromosome_batches_with_effects() -> None:
    class CoordinatorPythonAssociationBackend:
        def prepare_group(self, group_input: _core.NativePreparedGroupInput) -> dict[str, object]:
            return {"phenotype_count": group_input.phenotype_count}

        def prepare_chromosome(
            self,
            group_state: object,
            chromosome: str,
            predictions: _core.NativePredictionView,
        ) -> dict[str, object]:
            group_state_mapping = typing.cast("dict[str, object]", group_state)
            return {
                "phenotype_count": group_state_mapping["phenotype_count"],
                "chromosome": chromosome,
                "prediction_row_count": predictions.row_count,
            }

        def compute_batch(
            self,
            chromosome_state: object,
            batch: _core.NativeGenotypeBatchView,
        ) -> _core.NativeAssociationBatchResult:
            chromosome_state_mapping = typing.cast("dict[str, object]", chromosome_state)
            statistic_sum = typing.cast("int", chromosome_state_mapping["phenotype_count"])
            statistic_sum *= batch.variant_count
            statistic_sum += typing.cast("int", chromosome_state_mapping["prediction_row_count"])
            statistic_sum += batch.variant_offset
            return _core.NativeAssociationBatchResult(batch.chromosome, batch.variant_count, float(statistic_sum))

    class RecordingEngineRunEffects:
        def __init__(self) -> None:
            self.written_results: list[tuple[str, int, float]] = []

        def write_batch_result(self, result: _core.NativeAssociationBatchResult) -> None:
            self.written_results.append((result.chromosome, result.variant_count, result.statistic_sum))

    python_effects = RecordingEngineRunEffects()
    native_backend = _core.NativePythonAssociationBackend(CoordinatorPythonAssociationBackend())
    native_effects = _core.NativePythonEngineRunEffects(python_effects)

    report = native_backend.run_chromosome_batches_with_effects(
        "binary",
        2,
        "chr2",
        "chr2",
        5,
        [_core.NativeGenotypeBatchView("chr2", 4, 3), _core.NativeGenotypeBatchView("chr2", 2, 7)],
        native_effects,
    )

    assert [(result.chromosome, result.variant_count, result.statistic_sum) for result in report.results] == [
        ("chr2", 4, 16.0),
        ("chr2", 2, 16.0),
    ]
    assert python_effects.written_results == [
        ("chr2", 4, 16.0),
        ("chr2", 2, 16.0),
    ]


def test_native_python_association_backend_runs_native_coordinator_group_chromosomes() -> None:
    class CoordinatorPythonAssociationBackend:
        def __init__(self) -> None:
            self.call_names: list[str] = []
            self.chromosomes: list[str] = []
            self.batch_offsets: list[int] = []

        def prepare_group(self, group_input: _core.NativePreparedGroupInput) -> dict[str, object]:
            self.call_names.append("prepare_group")
            return {"group_identifier": group_input.group_identifier, "phenotype_count": group_input.phenotype_count}

        def prepare_chromosome(
            self,
            group_state: object,
            chromosome: str,
            predictions: _core.NativePredictionView,
        ) -> dict[str, object]:
            self.call_names.append("prepare_chromosome")
            self.chromosomes.append(chromosome)
            group_state_mapping = typing.cast("dict[str, object]", group_state)
            return {
                "group_identifier": group_state_mapping["group_identifier"],
                "phenotype_count": group_state_mapping["phenotype_count"],
                "chromosome": chromosome,
                "prediction_row_count": predictions.row_count,
            }

        def compute_batch(
            self,
            chromosome_state: object,
            batch: _core.NativeGenotypeBatchView,
        ) -> _core.NativeAssociationBatchResult:
            self.call_names.append("compute_batch")
            self.batch_offsets.append(batch.variant_offset)
            chromosome_state_mapping = typing.cast("dict[str, object]", chromosome_state)
            statistic_sum = typing.cast("int", chromosome_state_mapping["phenotype_count"])
            statistic_sum *= batch.variant_count
            statistic_sum += typing.cast("int", chromosome_state_mapping["prediction_row_count"])
            statistic_sum += batch.variant_offset
            return _core.NativeAssociationBatchResult(batch.chromosome, batch.variant_count, float(statistic_sum))

    python_backend = CoordinatorPythonAssociationBackend()
    native_backend = _core.NativePythonAssociationBackend(python_backend)
    first_chromosome_input = _core.NativeAssociationChromosomeRunInput(
        "chr2",
        "chr2",
        5,
        [_core.NativeGenotypeBatchView("chr2", 4, 3), _core.NativeGenotypeBatchView("chr2", 2, 7)],
    )
    second_chromosome_input = _core.NativeAssociationChromosomeRunInput(
        "chr3",
        "chr3",
        8,
        [_core.NativeGenotypeBatchView("chr3", 3, 1)],
    )

    report = native_backend.run_group_chromosomes(
        "binary",
        2,
        [first_chromosome_input, second_chromosome_input],
    )

    assert first_chromosome_input.chromosome == "chr2"
    assert first_chromosome_input.prediction_chromosome == "chr2"
    assert first_chromosome_input.prediction_row_count == 5
    assert [
        (batch.chromosome, batch.variant_count, batch.variant_offset) for batch in first_chromosome_input.batches
    ] == [
        ("chr2", 4, 3),
        ("chr2", 2, 7),
    ]
    assert python_backend.call_names == [
        "prepare_group",
        "prepare_chromosome",
        "compute_batch",
        "compute_batch",
        "prepare_chromosome",
        "compute_batch",
    ]
    assert python_backend.chromosomes == ["chr2", "chr3"]
    assert python_backend.batch_offsets == [3, 7, 1]
    assert report.phase_history == [
        "planned",
        "inputs_opened",
        "inputs_aligned",
        "preflight_validated",
        "outputs_initialized",
        "running",
        "draining",
        "finalizing",
        "completed",
    ]
    assert [(result.chromosome, result.variant_count, result.statistic_sum) for result in report.results] == [
        ("chr2", 4, 16.0),
        ("chr2", 2, 16.0),
        ("chr3", 3, 15.0),
    ]


def test_native_python_association_backend_runs_group_chromosomes_with_effects() -> None:
    class CoordinatorPythonAssociationBackend:
        def __init__(self) -> None:
            self.call_names: list[str] = []

        def prepare_group(self, group_input: _core.NativePreparedGroupInput) -> dict[str, object]:
            self.call_names.append("prepare_group")
            return {"phenotype_count": group_input.phenotype_count}

        def prepare_chromosome(
            self,
            group_state: object,
            chromosome: str,
            predictions: _core.NativePredictionView,
        ) -> dict[str, object]:
            self.call_names.append(f"prepare_chromosome:{chromosome}")
            group_state_mapping = typing.cast("dict[str, object]", group_state)
            return {
                "phenotype_count": group_state_mapping["phenotype_count"],
                "prediction_row_count": predictions.row_count,
            }

        def compute_batch(
            self,
            chromosome_state: object,
            batch: _core.NativeGenotypeBatchView,
        ) -> _core.NativeAssociationBatchResult:
            self.call_names.append(f"compute_batch:{batch.chromosome}:{batch.variant_offset}")
            chromosome_state_mapping = typing.cast("dict[str, object]", chromosome_state)
            statistic_sum = typing.cast("int", chromosome_state_mapping["phenotype_count"])
            statistic_sum *= batch.variant_count
            statistic_sum += typing.cast("int", chromosome_state_mapping["prediction_row_count"])
            statistic_sum += batch.variant_offset
            return _core.NativeAssociationBatchResult(batch.chromosome, batch.variant_count, float(statistic_sum))

    class RecordingEngineRunEffects:
        def __init__(self) -> None:
            self.calls: list[str] = []
            self.written_results: list[tuple[str, int, float]] = []

        def emit_phase_event(self, phase: str) -> None:
            self.calls.append(f"phase:{phase}")

        def open_inputs(self) -> None:
            self.calls.append("open_inputs")

        def align_inputs(self) -> None:
            self.calls.append("align_inputs")

        def validate_preflight(self) -> None:
            self.calls.append("validate_preflight")

        def validate_output_compatibility(self) -> None:
            self.calls.append("validate_output_compatibility")

        def construct_writers(self) -> None:
            self.calls.append("construct_writers")

        def write_batch_result(self, result: _core.NativeAssociationBatchResult) -> None:
            self.calls.append("write_batch_result")
            self.written_results.append((result.chromosome, result.variant_count, result.statistic_sum))

        def drain_writers(self) -> None:
            self.calls.append("drain_writers")

        def finalize_outputs(self) -> None:
            self.calls.append("finalize_outputs")

    python_backend = CoordinatorPythonAssociationBackend()
    python_effects = RecordingEngineRunEffects()
    native_backend = _core.NativePythonAssociationBackend(python_backend)
    native_effects = _core.NativePythonEngineRunEffects(python_effects)
    chromosome_inputs = [
        _core.NativeAssociationChromosomeRunInput(
            "chr2",
            "chr2",
            5,
            [_core.NativeGenotypeBatchView("chr2", 4, 3), _core.NativeGenotypeBatchView("chr2", 2, 7)],
        ),
        _core.NativeAssociationChromosomeRunInput(
            "chr3",
            "chr3",
            8,
            [_core.NativeGenotypeBatchView("chr3", 3, 1)],
        ),
    ]

    report = native_backend.run_group_chromosomes_with_effects("binary", 2, chromosome_inputs, native_effects)

    assert python_backend.call_names == [
        "prepare_group",
        "prepare_chromosome:chr2",
        "compute_batch:chr2:3",
        "compute_batch:chr2:7",
        "prepare_chromosome:chr3",
        "compute_batch:chr3:1",
    ]
    assert python_effects.calls == [
        "phase:inputs_opened",
        "open_inputs",
        "phase:inputs_aligned",
        "align_inputs",
        "phase:preflight_validated",
        "validate_preflight",
        "phase:outputs_initialized",
        "validate_output_compatibility",
        "construct_writers",
        "phase:running",
        "write_batch_result",
        "write_batch_result",
        "write_batch_result",
        "phase:draining",
        "drain_writers",
        "phase:finalizing",
        "finalize_outputs",
        "phase:completed",
    ]
    assert python_effects.written_results == [
        ("chr2", 4, 16.0),
        ("chr2", 2, 16.0),
        ("chr3", 3, 15.0),
    ]
    assert [(result.chromosome, result.variant_count, result.statistic_sum) for result in report.results] == [
        ("chr2", 4, 16.0),
        ("chr2", 2, 16.0),
        ("chr3", 3, 15.0),
    ]


def test_native_python_association_backend_maps_python_effect_errors() -> None:
    class CoordinatorPythonAssociationBackend:
        def prepare_group(self, group_input: _core.NativePreparedGroupInput) -> dict[str, object]:
            return {"phenotype_count": group_input.phenotype_count}

        def prepare_chromosome(
            self,
            group_state: object,
            chromosome: str,
            predictions: _core.NativePredictionView,
        ) -> dict[str, object]:
            group_state_mapping = typing.cast("dict[str, object]", group_state)
            return {
                "phenotype_count": group_state_mapping["phenotype_count"],
                "prediction_row_count": predictions.row_count,
            }

        def compute_batch(
            self,
            chromosome_state: object,
            batch: _core.NativeGenotypeBatchView,
        ) -> _core.NativeAssociationBatchResult:
            chromosome_state_mapping = typing.cast("dict[str, object]", chromosome_state)
            statistic_sum = typing.cast("int", chromosome_state_mapping["phenotype_count"])
            statistic_sum *= batch.variant_count
            statistic_sum += typing.cast("int", chromosome_state_mapping["prediction_row_count"])
            statistic_sum += batch.variant_offset
            return _core.NativeAssociationBatchResult(batch.chromosome, batch.variant_count, float(statistic_sum))

    class FailingEngineRunEffects:
        def __init__(self) -> None:
            self.aborted_phases: list[str] = []

        def write_batch_result(self, result: _core.NativeAssociationBatchResult) -> None:
            raise ValueError(f"writer exploded for {result.chromosome}")

        def abort_outputs(self, phase: str) -> None:
            self.aborted_phases.append(phase)

    python_effects = FailingEngineRunEffects()
    native_backend = _core.NativePythonAssociationBackend(CoordinatorPythonAssociationBackend())
    native_effects = _core.NativePythonEngineRunEffects(python_effects)
    chromosome_inputs = [
        _core.NativeAssociationChromosomeRunInput(
            "chr2",
            "chr2",
            5,
            [_core.NativeGenotypeBatchView("chr2", 4, 3)],
        )
    ]

    with pytest.raises(RuntimeError, match="writer exploded for chr2"):
        native_backend.run_group_chromosomes_with_effects("binary", 2, chromosome_inputs, native_effects)

    assert python_effects.aborted_phases == ["running"]


def test_native_python_association_backend_maps_python_errors() -> None:
    class FailingPythonAssociationBackend:
        def prepare_group(self, group_input: _core.NativePreparedGroupInput) -> object:
            raise ValueError(f"planned backend failure for {group_input.group_identifier}")

    native_backend = _core.NativePythonAssociationBackend(FailingPythonAssociationBackend())

    with pytest.raises(RuntimeError, match="planned backend failure for binary"):
        native_backend.prepare_group("binary", 2)


def test_native_python_association_backend_requires_native_batch_result() -> None:
    class InvalidResultPythonAssociationBackend:
        def prepare_group(self, group_input: _core.NativePreparedGroupInput) -> object:
            return {"group_identifier": group_input.group_identifier}

        def prepare_chromosome(
            self,
            group_state: object,
            chromosome: str,
            predictions: _core.NativePredictionView,
        ) -> object:
            return {"group_state": group_state, "chromosome": chromosome, "predictions": predictions}

        def compute_batch(self, chromosome_state: object, batch: _core.NativeGenotypeBatchView) -> object:
            return {"chromosome_state": chromosome_state, "batch": batch}

    native_backend = _core.NativePythonAssociationBackend(InvalidResultPythonAssociationBackend())
    group_state = native_backend.prepare_group("linear", 1)
    chromosome_state = native_backend.prepare_chromosome(group_state, "chr1", "chr1", 3)

    with pytest.raises(RuntimeError, match="NativeAssociationBatchResult"):
        native_backend.compute_batch(chromosome_state, "chr1", 2, 0)


def test_native_preflight_shape_payloads_validate_deterministic_policy() -> None:
    native_preflight_validator = _core.NativePreflightValidator()

    single_payload = native_preflight_validator.validate_single_trait_preflight_shape_payload(
        phenotype_sample_count=3,
        covariate_dimension_count=2,
        covariate_sample_count=3,
        covariate_count=2,
    )
    assert single_payload == {"sample_count": 3, "covariate_count": 2}

    multi_payload = native_preflight_validator.validate_multi_trait_preflight_shape_payload(
        phenotype_dimension_count=2,
        phenotype_trait_count=2,
        phenotype_sample_count=3,
        covariate_dimension_count=2,
        covariate_sample_count=3,
        covariate_count=2,
    )
    assert multi_payload == {"trait_count": 2, "sample_count": 3, "covariate_count": 2}

    with pytest.raises(ValueError, match="Covariate matrix must be two-dimensional"):
        native_preflight_validator.validate_single_trait_preflight_shape_payload(
            phenotype_sample_count=3,
            covariate_dimension_count=1,
            covariate_sample_count=3,
            covariate_count=0,
        )

    with pytest.raises(ValueError, match="Phenotype matrix must contain at least one trait"):
        native_preflight_validator.validate_multi_trait_preflight_shape_payload(
            phenotype_dimension_count=2,
            phenotype_trait_count=0,
            phenotype_sample_count=3,
            covariate_dimension_count=2,
            covariate_sample_count=3,
            covariate_count=2,
        )


def test_native_preflight_binary_and_prediction_shape_policy() -> None:
    native_preflight_validator = _core.NativePreflightValidator()

    native_preflight_validator.validate_finite_array_values("Phenotype", np.asarray([0.0, 1.0], dtype=np.float32))
    native_preflight_validator.validate_finite_array_values("Integer phenotype", np.asarray([0, 1], dtype=np.int64))
    native_preflight_validator.validate_covariate_matrix_rank(covariate_rank=2, covariate_count=2)
    native_preflight_validator.validate_covariate_matrix_rank_array(
        np.asarray([[1.0, 0.0], [1.0, 1.0], [1.0, 2.0]], dtype=np.float32),
        covariate_count=2,
    )
    native_preflight_validator.validate_binary_phenotype_array(np.asarray([0.0, 1.0, 1.0], dtype=np.float64))
    native_preflight_validator.validate_binary_phenotype_array(np.asarray([False, True, True], dtype=np.bool_))
    native_preflight_validator.validate_single_prediction_preflight_shape("1", (3,), sample_count=3)
    native_preflight_validator.validate_multi_prediction_preflight_shape("2", (2, 3), trait_count=2, sample_count=3)

    with pytest.raises(ValueError, match="Phenotype contains non-finite values"):
        native_preflight_validator.validate_finite_array_values(
            "Phenotype", np.asarray([0.0, np.nan], dtype=np.float64)
        )

    with pytest.raises(ValueError, match="Covariate matrix is rank deficient"):
        native_preflight_validator.validate_covariate_matrix_rank(covariate_rank=1, covariate_count=2)

    with pytest.raises(ValueError, match="Covariate matrix is rank deficient"):
        native_preflight_validator.validate_covariate_matrix_rank_array(
            np.asarray([[1.0, 2.0], [1.0, 2.0], [1.0, 2.0]], dtype=np.float64),
            covariate_count=2,
        )

    with pytest.raises(ValueError, match="Binary phenotype must be coded as 0/1 after alignment"):
        native_preflight_validator.validate_binary_phenotype_array(np.asarray([0.0, 0.5, 1.0], dtype=np.float32))

    with pytest.raises(ValueError, match="Binary phenotype must contain at least one case and one control"):
        native_preflight_validator.validate_binary_phenotype_array(np.asarray([0, 0, 0], dtype=np.int32))

    with pytest.raises(ValueError, match="Prediction sample count for chromosome 1 is 2, expected 3"):
        native_preflight_validator.validate_single_prediction_preflight_shape("1", (2,), sample_count=3)

    with pytest.raises(
        ValueError,
        match=r"Prediction matrix shape for chromosome 2 is \(2, 2\), expected \(2, 3\)",
    ):
        native_preflight_validator.validate_multi_prediction_preflight_shape("2", (2, 2), trait_count=2, sample_count=3)


def test_detached_native_preflight_helpers_removed_from_root_surface() -> None:
    removed_helper_names = (
        "build_preflight_report_payload",
        "resolve_preflight_variant_count",
        "validate_binary_phenotype_array",
        "validate_covariate_matrix_rank",
        "validate_covariate_matrix_rank_array",
        "validate_finite_array_values",
        "validate_multi_prediction_preflight_shape",
        "validate_multi_trait_preflight_shape_payload",
        "validate_single_prediction_preflight_shape",
        "validate_single_trait_preflight_shape_payload",
    )

    for helper_name in removed_helper_names:
        assert not hasattr(_core, helper_name)


def test_native_preflight_covariate_rank_array_uses_numpy_default_tolerance() -> None:
    native_preflight_validator = _core.NativePreflightValidator()
    tiny_float32_singular_value = np.finfo(np.float32).eps
    covariate_matrix = np.asarray(
        [
            [1.0, 0.0],
            [0.0, tiny_float32_singular_value],
            [0.0, 0.0],
        ],
        dtype=np.float32,
    )

    with pytest.raises(ValueError, match="Covariate matrix is rank deficient"):
        native_preflight_validator.validate_covariate_matrix_rank_array(covariate_matrix, covariate_count=2)

    native_preflight_validator.validate_covariate_matrix_rank_array(
        covariate_matrix.astype(np.float64), covariate_count=2
    )


def test_native_pipeline_resume_compatibility_validates_all_manifests(tmp_path: Path) -> None:
    run_directory = tmp_path / "run"
    chunks_directory = tmp_path / "chunks"
    chunks_directory.mkdir()
    manifest: dict[str, object] = {"schema_version": 7, "chunk_size": 32, "committed_chunks": []}
    current_header: dict[str, object] = {"schema_version": 7, "chunk_size": 32}

    def validate_resume_compatibility(
        *,
        existing_manifest_values: tuple[dict[str, object] | None, ...],
        current_header_values: tuple[dict[str, object], ...],
        resume_mode: str,
    ) -> None:
        preparation_batch = _core.build_pipeline_output_preparation_batch_from_values(
            run_directories=(str(run_directory),),
            chunks_directories=(str(chunks_directory),),
            existing_manifest_values=existing_manifest_values,
            current_header_values=current_header_values,
            resume=True,
            resume_mode=resume_mode,
        )
        preparation_batch.validate_resume_compatibility()

    validate_resume_compatibility(
        existing_manifest_values=(manifest,),
        current_header_values=(current_header,),
        resume_mode="fast",
    )
    validate_resume_compatibility(
        existing_manifest_values=(manifest,),
        current_header_values=(current_header,),
        resume_mode="strict",
    )

    with pytest.raises(ValueError, match=r"Resume requires run_manifest\.json"):
        validate_resume_compatibility(
            existing_manifest_values=(None,),
            current_header_values=(current_header,),
            resume_mode="fast",
        )

    with pytest.raises(ValueError, match="input counts must match"):
        validate_resume_compatibility(
            existing_manifest_values=(),
            current_header_values=(current_header,),
            resume_mode="fast",
        )

    incompatible_header: dict[str, object] = {"schema_version": 7, "chunk_size": 64}
    with pytest.raises(ValueError, match="chunk_size"):
        validate_resume_compatibility(
            existing_manifest_values=(manifest,),
            current_header_values=(incompatible_header,),
            resume_mode="fast",
        )


def test_native_pipeline_output_initialization_returns_committed_sets(tmp_path: Path) -> None:
    run_directory = tmp_path / "run"
    chunks_directory = run_directory / "chunks"
    chunks_directory.mkdir(parents=True)
    existing_manifest: dict[str, object] = {
        "schema_version": 7,
        "chunk_size": 32,
        "committed_chunks": [
            {
                "chunk_identifier": 2,
                "variant_start_index": 2,
                "variant_stop_index": 4,
                "row_count": 2,
                "chunk_file_name": "chunk_2.arrow",
            }
        ],
    }
    current_header: dict[str, object] = {"schema_version": 7, "chunk_size": 32}

    preparation_batch = _core.build_pipeline_output_preparation_batch_from_values(
        run_directories=(str(run_directory),),
        chunks_directories=(str(chunks_directory),),
        existing_manifest_values=(existing_manifest,),
        current_header_values=(current_header,),
        resume=True,
        resume_mode="fast",
    )
    native_initialization = preparation_batch.initialize(build_native_runtime_compatibility_token())
    committed_chunk_identifier_sets = native_initialization.committed_chunk_identifier_sets()

    assert committed_chunk_identifier_sets == [[2]]
    written_manifest = json.loads((run_directory / "run_manifest.json").read_text(encoding="utf-8"))
    assert written_manifest["committed_chunks"][0]["chunk_identifier"] == 2


def test_native_pipeline_output_initialization_handle_returns_committed_sets(tmp_path: Path) -> None:
    run_directory = tmp_path / "run"
    chunks_directory = run_directory / "chunks"
    chunks_directory.mkdir(parents=True)
    existing_manifest: dict[str, object] = {
        "schema_version": 7,
        "chunk_size": 32,
        "committed_chunks": [
            {
                "chunk_identifier": 2,
                "variant_start_index": 2,
                "variant_stop_index": 4,
                "row_count": 2,
                "chunk_file_name": "chunk_2.arrow",
            }
        ],
    }
    current_header: dict[str, object] = {"schema_version": 7, "chunk_size": 32}

    preparation_batch = _core.build_pipeline_output_preparation_batch_from_values(
        run_directories=(str(run_directory),),
        chunks_directories=(str(chunks_directory),),
        existing_manifest_values=(existing_manifest,),
        current_header_values=(current_header,),
        resume=True,
        resume_mode="fast",
    )
    native_initialization = preparation_batch.initialize(build_native_runtime_compatibility_token())

    assert isinstance(native_initialization, _core.NativePipelineOutputInitialization)
    assert native_initialization.output_count == 1
    assert native_initialization.committed_chunk_identifier_sets() == [[2]]
    assert native_initialization.committed_chunk_identifiers(0) == [2]
    with pytest.raises(ValueError, match="Output index 1 is out of range"):
        native_initialization.committed_chunk_identifiers(1)


def test_native_pipeline_output_preparation_batch_initializes_outputs(tmp_path: Path) -> None:
    run_directory = tmp_path / "run"
    chunks_directory = run_directory / "chunks"
    chunks_directory.mkdir(parents=True)
    existing_manifest: dict[str, object] = {
        "schema_version": 7,
        "chunk_size": 32,
        "committed_chunks": [
            {
                "chunk_identifier": 2,
                "variant_start_index": 2,
                "variant_stop_index": 4,
                "row_count": 2,
                "chunk_file_name": "chunk_2.arrow",
            }
        ],
    }
    current_header: dict[str, object] = {"schema_version": 7, "chunk_size": 32}
    native_preparation_batch = _core.build_pipeline_output_preparation_batch_from_values(
        run_directories=(str(run_directory),),
        chunks_directories=(str(chunks_directory),),
        existing_manifest_values=(existing_manifest,),
        current_header_values=(current_header,),
        resume=True,
        resume_mode="fast",
    )

    native_preparation_batch.validate_resume_compatibility()
    native_initialization = native_preparation_batch.initialize(build_native_runtime_compatibility_token())

    assert native_preparation_batch.output_count == 1
    assert native_preparation_batch.resume is True
    assert isinstance(native_initialization, _core.NativePipelineOutputInitialization)
    assert native_initialization.committed_chunk_identifiers(0) == [2]


def test_native_effective_trusted_no_missing_diploid_policy() -> None:
    assert not _core.resolve_effective_trusted_no_missing_diploid(
        requested_trusted_no_missing_diploid=False,
        variant_major_packed8_probability_pairs=False,
    )
    assert _core.resolve_effective_trusted_no_missing_diploid(
        requested_trusted_no_missing_diploid=True,
        variant_major_packed8_probability_pairs=False,
    )
    assert _core.resolve_effective_trusted_no_missing_diploid(
        requested_trusted_no_missing_diploid=False,
        variant_major_packed8_probability_pairs=True,
    )


def test_native_null_logistic_nonconvergence_policy() -> None:
    continue_plan = _core.plan_null_logistic_nonconvergence_from_array(
        chromosome="22",
        convergence_values=np.asarray(1, dtype=np.bool_),
        phenotype_names=None,
        policy="fail",
    )
    assert continue_plan.action == "continue"
    assert continue_plan.failed_trait_indices == []
    assert continue_plan.message is None
    assert continue_plan.warning_message is None
    assert continue_plan.nonconverged_count == 0
    assert continue_plan.scalar_convergence is True
    assert continue_plan.total_fit_count == 1

    fail_plan = _core.plan_null_logistic_nonconvergence_from_array(
        chromosome="22",
        convergence_values=np.asarray(0, dtype=np.bool_),
        phenotype_names=None,
        policy="fail",
    )
    assert fail_plan.action == "fail"
    assert fail_plan.failed_trait_indices == [0]
    assert fail_plan.message == "Binary null logistic model did not converge for chromosome 22."
    assert fail_plan.warning_message is None
    assert fail_plan.nonconverged_count == 1
    assert fail_plan.scalar_convergence is True
    assert fail_plan.total_fit_count == 1

    warn_plan = _core.plan_null_logistic_nonconvergence_from_array(
        chromosome="22",
        convergence_values=np.asarray([True, False], dtype=np.bool_),
        phenotype_names=("trait_a", "trait_b"),
        policy="warn",
    )
    assert warn_plan.action == "warn"
    assert warn_plan.failed_trait_indices == [1]
    assert warn_plan.message == "Binary null logistic model did not converge for chromosome 22: trait_b."
    assert warn_plan.warning_message == (
        "Binary null logistic model did not converge for chromosome 22: trait_b. "
        "Continuing because --null_logistic_nonconvergence_policy=warn."
    )
    assert warn_plan.nonconverged_count == 1
    assert warn_plan.scalar_convergence is False
    assert warn_plan.total_fit_count == 2

    array_plan = _core.plan_null_logistic_nonconvergence_from_array(
        chromosome="22",
        convergence_values=np.asarray([True, False, False], dtype=np.bool_),
        phenotype_names=("trait_a", "trait_b", "trait_c"),
        policy="warn",
    )
    assert array_plan.action == "warn"
    assert array_plan.failed_trait_indices == [1, 2]
    assert array_plan.nonconverged_count == 2
    assert array_plan.scalar_convergence is False
    assert array_plan.total_fit_count == 3

    with pytest.raises(ValueError, match="Unsupported null logistic nonconvergence policy"):
        _core.plan_null_logistic_nonconvergence_from_array(
            chromosome="22",
            convergence_values=np.asarray(0, dtype=np.bool_),
            phenotype_names=None,
            policy="ignore",
        )

    with pytest.raises(ValueError, match="bool dtype"):
        _core.plan_null_logistic_nonconvergence_from_array(
            chromosome="22",
            convergence_values=np.asarray([0, 1], dtype=np.int32),
            phenotype_names=None,
            policy="warn",
        )


def test_native_runtime_state_issues_compatibility_token() -> None:
    runtime_state = _core.NativeRuntimeState()
    logging_policy_payload = runtime_state.build_logging_runtime_policy_payload(
        log_filter="info",
        log_file=None,
        log_stderr=False,
        log_queue_size=1024,
        log_lossy=True,
        include_source_location=False,
        include_span_events=False,
        trace_file=None,
        trace_filter="info",
        trace_event_cap=None,
        telemetry_mode="off",
        telemetry_stream_file=None,
    )
    jax_policy_payload: dict[str, object] = {
        "device": "cpu",
        "cache_directory": None,
        "matmul_precision": None,
        "persistent_cache": True,
        "persistent_cache_min_entry_size_bytes": 0,
        "persistent_cache_min_compile_time_seconds": 0,
        "xla_autotune_cache": False,
        "transfer_guard": False,
    }

    runtime_token = runtime_state.require_compatible_runtime_policy(
        logging_policy_payload,
        None,
        jax_policy_payload,
    )
    runtime_policy = runtime_state.build_runtime_policy_handle(logging_policy_payload, None, jax_policy_payload)
    runtime_token_from_policy_handle = runtime_state.require_compatible_runtime_policy_handle(runtime_policy)
    run_runtime = runtime_state.build_run_runtime(runtime_policy)

    assert isinstance(runtime_token, _core.NativeRuntimeCompatibilityToken)
    assert isinstance(runtime_policy, _core.NativeRuntimePolicy)
    assert isinstance(runtime_token_from_policy_handle, _core.NativeRuntimeCompatibilityToken)
    assert isinstance(run_runtime, _core.NativeRunRuntime)
    assert isinstance(run_runtime.runtime_compatibility_token(), _core.NativeRuntimeCompatibilityToken)
    assert runtime_policy.rayon_thread_count is None
    assert runtime_policy.logging_runtime_policy_payload() == logging_policy_payload
    assert runtime_policy.jax_runtime_policy_payload() == jax_policy_payload
    assert run_runtime.rayon_thread_count is None
    assert run_runtime.logging_runtime_policy_payload() == logging_policy_payload
    assert run_runtime.jax_runtime_policy_payload() == jax_policy_payload
    assert not hasattr(_core, "build_runtime_policy_handle")
    assert not hasattr(_core, "describe_logging_runtime_policy_value")

    runtime_state.record_jax_runtime_policy({**jax_policy_payload, "cache_directory": "/tmp/first-cache"})
    with pytest.raises(RuntimeError, match="JAX runtime is already configured"):
        runtime_state.require_compatible_runtime_policy(
            logging_policy_payload,
            None,
            {**jax_policy_payload, "cache_directory": "/tmp/second-cache"},
        )


def test_native_cli_run_lifecycle_state_tracks_runner_started() -> None:
    cli_lifecycle_state = _core.NativeCliRunLifecycleState()

    assert cli_lifecycle_state.runner_started is False

    cli_lifecycle_state.mark_runner_started()

    assert cli_lifecycle_state.runner_started is True
    assert not hasattr(_core, "NativeCliRunFailureTelemetryPlan")
    assert not hasattr(_core, "emit_cli_run_failed_telemetry_event")
    assert not hasattr(_core, "plan_cli_telemetry_close_failure")


def test_native_cli_run_failed_telemetry_emission() -> None:
    class RecordingTelemetrySession:
        def __init__(self) -> None:
            self.native_telemetry_session = self
            self.events: list[object] = []

        def emit_run_failed_event(self, event: object) -> None:
            self.events.append(event)

    class FailingTelemetrySession:
        def __init__(self) -> None:
            self.native_telemetry_session = self
            self.call_count = 0

        def emit_run_failed_event(self, event: object) -> None:
            del event
            self.call_count += 1
            raise RuntimeError("telemetry write failed")

    class DisabledTelemetrySession:
        native_telemetry_session = None

    class LegacyTelemetrySession:
        def __init__(self) -> None:
            self.call_count = 0

        def log_run_failed(self, event: object) -> None:
            del event
            self.call_count += 1

    failed_event = object()
    recording_session = RecordingTelemetrySession()
    failing_session = FailingTelemetrySession()
    legacy_session = LegacyTelemetrySession()
    cli_lifecycle_state = _core.NativeCliRunLifecycleState()
    started_cli_lifecycle_state = _core.NativeCliRunLifecycleState()
    started_cli_lifecycle_state.mark_runner_started()

    cli_lifecycle_state.emit_run_failed_telemetry_event(
        None,
        failed_event,
    )
    started_cli_lifecycle_state.emit_run_failed_telemetry_event(
        recording_session,
        failed_event,
    )
    assert recording_session.events == []
    cli_lifecycle_state.emit_run_failed_telemetry_event(
        DisabledTelemetrySession(),
        failed_event,
    )
    cli_lifecycle_state.emit_run_failed_telemetry_event(
        legacy_session,
        failed_event,
    )
    assert legacy_session.call_count == 0

    cli_lifecycle_state.emit_run_failed_telemetry_event(
        recording_session,
        failed_event,
    )
    assert recording_session.events == [failed_event]

    cli_lifecycle_state.emit_run_failed_telemetry_event(
        failing_session,
        failed_event,
    )
    assert failing_session.call_count == 1


def test_native_runner_telemetry_dispatch_helpers() -> None:
    class RecordingNativeSessionHandle:
        def __init__(self) -> None:
            self.calls: list[tuple[str, tuple[object, ...]]] = []

        def emit_run_started_event(
            self,
            association_mode: str,
            trait_type: str,
            phenotype_count: int,
            output_run_root: str,
        ) -> None:
            self.calls.append(("run_started", (association_mode, trait_type, phenotype_count, output_run_root)))

        def emit_run_interrupted_event(self, event: object) -> None:
            self.calls.append(("run_interrupted", (event,)))

        def emit_run_failed_event(self, event: object) -> None:
            self.calls.append(("run_failed", (event,)))

        def emit_run_completed_event(self, event: object) -> None:
            self.calls.append(("run_completed", (event,)))

        def emit_execution_plan_prepared_event(
            self,
            association_mode: str,
            trait_type: str,
            phenotype_count: int,
            chunk_size: int,
            variant_limit: int | None,
            device: str,
        ) -> None:
            self.calls.append(
                (
                    "execution_plan_prepared",
                    (association_mode, trait_type, phenotype_count, chunk_size, variant_limit, device),
                )
            )

        def emit_effective_config_written_event(
            self,
            association_mode: str,
            phenotype: str,
            effective_config: str,
            output_run_directory: str,
        ) -> None:
            self.calls.append(
                (
                    "effective_config_written",
                    (association_mode, phenotype, effective_config, output_run_directory),
                )
            )

        def emit_phenotype_writer_finished_event(
            self,
            association_mode: str,
            phenotype: str,
            final_output_path: str | None,
        ) -> None:
            self.calls.append(("writer_finished", (association_mode, phenotype, final_output_path)))

        def emit_multi_phenotype_writer_finished_event(
            self,
            association_mode: str,
            phenotype_count: int,
            final_output_paths: typing.Sequence[str | None],
        ) -> None:
            self.calls.append(("multi_writer_finished", (association_mode, phenotype_count, tuple(final_output_paths))))

    class RecordingTelemetrySession:
        def __init__(self, native_session_handle: RecordingNativeSessionHandle) -> None:
            self.native_session_handle = native_session_handle

    native_session_handle = RecordingNativeSessionHandle()
    telemetry_session = RecordingTelemetrySession(native_session_handle)

    _core.record_execution_plan_prepared_telemetry_event(
        None,
        "regenie2_linear",
        "quantitative",
        2,
        1024,
        None,
        "gpu",
    )
    assert native_session_handle.calls == []

    interrupted_event = object()
    failed_event = object()
    completed_event = object()
    _core.record_runner_run_started_telemetry_event(
        telemetry_session,
        "regenie2_linear",
        "quantitative",
        2,
        "output.g",
    )
    _core.record_runner_run_interrupted_telemetry_event(telemetry_session, interrupted_event)
    _core.record_runner_run_failed_telemetry_event(telemetry_session, failed_event)
    _core.record_runner_run_completed_telemetry_event(telemetry_session, completed_event)
    _core.record_execution_plan_prepared_telemetry_event(
        telemetry_session,
        "regenie2_linear",
        "quantitative",
        2,
        1024,
        None,
        "gpu",
    )
    _core.record_effective_config_written_telemetry_event(
        telemetry_session,
        "regenie2_linear",
        "height",
        "height/effective_config.toml",
        "height",
    )
    _core.record_writer_finished_telemetry_event(
        telemetry_session,
        "regenie2_linear",
        "height",
        "height.parquet",
    )
    _core.record_multi_writer_finished_telemetry_event(
        telemetry_session,
        "regenie2_binary",
        2,
        ("case_status.parquet", None),
    )

    assert native_session_handle.calls == [
        ("run_started", ("regenie2_linear", "quantitative", 2, "output.g")),
        ("run_interrupted", (interrupted_event,)),
        ("run_failed", (failed_event,)),
        ("run_completed", (completed_event,)),
        ("execution_plan_prepared", ("regenie2_linear", "quantitative", 2, 1024, None, "gpu")),
        ("effective_config_written", ("regenie2_linear", "height", "height/effective_config.toml", "height")),
        ("writer_finished", ("regenie2_linear", "height", "height.parquet")),
        ("multi_writer_finished", ("regenie2_binary", 2, ("case_status.parquet", None))),
    ]


def test_native_cli_telemetry_close_failure_plan() -> None:
    cli_lifecycle_state = _core.NativeCliRunLifecycleState()
    successful_run_plan = cli_lifecycle_state.plan_telemetry_close_failure(
        current_exit_code=0,
        runtime_failure_exit_code=1,
    )
    interrupted_run_plan = cli_lifecycle_state.plan_telemetry_close_failure(
        current_exit_code=130,
        runtime_failure_exit_code=1,
    )

    assert isinstance(successful_run_plan, _core.NativeCliTelemetryCloseFailurePlan)
    assert successful_run_plan.should_report_failure is True
    assert successful_run_plan.exit_code == 1
    assert interrupted_run_plan.should_report_failure is False
    assert interrupted_run_plan.exit_code == 130


def test_native_runtime_state_returns_snapshot_payload() -> None:
    runtime_state = _core.NativeRuntimeState()
    logging_policy_payload = runtime_state.build_logging_runtime_policy_payload(
        log_filter="info",
        log_file=None,
        log_stderr=False,
        log_queue_size=1024,
        log_lossy=True,
        include_source_location=False,
        include_span_events=False,
        trace_file=None,
        trace_filter="info",
        trace_event_cap=None,
        telemetry_mode="off",
        telemetry_stream_file=None,
    )
    jax_policy_payload: dict[str, object] = {
        "device": "cpu",
        "cache_directory": "/tmp/g-jax-cache",
        "matmul_precision": None,
        "persistent_cache": True,
        "persistent_cache_min_entry_size_bytes": 0,
        "persistent_cache_min_compile_time_seconds": 0,
        "xla_autotune_cache": False,
        "transfer_guard": False,
    }

    empty_payload = runtime_state.runtime_state_payload()
    runtime_state.record_logging_runtime_policy(logging_policy_payload)
    runtime_state.record_rayon_thread_count(4)
    runtime_state.record_jax_runtime_policy(jax_policy_payload)
    configured_payload = runtime_state.runtime_state_payload()

    assert empty_payload == {
        "logging_policy": None,
        "rayon_thread_count": None,
        "jax_policy": None,
    }
    assert configured_payload == {
        "logging_policy": logging_policy_payload,
        "rayon_thread_count": 4,
        "jax_policy": jax_policy_payload,
    }
    assert not hasattr(_core, "build_logging_runtime_policy_payload")


def test_native_process_runtime_state_handle_seeds_snapshot_payload() -> None:
    runtime_state_builder = _core.NativeRuntimeState()
    logging_policy_payload = runtime_state_builder.build_logging_runtime_policy_payload(
        log_filter="info",
        log_file=None,
        log_stderr=False,
        log_queue_size=1024,
        log_lossy=True,
        include_source_location=False,
        include_span_events=False,
        trace_file=None,
        trace_filter="info",
        trace_event_cap=None,
        telemetry_mode="off",
        telemetry_stream_file=None,
    )
    jax_policy_payload: dict[str, object] = {
        "device": "cpu",
        "cache_directory": "/tmp/g-jax-cache",
        "matmul_precision": None,
        "persistent_cache": True,
        "persistent_cache_min_entry_size_bytes": 0,
        "persistent_cache_min_compile_time_seconds": 0,
        "xla_autotune_cache": False,
        "transfer_guard": False,
    }

    runtime_state = runtime_state_builder.build_process_runtime_state_handle(
        logging_policy_payload, 4, jax_policy_payload
    )
    empty_runtime_state = runtime_state_builder.build_process_runtime_state_handle(None, None, None)

    assert runtime_state.runtime_state_payload() == {
        "logging_policy": logging_policy_payload,
        "rayon_thread_count": 4,
        "jax_policy": jax_policy_payload,
    }
    assert empty_runtime_state.runtime_state_payload() == {
        "logging_policy": None,
        "rayon_thread_count": None,
        "jax_policy": None,
    }
    assert not hasattr(_core, "build_process_runtime_state_handle")


def test_global_process_runtime_state_is_native_owned_singleton() -> None:
    completed_process = run_logging_subprocess(
        "\n".join(
            [
                "from g import _core",
                "first_state = _core.global_process_runtime_state()",
                "second_state = _core.global_process_runtime_state()",
                "first_state.record_rayon_thread_count(6)",
                "print(second_state.rayon_thread_count)",
            ]
        )
    )

    assert completed_process.stdout.strip() == "6"


def test_native_runtime_state_plans_rayon_thread_pool_configuration() -> None:
    runtime_state = _core.NativeRuntimeState()

    configure_plan = runtime_state.plan_rayon_thread_pool_configuration(4)
    runtime_state.record_rayon_thread_count(4)
    skip_plan = runtime_state.plan_rayon_thread_pool_configuration(4)

    assert isinstance(configure_plan, _core.NativeRayonThreadPoolConfigurationPlan)
    assert configure_plan.should_configure is True
    assert configure_plan.thread_count == 4
    assert skip_plan.should_configure is False
    assert skip_plan.thread_count is None
    configured_skip_plan = runtime_state.configure_rayon_thread_pool(4)
    assert configured_skip_plan.should_configure is False
    assert configured_skip_plan.thread_count is None
    with pytest.raises(ValueError, match="Rayon thread count must be positive"):
        _core.NativeRuntimeState().configure_rayon_thread_pool(0)
    with pytest.raises(RuntimeError, match="Rayon --threads is process-global"):
        runtime_state.plan_rayon_thread_pool_configuration(8)


def test_native_runtime_state_configures_runtime_knobs() -> None:
    runtime_state = _core.NativeRuntimeState()
    runtime_state.record_rayon_thread_count(4)

    no_thread_plan = runtime_state.configure_runtime_knobs(32, None)
    matching_thread_plan = runtime_state.configure_runtime_knobs(32, 4)

    assert no_thread_plan is None
    assert matching_thread_plan is not None
    assert matching_thread_plan.should_configure is False
    assert matching_thread_plan.thread_count is None


def test_native_runtime_state_initializes_logging_runtime_policy_preflight() -> None:
    runtime_state = _core.NativeRuntimeState()
    configured_payload = runtime_state.build_logging_runtime_policy_payload(
        log_filter="info",
        log_file="/tmp/g-first.jsonl",
        log_stderr=False,
        log_queue_size=1024,
        log_lossy=True,
        include_source_location=False,
        include_span_events=False,
        trace_file=None,
        trace_filter="info",
        trace_event_cap=None,
        telemetry_mode="off",
        telemetry_stream_file=None,
    )
    requested_payload = {**configured_payload, "log_file": "/tmp/g-second.jsonl"}

    runtime_state.record_logging_runtime_policy(configured_payload)

    with pytest.raises(RuntimeError, match="Logging runtime policy is process-global"):
        runtime_state.initialize_logging_runtime_policy(requested_payload)

    invalid_payload = {**configured_payload, "log_queue_size": -1}
    with pytest.raises(ValueError, match="log_queue_size must be non-negative"):
        _core.NativeRuntimeState().initialize_logging_runtime_policy(invalid_payload)


def test_native_runtime_state_plans_jax_runtime_setup_lifecycle() -> None:
    runtime_state = _core.NativeRuntimeState()
    jax_policy_payload: dict[str, object] = {
        "device": "cpu",
        "cache_directory": "/tmp/g-jax-cache",
        "matmul_precision": None,
        "persistent_cache": True,
        "persistent_cache_min_entry_size_bytes": 0,
        "persistent_cache_min_compile_time_seconds": 0,
        "xla_autotune_cache": False,
        "transfer_guard": False,
    }

    configure_plan = runtime_state.plan_jax_runtime_setup_lifecycle(jax_policy_payload)
    configure_session = runtime_state.build_jax_runtime_setup_session(jax_policy_payload, "/tmp/g-jax-cache")
    resolving_session = _core.NativeRuntimeState().build_jax_runtime_setup_session_resolving_cache_directory(
        {**jax_policy_payload, "cache_directory": None}
    )
    runtime_state.complete_jax_runtime_setup_session(jax_policy_payload, configure_session)
    skip_plan = runtime_state.plan_jax_runtime_setup_lifecycle(jax_policy_payload)
    skip_session = runtime_state.build_jax_runtime_setup_session(jax_policy_payload, "/tmp/g-jax-cache")

    assert isinstance(configure_plan, _core.NativeJaxRuntimeSetupLifecyclePlan)
    assert configure_plan.should_configure is True
    assert isinstance(configure_session, _core.NativeJaxRuntimeSetupSession)
    assert configure_session.should_configure is True
    assert configure_session.should_validate_gpu is False
    assert configure_session.setup_payload()["cache_directory"] == "/tmp/g-jax-cache"
    resolving_cache_directory = resolving_session.setup_payload()["cache_directory"]
    assert isinstance(resolving_cache_directory, str)
    assert resolving_cache_directory.endswith("/g-jax-cache")
    assert configure_session.diagnostic_event_payloads()[0]["event_name"] == "jax_platform_selected"
    assert not hasattr(configure_session, "side_effect_plan_payload")
    assert not hasattr(configure_session, "config_update_payloads")
    assert skip_plan.should_configure is False
    assert skip_session.should_configure is False
    with pytest.raises(RuntimeError, match="JAX runtime is already configured"):
        runtime_state.plan_jax_runtime_setup_lifecycle({**jax_policy_payload, "cache_directory": "/tmp/other-cache"})
    with pytest.raises(RuntimeError, match="JAX runtime is already configured"):
        runtime_state.build_jax_runtime_setup_session(
            {**jax_policy_payload, "cache_directory": "/tmp/other-cache"},
            "/tmp/other-cache",
        )
    with pytest.raises(RuntimeError, match="JAX runtime is already configured"):
        runtime_state.complete_jax_runtime_setup({**jax_policy_payload, "cache_directory": "/tmp/other-cache"})


def test_native_runtime_state_rejects_pending_jax_setup_session_completion() -> None:
    runtime_state = _core.NativeRuntimeState()
    jax_policy_payload: dict[str, object] = {
        "device": "gpu",
        "cache_directory": "/tmp/g-jax-cache",
        "matmul_precision": None,
        "persistent_cache": True,
        "persistent_cache_min_entry_size_bytes": 0,
        "persistent_cache_min_compile_time_seconds": 0,
        "xla_autotune_cache": False,
        "transfer_guard": False,
    }
    pending_setup_session = runtime_state.build_jax_runtime_setup_session(jax_policy_payload, "/tmp/g-jax-cache")

    with pytest.raises(RuntimeError, match="before GPU validation completes"):
        runtime_state.complete_jax_runtime_setup_session(jax_policy_payload, pending_setup_session)


def test_native_jax_runtime_setup_session_completes_validation() -> None:
    native_setup_session = build_native_jax_runtime_setup_session(
        requested_device="gpu",
        cache_directory="/tmp/g-jax-cache",
        persistent_cache=True,
    )

    completed_payload = native_setup_session.complete_validation_payload("succeeded", "gpu ready")
    diagnostic_payloads = native_setup_session.diagnostic_event_payloads()
    gpu_validation_fields = typing.cast("tuple[dict[str, object], ...]", diagnostic_payloads[-1]["fields"])

    assert native_setup_session.should_configure is True
    assert completed_payload["gpu_validation_status"] == "succeeded"
    assert native_setup_session.setup_payload()["gpu_validation_message"] == "gpu ready"
    assert gpu_validation_fields[0]["value"] == "succeeded"


def test_native_jax_runtime_setup_session_applies_config_updates() -> None:
    native_setup_session = build_native_jax_runtime_setup_session(
        requested_device="gpu",
        cache_directory="/tmp/g-jax-cache",
        matmul_precision="highest",
        persistent_cache=True,
        persistent_cache_min_entry_size_bytes=1024,
        persistent_cache_min_compile_time_seconds=5,
        xla_autotune_cache=True,
        transfer_guard=True,
    )

    with unittest.mock.patch("jax.config.update") as config_update_mock:
        applied_count = native_setup_session.apply_config_updates()

    assert applied_count == 8
    assert [call.args for call in config_update_mock.call_args_list] == [
        ("jax_platforms", "cuda"),
        ("jax_enable_x64", True),
        ("jax_default_matmul_precision", "highest"),
        ("jax_compilation_cache_dir", "/tmp/g-jax-cache"),
        ("jax_persistent_cache_min_entry_size_bytes", 1024),
        ("jax_persistent_cache_min_compile_time_secs", 5),
        ("jax_persistent_cache_enable_xla_caches", "xla_gpu_per_fusion_autotune_cache_dir"),
        ("jax_transfer_guard", "disallow"),
    ]


def test_native_jax_runtime_setup_session_validates_gpu_devices(tmp_path: Path) -> None:
    class FakeDevice:
        platform = "gpu"

        def __str__(self) -> str:
            return "GpuDevice(id=0)"

    control_device_path = tmp_path / "nvidiactl"
    control_device_path.touch()
    native_setup_session = build_native_jax_runtime_setup_session(
        requested_device="gpu",
        cache_directory="/tmp/g-jax-cache",
        persistent_cache=False,
    )

    with unittest.mock.patch("jax.devices", return_value=[FakeDevice()]) as devices_mock:
        validated_payload = native_setup_session.validate_gpu_if_configured(
            str(control_device_path),
            str(tmp_path / "missing-nvidia-uvm"),
            str(tmp_path / "missing-driver"),
        )

    devices_mock.assert_called_once_with()
    assert validated_payload["gpu_validation_status"] == "succeeded"
    assert native_setup_session.setup_payload()["gpu_validation_status"] == "succeeded"
    assert native_setup_session.setup_payload()["gpu_validation_message"] == "JAX reported at least one GPU device."


def test_native_jax_runtime_policy_payload() -> None:
    runtime_state = _core.NativeRuntimeState()
    jax_policy_payload = runtime_state.build_jax_runtime_policy_payload(
        device="gpu",
        cache_directory="/tmp/g-jax-cache",
        matmul_precision="highest",
        persistent_cache=False,
        persistent_cache_min_entry_size_bytes=1024,
        persistent_cache_min_compile_time_seconds=5,
        xla_autotune_cache=True,
        transfer_guard=True,
    )

    assert jax_policy_payload == {
        "device": "gpu",
        "cache_directory": "/tmp/g-jax-cache",
        "matmul_precision": "highest",
        "persistent_cache": False,
        "persistent_cache_min_entry_size_bytes": 1024,
        "persistent_cache_min_compile_time_seconds": 5,
        "xla_autotune_cache": True,
        "transfer_guard": True,
    }
    assert not hasattr(_core, "build_jax_runtime_policy_payload")


def test_native_jax_runtime_setup_session_owns_diagnostic_payloads() -> None:
    native_setup_session = build_native_jax_runtime_setup_session(
        requested_device="gpu",
        cache_directory="/tmp/g-cache",
        matmul_precision="float32",
        persistent_cache=True,
        persistent_cache_min_entry_size_bytes=1024,
        persistent_cache_min_compile_time_seconds=5,
        xla_autotune_cache=True,
        transfer_guard=True,
    )
    native_setup_session.complete_validation_payload("failed", "no gpu")

    diagnostic_payloads = native_setup_session.diagnostic_event_payloads()

    assert [payload["event_name"] for payload in diagnostic_payloads] == [
        "jax_platform_selected",
        "jax_persistent_cache_configured",
        "jax_xla_auxiliary_cache_configured",
        "jax_transfer_guard_configured",
        "jax_gpu_validation",
    ]
    persistent_cache_fields = typing.cast("tuple[dict[str, object], ...]", diagnostic_payloads[1]["fields"])
    auxiliary_cache_fields = typing.cast("tuple[dict[str, object], ...]", diagnostic_payloads[2]["fields"])
    gpu_validation_fields = typing.cast("tuple[dict[str, object], ...]", diagnostic_payloads[4]["fields"])
    assert diagnostic_payloads[0]["message"] == "Selected JAX platform cuda."
    assert persistent_cache_fields[0] == {"name": "enabled", "value": True}
    assert auxiliary_cache_fields[0] == {"name": "enabled", "value": True}
    assert diagnostic_payloads[4]["level"] == "error"
    assert list(gpu_validation_fields) == [
        {"name": "status", "value": "failed"},
        {"name": "message", "value": "no gpu"},
    ]
    assert not hasattr(_core, "build_jax_runtime_setup_diagnostic_payloads")


def test_native_jax_runtime_config_update_payloads_are_not_exported() -> None:
    native_setup_session = build_native_jax_runtime_setup_session(
        requested_device="cpu",
        cache_directory="/tmp/g-cache",
    )

    assert not hasattr(native_setup_session, "config_update_payloads")
    assert not hasattr(_core, "plan_jax_runtime_config_update_payloads")


def test_native_jax_runtime_setup_side_effect_plan_payload_is_not_exported() -> None:
    cpu_setup_session = build_native_jax_runtime_setup_session(
        requested_device="cpu",
        cache_directory="/tmp/g-cache",
        persistent_cache=True,
    )
    gpu_setup_session = build_native_jax_runtime_setup_session(
        requested_device="gpu",
        cache_directory="/tmp/g-cache",
        persistent_cache=False,
    )

    assert cpu_setup_session.should_validate_gpu is False
    assert gpu_setup_session.should_validate_gpu is True
    assert not hasattr(cpu_setup_session, "side_effect_plan_payload")
    assert not hasattr(gpu_setup_session, "side_effect_plan_payload")
    assert not hasattr(_core, "plan_jax_runtime_setup_side_effects_payload")


def test_native_jax_runtime_setup_validation_completion() -> None:
    native_setup_session = build_native_jax_runtime_setup_session(
        requested_device="gpu",
        cache_directory="cache",
        matmul_precision="float32",
        persistent_cache=True,
    )

    completed_setup = native_setup_session.complete_validation_payload("succeeded", "gpu ready")

    assert completed_setup["requested_device"] == "gpu"
    assert completed_setup["cache_directory"] == "cache"
    assert completed_setup["gpu_validation_status"] == "succeeded"
    assert completed_setup["gpu_validation_message"] == "gpu ready"
    assert not hasattr(_core, "resolve_jax_runtime_setup_payload")
    assert not hasattr(_core, "complete_jax_runtime_setup_validation_payload")
    assert not hasattr(_core, "plan_jax_gpu_validation_payload")


def test_native_jax_runtime_setup_session_rejects_direct_construction() -> None:
    setup_session_type = typing.cast("typing.Any", _core.NativeJaxRuntimeSetupSession)
    with pytest.raises(TypeError, match=r"cannot create .*NativeJaxRuntimeSetupSession"):
        setup_session_type({})


def test_native_jax_runtime_diagnostic_event_records_telemetry() -> None:
    class DiagnosticField:
        def __init__(self, name: str, value: object) -> None:
            self.name = name
            self.value = value

    class DiagnosticEvent:
        def __init__(self) -> None:
            self.event_name = "jax_native_dispatch_test"
            self.level = "info"
            self.message = "JAX diagnostic"
            self.fields = (DiagnosticField("platform", "cpu"),)

    class RecordingNativeTelemetrySession:
        def __init__(self) -> None:
            self.events: list[tuple[object, str]] = []

        def emit_jax_runtime_diagnostic_event(
            self,
            diagnostic_event: object,
            telemetry_level: str,
        ) -> None:
            self.events.append((diagnostic_event, telemetry_level))

    class RecordingTelemetrySession:
        def __init__(self) -> None:
            self.native_telemetry_session = RecordingNativeTelemetrySession()

    class DisabledTelemetrySession:
        native_telemetry_session = None

    class LegacyTelemetrySession:
        def log_jax_runtime_diagnostic_event(
            self,
            diagnostic_event: object,
            *,
            telemetry_level: str,
        ) -> None:
            raise AssertionError((diagnostic_event, telemetry_level))

    diagnostic_event = DiagnosticEvent()
    telemetry_session = RecordingTelemetrySession()

    emitted_plan = _core.record_jax_runtime_diagnostic_event(diagnostic_event, telemetry_session)
    disabled_plan = _core.record_jax_runtime_diagnostic_event(diagnostic_event, DisabledTelemetrySession())
    skipped_plan = _core.record_jax_runtime_diagnostic_event(diagnostic_event, None)
    with pytest.raises(TypeError, match="native telemetry session handle"):
        _core.record_jax_runtime_diagnostic_event(diagnostic_event, LegacyTelemetrySession())

    assert emitted_plan.should_emit_telemetry is True
    assert emitted_plan.telemetry_level == "info"
    assert disabled_plan.should_emit_telemetry is False
    assert skipped_plan.should_emit_telemetry is False
    assert telemetry_session.native_telemetry_session.events == [(diagnostic_event, "info")]
    assert not hasattr(_core, "plan_jax_runtime_diagnostic_record")
    assert not hasattr(_core, "record_jax_runtime_diagnostic_log_event")
    assert not hasattr(_core, "plan_jax_runtime_diagnostic_record_payload")


def test_native_binary_correction_summary_plans_record_and_emit_policy() -> None:
    summary = _core.NativeBinaryCorrectionSummary()

    record_plan = summary.plan_diagnostics_record(
        has_telemetry_session=True,
        has_diagnostics=True,
    )
    assert record_plan.should_record is True
    missing_telemetry_record_plan = summary.plan_diagnostics_record(
        has_telemetry_session=False,
        has_diagnostics=True,
    )
    assert missing_telemetry_record_plan.should_record is False
    assert summary.chunk_count_with_pending(3) == 3

    empty_emit_plan = summary.plan_summary_emit(
        has_telemetry_session=True,
        pending_diagnostics_count=0,
    )
    assert empty_emit_plan.should_flush_pending_diagnostics is False
    assert empty_emit_plan.should_emit_summary is False
    pending_emit_plan = summary.plan_summary_emit(
        has_telemetry_session=True,
        pending_diagnostics_count=2,
    )
    assert pending_emit_plan.should_flush_pending_diagnostics is True
    assert pending_emit_plan.should_emit_summary is True
    summary.add_null_model_failure_count(1)
    summary_emit_plan = summary.plan_summary_emit(
        has_telemetry_session=True,
        pending_diagnostics_count=0,
    )
    assert summary_emit_plan.should_flush_pending_diagnostics is False
    assert summary_emit_plan.should_emit_summary is True
    missing_telemetry_emit_plan = summary.plan_summary_emit(
        has_telemetry_session=False,
        pending_diagnostics_count=2,
    )
    assert missing_telemetry_emit_plan.should_flush_pending_diagnostics is False
    assert missing_telemetry_emit_plan.should_emit_summary is False


def test_emit_binary_correction_summary_telemetry_uses_native_missing_session_policy() -> None:
    class DisabledTelemetrySession:
        native_telemetry_session = None

    class LegacyTelemetrySession:
        def log_binary_correction_summary(self, summary_payload: dict[str, int]) -> None:
            raise AssertionError(summary_payload)

    telemetry_session = RecordingNativeCallbackTelemetrySession()
    summary = _core.NativeBinaryCorrectionSummary()
    summary.add_null_model_failure_count(3)
    summary_payload = summary.summary_payload()

    _core.emit_binary_correction_summary_telemetry(telemetry_session, summary_payload, "missing summary session")
    _core.emit_binary_correction_summary_telemetry(None, None, "missing summary session")
    _core.emit_binary_correction_summary_telemetry(
        DisabledTelemetrySession(), summary_payload, "missing summary session"
    )

    assert telemetry_session.binary_summaries == [summary_payload]
    with pytest.raises(RuntimeError, match="missing summary session"):
        _core.emit_binary_correction_summary_telemetry(None, summary_payload, "missing summary session")
    with pytest.raises(TypeError, match="native telemetry session handle"):
        _core.emit_binary_correction_summary_telemetry(
            LegacyTelemetrySession(),
            summary_payload,
            "missing summary session",
        )


def test_native_nvidia_driver_visibility_uses_any_driver_path(tmp_path: Path) -> None:
    control_device_path = tmp_path / "nvidiactl"
    uvm_device_path = tmp_path / "nvidia-uvm"
    driver_directory_path = tmp_path / "driver"
    setup_session = build_native_jax_runtime_setup_session(requested_device="gpu", cache_directory="")

    assert not setup_session.nvidia_driver_files_are_visible(
        str(control_device_path),
        str(uvm_device_path),
        str(driver_directory_path),
    )

    driver_directory_path.mkdir()

    assert setup_session.nvidia_driver_files_are_visible(
        str(control_device_path),
        str(uvm_device_path),
        str(driver_directory_path),
    )


def test_native_default_nvidia_driver_probe_paths_payload() -> None:
    setup_session = build_native_jax_runtime_setup_session(requested_device="gpu", cache_directory="")

    assert setup_session.default_nvidia_driver_probe_paths_payload() == {
        "control_device_path": "/dev/nvidiactl",
        "uvm_device_path": "/dev/nvidia-uvm",
        "driver_directory_path": "/proc/driver/nvidia",
    }
    assert not hasattr(_core, "nvidia_driver_files_are_visible_value")
    assert not hasattr(_core, "default_nvidia_driver_probe_paths_payload")


def test_native_gpu_genotype_format_resolution_policy() -> None:
    assert (
        _core.resolve_manifest_gpu_genotype_format(
            resume=True,
            manifest_gpu_genotype_format="packed8",
            association_backend_genotype_format="dosage",
        )
        == "packed8"
    )
    assert (
        _core.resolve_manifest_gpu_genotype_format(
            resume=True,
            manifest_gpu_genotype_format=None,
            association_backend_genotype_format="dosage",
        )
        == "dosage"
    )
    assert (
        _core.resolve_manifest_gpu_genotype_format(
            resume=False,
            manifest_gpu_genotype_format="packed8",
            association_backend_genotype_format=None,
        )
        is None
    )

    auto_to_dosage_plan = _core.plan_gpu_genotype_format_auto_to_dosage(
        requested_gpu_genotype_format="auto",
        resolution_reason="multi_trait_or_linear_pipeline",
    )
    assert auto_to_dosage_plan.requested_gpu_genotype_format == "auto"
    assert auto_to_dosage_plan.resolved_gpu_genotype_format == "dosage"
    assert auto_to_dosage_plan.resolution_reason == "multi_trait_or_linear_pipeline"
    assert auto_to_dosage_plan.fallback_error is None
    assert auto_to_dosage_plan.requires_trusted_validation is False
    assert auto_to_dosage_plan.is_resolved is True
    assert auto_to_dosage_plan.should_log_auto_resolution is True

    explicit_plan = _core.plan_single_trait_binary_gpu_genotype_format_resolution(
        requested_gpu_genotype_format="packed8",
        manifest_gpu_genotype_format=None,
        association_backend_genotype_format=None,
        resume=False,
        jax_device="gpu",
    )
    assert explicit_plan.resolved_gpu_genotype_format == "packed8"
    assert explicit_plan.resolution_reason == "explicit"
    assert explicit_plan.should_log_auto_resolution is False

    manifest_plan = _core.plan_single_trait_binary_gpu_genotype_format_resolution(
        requested_gpu_genotype_format="auto",
        manifest_gpu_genotype_format=None,
        association_backend_genotype_format="dosage",
        resume=True,
        jax_device="gpu",
    )
    assert manifest_plan.resolved_gpu_genotype_format == "dosage"
    assert manifest_plan.resolution_reason == "resume_manifest"
    assert manifest_plan.requires_trusted_validation is False

    validation_plan = _core.plan_single_trait_binary_gpu_genotype_format_resolution(
        requested_gpu_genotype_format="auto",
        manifest_gpu_genotype_format=None,
        association_backend_genotype_format=None,
        resume=False,
        jax_device="gpu",
    )
    assert validation_plan.resolved_gpu_genotype_format is None
    assert validation_plan.resolution_reason is None
    assert validation_plan.requires_trusted_validation is True

    passed_plan = _core.plan_auto_gpu_genotype_format_after_trusted_validation(fallback_error=None)
    assert passed_plan.resolved_gpu_genotype_format == "packed8"
    assert passed_plan.resolution_reason == "trusted_validation_passed"

    failed_plan = _core.plan_auto_gpu_genotype_format_after_trusted_validation(
        fallback_error="packed8 incompatible",
    )
    assert failed_plan.resolved_gpu_genotype_format == "dosage"
    assert failed_plan.resolution_reason == "trusted_validation_failed"
    assert failed_plan.fallback_error == "packed8 incompatible"

    with pytest.raises(ValueError, match="Unsupported GPU genotype format"):
        _core.plan_gpu_genotype_format_auto_to_dosage(
            requested_gpu_genotype_format="unknown",
            resolution_reason="unused",
        )


def test_resolve_delivery_callback_batch_size_enforces_native_delivery_policy() -> None:
    assert (
        _core.resolve_delivery_callback_batch_size(
            callback_batch_size=None,
            variant_major_packed8_probability_pairs=False,
        )
        == 1
    )
    assert (
        _core.resolve_delivery_callback_batch_size(
            callback_batch_size=2,
            variant_major_packed8_probability_pairs=False,
        )
        == 2
    )
    assert (
        _core.resolve_delivery_callback_batch_size(
            callback_batch_size=1,
            variant_major_packed8_probability_pairs=True,
        )
        == 1
    )
    with pytest.raises(ValueError, match="native_callback_batch_size must be positive"):
        _core.resolve_delivery_callback_batch_size(
            callback_batch_size=0,
            variant_major_packed8_probability_pairs=False,
        )
    with pytest.raises(ValueError, match="packed8 BGEN delivery"):
        _core.resolve_delivery_callback_batch_size(
            callback_batch_size=2,
            variant_major_packed8_probability_pairs=True,
        )


def test_plan_bgen_delivery_invocation_uses_native_delivery_policy() -> None:
    dosage_plan = _core.plan_bgen_delivery_invocation(
        callback_batch_size=2,
        variant_major_packed8_probability_pairs=False,
        has_native_multi_aligned_sample_data=True,
        has_native_aligned_sample_data=True,
    )
    assert dosage_plan.delivery_method == "dosage_native_multi_aligned_samples"
    assert dosage_plan.callback_batch_size == 2

    fallback_dosage_plan = _core.plan_bgen_delivery_invocation(
        callback_batch_size=None,
        variant_major_packed8_probability_pairs=False,
        has_native_multi_aligned_sample_data=False,
        has_native_aligned_sample_data=False,
    )
    assert fallback_dosage_plan.delivery_method == "dosage_sample_indices"
    assert fallback_dosage_plan.callback_batch_size == 1

    packed8_plan = _core.plan_bgen_delivery_invocation(
        callback_batch_size=1,
        variant_major_packed8_probability_pairs=True,
        has_native_multi_aligned_sample_data=False,
        has_native_aligned_sample_data=True,
    )
    assert packed8_plan.delivery_method == "packed8_native_aligned_samples"
    assert packed8_plan.callback_batch_size == 1

    with pytest.raises(ValueError, match="packed8 BGEN delivery"):
        _core.plan_bgen_delivery_invocation(
            callback_batch_size=2,
            variant_major_packed8_probability_pairs=True,
            has_native_multi_aligned_sample_data=False,
            has_native_aligned_sample_data=False,
        )


def test_resolve_grouped_union_callback_batch_size_enforces_native_delivery_policy() -> None:
    assert _core.resolve_grouped_union_callback_batch_size(native_callback_batch_size=1) == 1
    with pytest.raises(ValueError, match="native_callback_batch_size must be positive"):
        _core.resolve_grouped_union_callback_batch_size(native_callback_batch_size=0)
    with pytest.raises(ValueError, match="grouped union BGEN delivery"):
        _core.resolve_grouped_union_callback_batch_size(native_callback_batch_size=2)


def test_native_callback_worker_lifecycle_state_tracks_start() -> None:
    lifecycle_state = _core.NativeCallbackWorkerLifecycleState()

    assert lifecycle_state.has_started is False
    assert lifecycle_state.mark_started() is True
    assert lifecycle_state.has_started is True
    assert lifecycle_state.mark_started() is False


def test_plan_callback_worker_start_uses_native_start_policy() -> None:
    scheduler_state = _core.NativeCallbackSchedulerState(
        staging_depth=1,
        native_callback_batch_size=1,
        result_in_flight_limit=None,
        dosage_buffer_limit=None,
    )
    start_plan = scheduler_state.plan_worker_start()

    assert start_plan.should_start is True
    assert start_plan.start_result_worker is True
    assert start_plan.start_dosage_worker is True
    assert start_plan.start_actions == ["start_result_worker", "start_dosage_worker"]

    assert scheduler_state.mark_started() is True
    already_started_plan = scheduler_state.plan_worker_start()
    assert already_started_plan.should_start is False
    assert already_started_plan.start_result_worker is False
    assert already_started_plan.start_dosage_worker is False
    assert already_started_plan.start_actions == []


def test_native_callback_scheduler_state_plans_worker_start_attempts() -> None:
    scheduler_state = _core.NativeCallbackSchedulerState(
        staging_depth=1,
        native_callback_batch_size=1,
        result_in_flight_limit=None,
        dosage_buffer_limit=None,
    )

    start_attempt_plan = scheduler_state.plan_worker_start_attempt()
    assert start_attempt_plan.should_start is True
    assert start_attempt_plan.start_result_worker is True
    assert start_attempt_plan.start_dosage_worker is True
    assert start_attempt_plan.start_actions == ["start_result_worker", "start_dosage_worker"]
    assert start_attempt_plan.has_marked_started is True
    assert start_attempt_plan.has_start_error is False
    assert start_attempt_plan.error_message is None
    assert scheduler_state.has_started is True

    already_started_attempt_plan = scheduler_state.plan_worker_start_attempt()
    assert already_started_attempt_plan.should_start is False
    assert already_started_attempt_plan.start_result_worker is False
    assert already_started_attempt_plan.start_dosage_worker is False
    assert already_started_attempt_plan.start_actions == []
    assert already_started_attempt_plan.has_marked_started is False
    assert already_started_attempt_plan.has_start_error is False
    assert already_started_attempt_plan.error_message is None


def test_native_callback_scheduler_state_owns_callback_resource_state() -> None:
    scheduler_state = _core.NativeCallbackSchedulerState(
        staging_depth=3,
        native_callback_batch_size=2,
        result_in_flight_limit=7,
        dosage_buffer_limit=8,
    )

    assert scheduler_state.native_callback_batch_size == 2
    assert scheduler_state.dosage_queue_depth == 3
    assert scheduler_state.dosage_queue_capacity == 3
    assert scheduler_state.dosage_queue_occupied_count == 0
    assert scheduler_state.has_available_dosage_queue_slot() is True
    assert scheduler_state.result_queue_depth == 3
    assert scheduler_state.result_queue_capacity == 3
    assert scheduler_state.result_queue_occupied_count == 0
    assert scheduler_state.has_available_result_queue_slot() is True
    assert scheduler_state.result_in_flight_limit == 7
    assert scheduler_state.result_in_flight_slot_limit == 7
    assert scheduler_state.dosage_buffer_limit == 8
    assert scheduler_state.dosage_buffer_pool_limit == 8
    assert scheduler_state.has_started is False
    start_plan = scheduler_state.plan_worker_start()
    assert start_plan.should_start is True
    assert start_plan.start_actions == ["start_result_worker", "start_dosage_worker"]
    assert scheduler_state.mark_started() is True
    assert scheduler_state.has_started is True
    assert scheduler_state.plan_worker_start().should_start is False
    assert scheduler_state.mark_started() is False

    assert scheduler_state.acquire_dosage_queue_slot() is True
    assert scheduler_state.dosage_queue_occupied_count == 1
    assert scheduler_state.has_available_dosage_queue_slot() is True
    assert scheduler_state.release_dosage_queue_slot() is True
    assert scheduler_state.dosage_queue_occupied_count == 0

    assert scheduler_state.acquire_result_queue_slot() is True
    assert scheduler_state.result_queue_occupied_count == 1
    assert scheduler_state.has_available_result_queue_slot() is True
    assert scheduler_state.release_result_queue_slot() is True
    assert scheduler_state.result_queue_occupied_count == 0

    assert scheduler_state.acquire_result_in_flight_slot() is True
    assert scheduler_state.result_in_flight_occupied_count == 1
    assert scheduler_state.has_available_result_in_flight_slot() is True
    assert scheduler_state.release_result_in_flight_slot() is True
    assert scheduler_state.result_in_flight_occupied_count == 0

    assert scheduler_state.register_dosage_buffer(11) is True
    assert scheduler_state.owns_dosage_buffer(11) is True
    assert scheduler_state.dosage_buffer_allocated_count == 1
    assert scheduler_state.dosage_buffer_identifiers == [11]
    assert scheduler_state.has_available_dosage_buffer_slot() is True
    assert scheduler_state.discard_dosage_buffer(11) is True
    assert scheduler_state.dosage_buffer_allocated_count == 0
    assert scheduler_state.has_dosage_worker_error is False
    assert scheduler_state.has_result_worker_error is False
    assert scheduler_state.dosage_worker_error_message is None
    assert scheduler_state.result_worker_error_message is None
    no_error_raise_plan = scheduler_state.plan_worker_error_raise()
    assert no_error_raise_plan.should_raise is False
    assert no_error_raise_plan.raise_dosage_worker_error is False
    assert no_error_raise_plan.raise_result_worker_error is False
    assert no_error_raise_plan.error_message is None

    scheduler_state.record_result_worker_error("writer failed")
    result_error_raise_plan = scheduler_state.plan_worker_error_raise()
    assert result_error_raise_plan.should_raise is True
    assert result_error_raise_plan.raise_dosage_worker_error is False
    assert result_error_raise_plan.raise_result_worker_error is True
    assert result_error_raise_plan.error_message == "native pipeline result writer worker failed: writer failed"

    scheduler_state.record_dosage_worker_error("dosage failed")

    assert scheduler_state.has_dosage_worker_error is True
    assert scheduler_state.has_result_worker_error is True
    assert scheduler_state.dosage_worker_error_message == "native pipeline callback worker failed: dosage failed"
    assert scheduler_state.result_worker_error_message == "native pipeline result writer worker failed: writer failed"
    dosage_error_raise_plan = scheduler_state.plan_worker_error_raise()
    assert dosage_error_raise_plan.should_raise is True
    assert dosage_error_raise_plan.raise_dosage_worker_error is True
    assert dosage_error_raise_plan.raise_result_worker_error is False
    assert dosage_error_raise_plan.error_message == "native pipeline callback worker failed: dosage failed"
    assert scheduler_state.clear_dosage_worker_error() is True
    assert scheduler_state.clear_result_worker_error() is True
    assert scheduler_state.dosage_worker_error_message is None
    assert scheduler_state.result_worker_error_message is None

    assert scheduler_state.backpressure_poll_timeout_seconds == 0.1
    finish_plan = scheduler_state.plan_worker_finish()
    assert finish_plan.finish_actions == [
        "stop_dosage_worker",
        "join_dosage_worker",
        "stop_result_worker",
        "join_result_worker",
        "raise_worker_error",
        "complete_progress",
        "emit_binary_correction_summary",
    ]
    assert finish_plan.stop_dosage_worker is True
    assert finish_plan.join_dosage_worker is True
    assert finish_plan.stop_result_worker is True
    assert finish_plan.join_result_worker is True
    assert finish_plan.raise_worker_error is True
    assert finish_plan.complete_progress is True
    assert finish_plan.emit_binary_correction_summary is True
    assert finish_plan.dosage_stop_timeout_seconds == 60.0
    assert finish_plan.dosage_join_timeout_seconds == 300.0
    assert finish_plan.result_stop_timeout_seconds == 60.0
    assert finish_plan.result_join_timeout_seconds == 300.0
    abort_plan = scheduler_state.plan_worker_abort()
    assert abort_plan.abort_actions == ["stop_dosage_worker", "stop_result_worker"]
    assert abort_plan.stop_dosage_worker is True
    assert abort_plan.stop_result_worker is True
    assert abort_plan.dosage_stop_timeout_seconds == 1.0
    assert abort_plan.result_stop_timeout_seconds == 1.0

    queue_operation_plan = scheduler_state.plan_queue_operation_observation(
        queue_name="dosage_buffer_pool",
        operation_name="return",
        elapsed_seconds=0.25,
        blocked=True,
    )
    assert queue_operation_plan.queue_name == "dosage_buffer_pool"
    assert queue_operation_plan.operation_name == "return"
    assert queue_operation_plan.blocked_seconds == 0.25

    queue_backpressure_observation = scheduler_state.plan_queue_backpressure_observation(
        queue_name="dosage_buffer_pool",
        operation_name="return",
        queue_depth=1,
        queue_capacity=2,
        elapsed_seconds=0.25,
        blocked=True,
    )
    assert queue_backpressure_observation.queue_name == "dosage_buffer_pool"
    assert queue_backpressure_observation.operation_name == "return"
    assert queue_backpressure_observation.queue_depth == 1
    assert queue_backpressure_observation.queue_capacity == 2
    assert queue_backpressure_observation.elapsed_seconds == 0.25
    assert queue_backpressure_observation.blocked_seconds == 0.25

    queue_stage_plan = scheduler_state.plan_queue_stage_observation(
        queue_name="dosage_queue",
        operation_name="producer_blocking",
        elapsed_seconds=0.5,
        blocked=True,
    )
    assert queue_stage_plan.queue_name == "dosage_queue"
    assert queue_stage_plan.operation_name == "producer_blocking"
    assert queue_stage_plan.stage_name == "callback_queue_producer_blocking"
    assert queue_stage_plan.blocked_seconds == 0.5

    queue_stage_backpressure_observation = scheduler_state.plan_queue_stage_backpressure_observation(
        queue_name="dosage_queue",
        operation_name="producer_blocking",
        queue_depth=3,
        queue_capacity=3,
        elapsed_seconds=0.5,
        blocked=True,
    )
    assert queue_stage_backpressure_observation.queue_name == "dosage_queue"
    assert queue_stage_backpressure_observation.operation_name == "producer_blocking"
    assert queue_stage_backpressure_observation.stage_name == "callback_queue_producer_blocking"
    assert queue_stage_backpressure_observation.queue_depth == 3
    assert queue_stage_backpressure_observation.queue_capacity == 3
    assert queue_stage_backpressure_observation.elapsed_seconds == 0.5
    assert queue_stage_backpressure_observation.blocked_seconds == 0.5

    reuse_plan = scheduler_state.plan_dosage_buffer_reuse(
        buffered_shape=(4, 5),
        expected_shape=(2, 3),
    )
    assert reuse_plan is not None
    assert reuse_plan.requires_slice is True
    assert reuse_plan.slice_dimensions == [2, 3]
    assert (
        scheduler_state.plan_dosage_buffer_reuse(
            buffered_shape=(2, 3),
            expected_shape=(3, 2),
        )
        is None
    )

    batch_handoff_plan = scheduler_state.plan_variant_major_dosage_batch_handoff(
        metadata_count=2,
        genotype_matrix_by_variant_count=2,
        chunk_stats_count=2,
    )
    assert batch_handoff_plan.chunk_count == 2
    with pytest.raises(ValueError, match="identical lengths"):
        scheduler_state.plan_variant_major_dosage_batch_handoff(
            metadata_count=2,
            genotype_matrix_by_variant_count=1,
            chunk_stats_count=2,
        )

    dosage_join_plan = scheduler_state.plan_dosage_worker_join(timeout_seconds=None)
    assert dosage_join_plan.should_join is True
    assert dosage_join_plan.timeout_seconds == 60.0
    dosage_stop_plan = scheduler_state.plan_dosage_worker_stop(timeout_seconds=None, is_worker_alive=True)
    assert dosage_stop_plan.should_stop is True
    assert dosage_stop_plan.timeout_seconds == 60.0
    dosage_poll_plan = scheduler_state.plan_dosage_worker_stop_poll(
        remaining_timeout_seconds=1.0,
        is_worker_alive=True,
    )
    assert dosage_poll_plan.should_stop is True
    assert dosage_poll_plan.poll_timeout_seconds == 0.1

    scheduler_state.record_result_worker_error("writer failed")
    result_stop_plan = scheduler_state.plan_result_worker_stop(timeout_seconds=None, is_worker_alive=True)
    assert result_stop_plan.should_stop is False
    assert result_stop_plan.timeout_seconds == 60.0
    result_poll_plan = scheduler_state.plan_result_worker_stop_poll(
        remaining_timeout_seconds=1.0,
        is_worker_alive=True,
    )
    assert result_poll_plan.should_stop is False
    assert result_poll_plan.poll_timeout_seconds == 0.1
    result_join_plan = scheduler_state.plan_result_worker_join(timeout_seconds=0.25)
    assert result_join_plan.should_join is True
    assert result_join_plan.timeout_seconds == 0.25

    with pytest.raises(ValueError, match="effective dosage_buffer_limit"):
        _core.NativeCallbackSchedulerState(
            staging_depth=1,
            native_callback_batch_size=3,
            result_in_flight_limit=None,
            dosage_buffer_limit=2,
        )


def test_native_callback_scheduler_state_updates_worker_errors() -> None:
    scheduler_state = _core.NativeCallbackSchedulerState(
        staging_depth=1,
        native_callback_batch_size=1,
        result_in_flight_limit=None,
        dosage_buffer_limit=None,
    )

    dosage_update_plan = scheduler_state.update_dosage_worker_error("dosage failed")
    assert dosage_update_plan.had_error is False
    assert dosage_update_plan.has_error is True
    assert dosage_update_plan.error_message == "native pipeline callback worker failed: dosage failed"
    assert scheduler_state.dosage_worker_error_message == "native pipeline callback worker failed: dosage failed"

    dosage_clear_plan = scheduler_state.update_dosage_worker_error(None)
    assert dosage_clear_plan.had_error is True
    assert dosage_clear_plan.has_error is False
    assert dosage_clear_plan.error_message is None
    assert scheduler_state.dosage_worker_error_message is None

    result_update_plan = scheduler_state.update_result_worker_error("writer failed")
    assert result_update_plan.had_error is False
    assert result_update_plan.has_error is True
    assert result_update_plan.error_message == "native pipeline result writer worker failed: writer failed"
    assert scheduler_state.result_worker_error_message == "native pipeline result writer worker failed: writer failed"


def test_native_callback_scheduler_state_plans_queue_put_and_get_attempts() -> None:
    scheduler_state = _core.NativeCallbackSchedulerState(
        staging_depth=1,
        native_callback_batch_size=1,
        result_in_flight_limit=None,
        dosage_buffer_limit=None,
    )

    dosage_put_plan = scheduler_state.plan_dosage_queue_put_attempt(wait_timeout_seconds=0.25)
    assert dosage_put_plan.should_put is True
    assert dosage_put_plan.should_wait is False
    assert dosage_put_plan.wait_timeout_seconds == 0.0
    assert dosage_put_plan.queue_depth == 1
    assert dosage_put_plan.queue_capacity == 1

    blocked_dosage_put_plan = scheduler_state.plan_dosage_queue_put_attempt(wait_timeout_seconds=0.25)
    assert blocked_dosage_put_plan.should_put is False
    assert blocked_dosage_put_plan.should_wait is True
    assert blocked_dosage_put_plan.wait_timeout_seconds == 0.25
    assert blocked_dosage_put_plan.queue_depth == 1
    assert blocked_dosage_put_plan.queue_capacity == 1

    dosage_get_plan = scheduler_state.plan_dosage_queue_get_attempt(has_queued_item=True)
    assert dosage_get_plan.should_get is True
    assert dosage_get_plan.should_wait is False
    assert dosage_get_plan.has_release_error is False
    assert dosage_get_plan.wait_timeout_seconds == 0.0
    assert dosage_get_plan.queue_depth == 0
    assert dosage_get_plan.queue_capacity == 1

    empty_dosage_get_plan = scheduler_state.plan_dosage_queue_get_attempt(has_queued_item=False)
    assert empty_dosage_get_plan.should_get is False
    assert empty_dosage_get_plan.should_wait is True
    assert empty_dosage_get_plan.has_release_error is False
    assert empty_dosage_get_plan.wait_timeout_seconds == 0.1
    assert empty_dosage_get_plan.queue_depth == 0
    assert empty_dosage_get_plan.queue_capacity == 1

    release_error_plan = scheduler_state.plan_result_queue_get_attempt(has_queued_item=True)
    assert release_error_plan.should_get is False
    assert release_error_plan.should_wait is False
    assert release_error_plan.has_release_error is True
    assert release_error_plan.wait_timeout_seconds == 0.0
    assert release_error_plan.queue_depth == 0
    assert release_error_plan.queue_capacity == 1

    result_put_plan = scheduler_state.plan_result_queue_put_attempt(wait_timeout_seconds=float("nan"))
    assert result_put_plan.should_put is True
    assert result_put_plan.should_wait is False
    assert result_put_plan.wait_timeout_seconds == 0.0
    assert result_put_plan.queue_depth == 1
    assert result_put_plan.queue_capacity == 1

    expired_result_put_plan = scheduler_state.plan_result_queue_put_attempt(wait_timeout_seconds=float("nan"))
    assert expired_result_put_plan.should_put is False
    assert expired_result_put_plan.should_wait is False
    assert expired_result_put_plan.wait_timeout_seconds == 0.0
    assert expired_result_put_plan.queue_depth == 1
    assert expired_result_put_plan.queue_capacity == 1

    dosage_backpressure_scheduler_state = _core.NativeCallbackSchedulerState(
        staging_depth=1,
        native_callback_batch_size=1,
        result_in_flight_limit=None,
        dosage_buffer_limit=None,
    )
    first_dosage_backpressure_plan = dosage_backpressure_scheduler_state.plan_dosage_queue_put_backpressure_attempt()
    assert first_dosage_backpressure_plan.should_put is True
    assert first_dosage_backpressure_plan.should_wait is False
    assert first_dosage_backpressure_plan.wait_timeout_seconds == 0.0
    assert first_dosage_backpressure_plan.queue_depth == 1
    assert first_dosage_backpressure_plan.queue_capacity == 1
    second_dosage_backpressure_plan = dosage_backpressure_scheduler_state.plan_dosage_queue_put_backpressure_attempt()
    assert second_dosage_backpressure_plan.should_put is False
    assert second_dosage_backpressure_plan.should_wait is True
    assert second_dosage_backpressure_plan.wait_timeout_seconds == 0.1
    assert second_dosage_backpressure_plan.queue_depth == 1
    assert second_dosage_backpressure_plan.queue_capacity == 1

    result_backpressure_scheduler_state = _core.NativeCallbackSchedulerState(
        staging_depth=1,
        native_callback_batch_size=1,
        result_in_flight_limit=None,
        dosage_buffer_limit=None,
    )
    first_result_backpressure_plan = result_backpressure_scheduler_state.plan_result_queue_put_backpressure_attempt()
    assert first_result_backpressure_plan.should_put is True
    assert first_result_backpressure_plan.should_wait is False
    assert first_result_backpressure_plan.wait_timeout_seconds == 0.0
    assert first_result_backpressure_plan.queue_depth == 1
    assert first_result_backpressure_plan.queue_capacity == 1
    second_result_backpressure_plan = result_backpressure_scheduler_state.plan_result_queue_put_backpressure_attempt()
    assert second_result_backpressure_plan.should_put is False
    assert second_result_backpressure_plan.should_wait is True
    assert second_result_backpressure_plan.wait_timeout_seconds == 0.1
    assert second_result_backpressure_plan.queue_depth == 1
    assert second_result_backpressure_plan.queue_capacity == 1


def test_native_callback_scheduler_state_plans_result_in_flight_slot_attempts() -> None:
    scheduler_state = _core.NativeCallbackSchedulerState(
        staging_depth=1,
        native_callback_batch_size=1,
        result_in_flight_limit=1,
        dosage_buffer_limit=None,
    )

    acquire_plan = scheduler_state.plan_result_in_flight_slot_acquire_attempt(wait_timeout_seconds=0.25)
    assert acquire_plan.should_acquire is True
    assert acquire_plan.should_wait is False
    assert acquire_plan.wait_timeout_seconds == 0.0
    assert acquire_plan.occupied_count == 1
    assert acquire_plan.slot_limit == 1

    blocked_acquire_plan = scheduler_state.plan_result_in_flight_slot_acquire_attempt(wait_timeout_seconds=0.25)
    assert blocked_acquire_plan.should_acquire is False
    assert blocked_acquire_plan.should_wait is True
    assert blocked_acquire_plan.wait_timeout_seconds == 0.25
    assert blocked_acquire_plan.occupied_count == 1
    assert blocked_acquire_plan.slot_limit == 1

    backpressure_acquire_plan = scheduler_state.plan_result_in_flight_slot_acquire_backpressure_attempt()
    assert backpressure_acquire_plan.should_acquire is False
    assert backpressure_acquire_plan.should_wait is True
    assert backpressure_acquire_plan.wait_timeout_seconds == 0.1
    assert backpressure_acquire_plan.occupied_count == 1
    assert backpressure_acquire_plan.slot_limit == 1

    release_plan = scheduler_state.plan_result_in_flight_slot_release_attempt()
    assert release_plan.should_release is True
    assert release_plan.has_release_error is False
    assert release_plan.occupied_count == 0
    assert release_plan.slot_limit == 1

    release_error_plan = scheduler_state.plan_result_in_flight_slot_release_attempt()
    assert release_error_plan.should_release is False
    assert release_error_plan.has_release_error is True
    assert release_error_plan.occupied_count == 0
    assert release_error_plan.slot_limit == 1

    nan_acquire_plan = scheduler_state.plan_result_in_flight_slot_acquire_attempt(wait_timeout_seconds=float("nan"))
    assert nan_acquire_plan.should_acquire is True
    assert nan_acquire_plan.should_wait is False
    assert nan_acquire_plan.wait_timeout_seconds == 0.0
    assert nan_acquire_plan.occupied_count == 1
    assert nan_acquire_plan.slot_limit == 1

    expired_acquire_plan = scheduler_state.plan_result_in_flight_slot_acquire_attempt(wait_timeout_seconds=float("nan"))
    assert expired_acquire_plan.should_acquire is False
    assert expired_acquire_plan.should_wait is False
    assert expired_acquire_plan.wait_timeout_seconds == 0.0
    assert expired_acquire_plan.occupied_count == 1
    assert expired_acquire_plan.slot_limit == 1


def test_native_callback_scheduler_state_plans_result_write_item_resource_release() -> None:
    scheduler_state = _core.NativeCallbackSchedulerState(
        staging_depth=1,
        native_callback_batch_size=1,
        result_in_flight_limit=1,
        dosage_buffer_limit=1,
    )

    pre_write_release_plan = scheduler_state.plan_result_write_item_pre_write_resource_release(
        has_host_dosage_buffer=True,
    )
    assert pre_write_release_plan.should_release_host_buffer is True
    assert pre_write_release_plan.should_release_result_in_flight_slot is False

    empty_pre_write_release_plan = scheduler_state.plan_result_write_item_pre_write_resource_release(
        has_host_dosage_buffer=False,
    )
    assert empty_pre_write_release_plan.should_release_host_buffer is False
    assert empty_pre_write_release_plan.should_release_result_in_flight_slot is False

    final_release_plan = scheduler_state.plan_result_write_item_final_resource_release(
        has_host_dosage_buffer=True,
        has_released_host_dosage_buffer=True,
        release_in_flight_slot=True,
    )
    assert final_release_plan.should_release_host_buffer is False
    assert final_release_plan.should_release_result_in_flight_slot is True

    cleanup_release_plan = scheduler_state.plan_result_write_item_final_resource_release(
        has_host_dosage_buffer=True,
        has_released_host_dosage_buffer=False,
        release_in_flight_slot=False,
    )
    assert cleanup_release_plan.should_release_host_buffer is True
    assert cleanup_release_plan.should_release_result_in_flight_slot is False


def test_native_callback_scheduler_state_plans_result_write_drain_completion() -> None:
    scheduler_state = _core.NativeCallbackSchedulerState(
        staging_depth=1,
        native_callback_batch_size=1,
        result_in_flight_limit=1,
        dosage_buffer_limit=1,
    )

    active_drain_plan = scheduler_state.plan_result_write_drain_completion(
        has_result_work_item=True,
        flush_binary_correction_diagnostics_on_stop=True,
    )
    assert active_drain_plan.should_stop is False
    assert active_drain_plan.should_flush_binary_correction_diagnostics is False

    binary_completion_plan = scheduler_state.plan_result_write_drain_completion(
        has_result_work_item=False,
        flush_binary_correction_diagnostics_on_stop=True,
    )
    assert binary_completion_plan.should_stop is True
    assert binary_completion_plan.should_flush_binary_correction_diagnostics is True

    linear_completion_plan = scheduler_state.plan_result_write_drain_completion(
        has_result_work_item=False,
        flush_binary_correction_diagnostics_on_stop=False,
    )
    assert linear_completion_plan.should_stop is True
    assert linear_completion_plan.should_flush_binary_correction_diagnostics is False


def test_native_callback_scheduler_state_plans_dosage_work_drain_completion() -> None:
    scheduler_state = _core.NativeCallbackSchedulerState(
        staging_depth=1,
        native_callback_batch_size=1,
        result_in_flight_limit=1,
        dosage_buffer_limit=1,
    )

    active_drain_plan = scheduler_state.plan_dosage_work_drain_completion(
        has_dosage_work_item=True,
    )
    assert active_drain_plan.should_stop is False

    completion_plan = scheduler_state.plan_dosage_work_drain_completion(
        has_dosage_work_item=False,
    )
    assert completion_plan.should_stop is True


def test_native_callback_scheduler_state_plans_dosage_buffer_attempts() -> None:
    scheduler_state = _core.NativeCallbackSchedulerState(
        staging_depth=1,
        native_callback_batch_size=1,
        result_in_flight_limit=None,
        dosage_buffer_limit=1,
    )

    allocate_plan = scheduler_state.plan_dosage_buffer_acquire_attempt(
        free_buffer_count=0,
        wait_timeout_seconds=0.25,
    )
    assert allocate_plan.should_take_free_buffer is False
    assert allocate_plan.should_allocate is True
    assert allocate_plan.should_wait is False
    assert allocate_plan.wait_timeout_seconds == 0.0
    assert allocate_plan.free_buffer_count == 0
    assert allocate_plan.allocated_count == 0
    assert allocate_plan.buffer_limit == 1

    register_plan = scheduler_state.plan_dosage_buffer_register_attempt(buffer_identifier=11)
    assert register_plan.should_register is True
    assert register_plan.has_registration_error is False
    assert register_plan.allocated_count == 1
    assert register_plan.buffer_limit == 1

    duplicate_register_plan = scheduler_state.plan_dosage_buffer_register_attempt(buffer_identifier=13)
    assert duplicate_register_plan.should_register is False
    assert duplicate_register_plan.has_registration_error is True
    assert duplicate_register_plan.allocated_count == 1
    assert duplicate_register_plan.buffer_limit == 1

    blocked_acquire_plan = scheduler_state.plan_dosage_buffer_acquire_attempt(
        free_buffer_count=0,
        wait_timeout_seconds=0.25,
    )
    assert blocked_acquire_plan.should_take_free_buffer is False
    assert blocked_acquire_plan.should_allocate is False
    assert blocked_acquire_plan.should_wait is True
    assert blocked_acquire_plan.wait_timeout_seconds == 0.25
    assert blocked_acquire_plan.free_buffer_count == 0
    assert blocked_acquire_plan.allocated_count == 1
    assert blocked_acquire_plan.buffer_limit == 1

    backpressure_acquire_plan = scheduler_state.plan_dosage_buffer_acquire_backpressure_attempt(free_buffer_count=0)
    assert backpressure_acquire_plan.should_take_free_buffer is False
    assert backpressure_acquire_plan.should_allocate is False
    assert backpressure_acquire_plan.should_wait is True
    assert backpressure_acquire_plan.wait_timeout_seconds == 0.1
    assert backpressure_acquire_plan.free_buffer_count == 0
    assert backpressure_acquire_plan.allocated_count == 1
    assert backpressure_acquire_plan.buffer_limit == 1

    free_buffer_plan = scheduler_state.plan_dosage_buffer_acquire_attempt(
        free_buffer_count=1,
        wait_timeout_seconds=0.25,
    )
    assert free_buffer_plan.should_take_free_buffer is True
    assert free_buffer_plan.should_allocate is False
    assert free_buffer_plan.should_wait is False
    assert free_buffer_plan.wait_timeout_seconds == 0.0
    assert free_buffer_plan.free_buffer_count == 1
    assert free_buffer_plan.allocated_count == 1
    assert free_buffer_plan.buffer_limit == 1

    return_plan = scheduler_state.plan_dosage_buffer_return_attempt(buffer_identifier=11)
    assert return_plan.should_return is True
    assert return_plan.allocated_count == 1
    assert return_plan.buffer_limit == 1

    unknown_return_plan = scheduler_state.plan_dosage_buffer_return_attempt(buffer_identifier=13)
    assert unknown_return_plan.should_return is False
    assert unknown_return_plan.allocated_count == 1
    assert unknown_return_plan.buffer_limit == 1

    discard_plan = scheduler_state.plan_dosage_buffer_discard_attempt(buffer_identifier=11)
    assert discard_plan.should_discard is True
    assert discard_plan.allocated_count == 0
    assert discard_plan.buffer_limit == 1

    missing_discard_plan = scheduler_state.plan_dosage_buffer_discard_attempt(buffer_identifier=11)
    assert missing_discard_plan.should_discard is False
    assert missing_discard_plan.allocated_count == 0
    assert missing_discard_plan.buffer_limit == 1

    nan_acquire_plan = scheduler_state.plan_dosage_buffer_acquire_attempt(
        free_buffer_count=0,
        wait_timeout_seconds=float("nan"),
    )
    assert nan_acquire_plan.should_take_free_buffer is False
    assert nan_acquire_plan.should_allocate is True
    assert nan_acquire_plan.should_wait is False
    assert nan_acquire_plan.wait_timeout_seconds == 0.0
    assert nan_acquire_plan.free_buffer_count == 0
    assert nan_acquire_plan.allocated_count == 0
    assert nan_acquire_plan.buffer_limit == 1


def test_native_callback_progress_state_tracks_chromosome_transitions() -> None:
    progress_state = _core.NativeCallbackProgressState()

    first_identity = _core.build_callback_chunk_identity("chr1", 0, 8)
    assert first_identity.chunk_identifier == 0
    assert first_identity.chromosome == "chr1"
    assert first_identity.variant_start_index == 0
    assert first_identity.variant_stop_index == 8
    assert first_identity.variant_count == 8

    first_update = progress_state.record_processed_chunk(first_identity)
    assert first_update.processed_chunk_count == 1
    assert first_update.completed_chromosome is None
    assert first_update.completed_processed_chunk_count is None
    assert first_update.started_chromosome == "chr1"
    assert first_update.chunk_identity.variant_count == 8
    assert progress_state.current_progress_chromosome == "chr1"
    first_telemetry_plan = first_update.telemetry_plan
    assert [
        (event.event_name, event.level, event.chromosome, event.processed_chunk_count)
        for event in first_telemetry_plan.events
    ] == [
        ("chromosome_started", "info", "chr1", 1),
    ]
    assert first_telemetry_plan.progress.processed_chunk_count == 1
    assert first_telemetry_plan.progress.chromosome == "chr1"
    assert first_telemetry_plan.progress.chunk_identifier == 0
    assert first_telemetry_plan.progress.variant_start_index == 0
    assert first_telemetry_plan.progress.variant_stop_index == 8
    assert first_telemetry_plan.progress.variant_count == 8

    second_update = progress_state.record_processed_chunk(_core.build_callback_chunk_identity("chr2", 8, 10))
    assert second_update.processed_chunk_count == 2
    assert second_update.completed_chromosome == "chr1"
    assert second_update.completed_processed_chunk_count == 1
    assert second_update.started_chromosome == "chr2"
    assert progress_state.current_progress_chromosome == "chr2"
    assert [
        (event.event_name, event.level, event.chromosome, event.processed_chunk_count)
        for event in second_update.telemetry_plan.events
    ] == [
        ("chromosome_completed", "info", "chr1", 1),
        ("chromosome_started", "info", "chr2", 2),
    ]

    progress_completion = progress_state.finish_progress()
    assert progress_completion is not None
    assert progress_completion.chromosome == "chr2"
    assert progress_completion.processed_chunk_count == 2
    progress_completion_telemetry_event = progress_completion.telemetry_event
    assert progress_completion_telemetry_event.event_name == "chromosome_completed"
    assert progress_completion_telemetry_event.level == "info"
    assert progress_completion_telemetry_event.chromosome == "chr2"
    assert progress_completion_telemetry_event.processed_chunk_count == 2
    assert progress_state.finish_progress() is None

    progress_state.record_processed_chunk_without_progress()
    assert progress_state.processed_chunk_count == 3
    assert progress_state.current_progress_chromosome is None
    assert progress_state.finish_progress() is None


def test_emit_callback_progress_update_telemetry_uses_native_plan() -> None:
    class DisabledTelemetrySession:
        native_telemetry_session = None

    class LegacyTelemetrySession:
        def log_callback_progress_event(self, progress_event: _core.NativeCallbackProgressTelemetryEvent) -> None:
            raise AssertionError(progress_event)

        def log_progress(self, **fields: object) -> None:
            raise AssertionError(fields)

    telemetry_session = RecordingNativeCallbackTelemetrySession()
    progress_state = _core.NativeCallbackProgressState()
    progress_update = progress_state.record_processed_chunk(_core.build_callback_chunk_identity("chr1", 0, 8))

    _core.emit_callback_progress_update_telemetry(telemetry_session, progress_update)
    _core.emit_callback_progress_update_telemetry(None, None)
    _core.emit_callback_progress_update_telemetry(DisabledTelemetrySession(), progress_update)

    assert telemetry_session.progress_events == [("chromosome_started", "info", "chr1", 1)]
    assert telemetry_session.progress_records == [
        {
            "processed_chunk_count": 1,
            "chromosome": "chr1",
            "chunk_identifier": 0,
            "variant_start_index": 0,
            "variant_stop_index": 8,
            "variant_count": 8,
        }
    ]
    with pytest.raises(RuntimeError, match="Native callback progress plan selected a missing telemetry session"):
        _core.emit_callback_progress_update_telemetry(None, progress_update)
    with pytest.raises(TypeError, match="native telemetry session handle"):
        _core.emit_callback_progress_update_telemetry(LegacyTelemetrySession(), progress_update)


def test_emit_callback_progress_completion_telemetry_preserves_optional_session_behavior() -> None:
    class DisabledTelemetrySession:
        native_telemetry_session = None

    class LegacyTelemetrySession:
        def log_callback_progress_event(self, progress_event: _core.NativeCallbackProgressTelemetryEvent) -> None:
            raise AssertionError(progress_event)

    telemetry_session = RecordingNativeCallbackTelemetrySession()
    progress_state = _core.NativeCallbackProgressState()
    progress_state.record_processed_chunk(_core.build_callback_chunk_identity("chr2", 0, 4))
    progress_completion = progress_state.finish_progress()
    assert progress_completion is not None

    _core.emit_callback_progress_completion_telemetry(None, progress_completion)
    _core.emit_callback_progress_completion_telemetry(telemetry_session, None)
    _core.emit_callback_progress_completion_telemetry(DisabledTelemetrySession(), progress_completion)
    _core.emit_callback_progress_completion_telemetry(telemetry_session, progress_completion)

    assert telemetry_session.progress_events == [("chromosome_completed", "info", "chr2", 1)]
    with pytest.raises(TypeError, match="native telemetry session handle"):
        _core.emit_callback_progress_completion_telemetry(LegacyTelemetrySession(), progress_completion)


def test_emit_callback_progress_event_telemetry_uses_native_missing_session_policy() -> None:
    class DisabledTelemetrySession:
        native_telemetry_session = None

    class LegacyTelemetrySession:
        def log_callback_progress_event(self, progress_event: _core.NativeCallbackProgressTelemetryEvent) -> None:
            raise AssertionError(progress_event)

    telemetry_session = RecordingNativeCallbackTelemetrySession()
    progress_state = _core.NativeCallbackProgressState()
    progress_state.record_processed_chunk(_core.build_callback_chunk_identity("chr3", 0, 6))
    progress_completion = progress_state.finish_progress()
    assert progress_completion is not None
    progress_event = progress_completion.telemetry_event

    _core.emit_callback_progress_event_telemetry(telemetry_session, progress_event, "missing progress session")
    _core.emit_callback_progress_event_telemetry(None, None, "missing progress session")
    _core.emit_callback_progress_event_telemetry(DisabledTelemetrySession(), progress_event, "missing progress session")

    assert telemetry_session.progress_events == [("chromosome_completed", "info", "chr3", 1)]
    with pytest.raises(RuntimeError, match="missing progress session"):
        _core.emit_callback_progress_event_telemetry(None, progress_event, "missing progress session")
    with pytest.raises(TypeError, match="native telemetry session handle"):
        _core.emit_callback_progress_event_telemetry(
            LegacyTelemetrySession(),
            progress_event,
            "missing progress session",
        )


def test_native_callback_worker_shutdown_timeouts_return_native_defaults() -> None:
    scheduler_state = _core.NativeCallbackSchedulerState(
        staging_depth=1,
        native_callback_batch_size=1,
        result_in_flight_limit=None,
        dosage_buffer_limit=None,
    )
    finish_plan = scheduler_state.plan_worker_finish()
    abort_plan = scheduler_state.plan_worker_abort()

    assert finish_plan.dosage_stop_timeout_seconds == 60.0
    assert finish_plan.result_stop_timeout_seconds == 60.0
    assert finish_plan.dosage_join_timeout_seconds == 300.0
    assert finish_plan.result_join_timeout_seconds == 300.0
    assert abort_plan.dosage_stop_timeout_seconds == 1.0
    assert abort_plan.result_stop_timeout_seconds == 1.0


def test_resolve_callback_worker_backpressure_poll_timeout_seconds_returns_native_default() -> None:
    scheduler_state = _core.NativeCallbackSchedulerState(
        staging_depth=1,
        native_callback_batch_size=1,
        result_in_flight_limit=None,
        dosage_buffer_limit=None,
    )

    assert scheduler_state.backpressure_poll_timeout_seconds == 0.1


def test_resolve_callback_worker_stop_poll_timeout_seconds_caps_deadline_remaining_time() -> None:
    scheduler_state = _core.NativeCallbackSchedulerState(
        staging_depth=1,
        native_callback_batch_size=1,
        result_in_flight_limit=None,
        dosage_buffer_limit=None,
    )
    assert scheduler_state.mark_started() is True

    assert scheduler_state.plan_dosage_worker_stop_poll(1.0, is_worker_alive=True).poll_timeout_seconds == 0.1
    assert scheduler_state.plan_dosage_worker_stop_poll(0.05, is_worker_alive=True).poll_timeout_seconds == 0.05
    assert scheduler_state.plan_dosage_worker_stop_poll(0.0, is_worker_alive=True).poll_timeout_seconds == 0.0
    assert scheduler_state.plan_dosage_worker_stop_poll(-1.0, is_worker_alive=True).poll_timeout_seconds == 0.0


def test_should_attempt_callback_worker_stop_uses_native_lifecycle_policy() -> None:
    active_scheduler_state = _core.NativeCallbackSchedulerState(
        staging_depth=1,
        native_callback_batch_size=1,
        result_in_flight_limit=None,
        dosage_buffer_limit=None,
    )
    assert active_scheduler_state.mark_started() is True
    assert active_scheduler_state.plan_dosage_worker_stop(None, is_worker_alive=True).should_stop is True

    unstarted_scheduler_state = _core.NativeCallbackSchedulerState(
        staging_depth=1,
        native_callback_batch_size=1,
        result_in_flight_limit=None,
        dosage_buffer_limit=None,
    )
    assert unstarted_scheduler_state.plan_dosage_worker_stop(None, is_worker_alive=True).should_stop is False

    failed_scheduler_state = _core.NativeCallbackSchedulerState(
        staging_depth=1,
        native_callback_batch_size=1,
        result_in_flight_limit=None,
        dosage_buffer_limit=None,
    )
    assert failed_scheduler_state.mark_started() is True
    failed_scheduler_state.record_dosage_worker_error("dosage failed")
    assert failed_scheduler_state.plan_dosage_worker_stop(None, is_worker_alive=True).should_stop is False
    assert active_scheduler_state.plan_dosage_worker_stop(None, is_worker_alive=False).should_stop is False


def test_plan_callback_worker_join_uses_native_timeout_policy() -> None:
    scheduler_state = _core.NativeCallbackSchedulerState(
        staging_depth=1,
        native_callback_batch_size=1,
        result_in_flight_limit=None,
        dosage_buffer_limit=None,
    )
    assert scheduler_state.mark_started() is True

    dosage_join_plan = scheduler_state.plan_dosage_worker_join(timeout_seconds=None)
    assert dosage_join_plan.should_join is True
    assert dosage_join_plan.timeout_seconds == 60.0

    result_join_plan = scheduler_state.plan_result_worker_join(timeout_seconds=0.25)
    assert result_join_plan.should_join is True
    assert result_join_plan.timeout_seconds == 0.25

    unstarted_scheduler_state = _core.NativeCallbackSchedulerState(
        staging_depth=1,
        native_callback_batch_size=1,
        result_in_flight_limit=None,
        dosage_buffer_limit=None,
    )
    unstarted_join_plan = unstarted_scheduler_state.plan_result_worker_join(timeout_seconds=None)
    assert unstarted_join_plan.should_join is False
    assert unstarted_join_plan.timeout_seconds == 60.0


def test_plan_callback_worker_stop_uses_native_timeout_policy() -> None:
    scheduler_state = _core.NativeCallbackSchedulerState(
        staging_depth=1,
        native_callback_batch_size=1,
        result_in_flight_limit=None,
        dosage_buffer_limit=None,
    )
    assert scheduler_state.mark_started() is True

    dosage_stop_plan = scheduler_state.plan_dosage_worker_stop(timeout_seconds=None, is_worker_alive=True)
    assert dosage_stop_plan.should_stop is True
    assert dosage_stop_plan.timeout_seconds == 60.0

    result_stop_plan = scheduler_state.plan_result_worker_stop(timeout_seconds=0.25, is_worker_alive=True)
    assert result_stop_plan.should_stop is True
    assert result_stop_plan.timeout_seconds == 0.25

    scheduler_state.record_result_worker_error("writer failed")
    failed_worker_stop_plan = scheduler_state.plan_result_worker_stop(timeout_seconds=None, is_worker_alive=True)
    assert failed_worker_stop_plan.should_stop is False
    assert failed_worker_stop_plan.timeout_seconds == 60.0


def test_plan_callback_worker_finish_and_abort_use_native_timeout_policy() -> None:
    scheduler_state = _core.NativeCallbackSchedulerState(
        staging_depth=1,
        native_callback_batch_size=1,
        result_in_flight_limit=None,
        dosage_buffer_limit=None,
    )

    finish_plan = scheduler_state.plan_worker_finish()
    assert finish_plan.finish_actions == [
        "stop_dosage_worker",
        "join_dosage_worker",
        "stop_result_worker",
        "join_result_worker",
        "raise_worker_error",
        "complete_progress",
        "emit_binary_correction_summary",
    ]
    assert finish_plan.stop_dosage_worker is True
    assert finish_plan.join_dosage_worker is True
    assert finish_plan.stop_result_worker is True
    assert finish_plan.join_result_worker is True
    assert finish_plan.raise_worker_error is True
    assert finish_plan.complete_progress is True
    assert finish_plan.emit_binary_correction_summary is True
    assert finish_plan.dosage_stop_timeout_seconds == 60.0
    assert finish_plan.dosage_join_timeout_seconds == 300.0
    assert finish_plan.result_stop_timeout_seconds == 60.0
    assert finish_plan.result_join_timeout_seconds == 300.0

    abort_plan = scheduler_state.plan_worker_abort()
    assert abort_plan.abort_actions == ["stop_dosage_worker", "stop_result_worker"]
    assert abort_plan.stop_dosage_worker is True
    assert abort_plan.stop_result_worker is True
    assert abort_plan.dosage_stop_timeout_seconds == 1.0
    assert abort_plan.result_stop_timeout_seconds == 1.0


def test_plan_callback_worker_stop_poll_uses_native_loop_policy() -> None:
    scheduler_state = _core.NativeCallbackSchedulerState(
        staging_depth=1,
        native_callback_batch_size=1,
        result_in_flight_limit=None,
        dosage_buffer_limit=None,
    )
    assert scheduler_state.mark_started() is True

    active_poll_plan = scheduler_state.plan_dosage_worker_stop_poll(
        remaining_timeout_seconds=1.0,
        is_worker_alive=True,
    )
    assert active_poll_plan.should_stop is True
    assert active_poll_plan.poll_timeout_seconds == 0.1

    scheduler_state.record_dosage_worker_error("dosage failed")
    failed_poll_plan = scheduler_state.plan_dosage_worker_stop_poll(
        remaining_timeout_seconds=0.05,
        is_worker_alive=True,
    )
    assert failed_poll_plan.should_stop is False
    assert failed_poll_plan.poll_timeout_seconds == 0.05

    scheduler_state.clear_dosage_worker_error()
    expired_poll_plan = scheduler_state.plan_dosage_worker_stop_poll(
        remaining_timeout_seconds=-1.0,
        is_worker_alive=True,
    )
    assert expired_poll_plan.should_stop is True
    assert expired_poll_plan.poll_timeout_seconds == 0.0


def test_format_callback_worker_error_messages_uses_native_policy() -> None:
    scheduler_state = _core.NativeCallbackSchedulerState(
        staging_depth=1,
        native_callback_batch_size=1,
        result_in_flight_limit=None,
        dosage_buffer_limit=None,
    )

    scheduler_state.record_dosage_worker_error("dosage failed")
    scheduler_state.record_result_worker_error("writer failed")

    assert scheduler_state.dosage_worker_error_message == "native pipeline callback worker failed: dosage failed"
    assert scheduler_state.result_worker_error_message == "native pipeline result writer worker failed: writer failed"


def test_resolve_native_callback_queue_limits_uses_native_capacity_policy() -> None:
    scheduler_state = _core.NativeCallbackSchedulerState(
        staging_depth=3,
        native_callback_batch_size=1,
        result_in_flight_limit=None,
        dosage_buffer_limit=None,
    )
    assert scheduler_state.dosage_queue_depth == 3
    assert scheduler_state.result_queue_depth == 3
    assert scheduler_state.result_in_flight_limit == 4
    assert scheduler_state.dosage_buffer_limit == 4

    explicit_scheduler_state = _core.NativeCallbackSchedulerState(
        staging_depth=3,
        native_callback_batch_size=2,
        result_in_flight_limit=7,
        dosage_buffer_limit=8,
    )
    assert explicit_scheduler_state.result_in_flight_limit == 7
    assert explicit_scheduler_state.dosage_buffer_limit == 8

    with pytest.raises(ValueError, match="staging_depth must be positive"):
        _core.NativeCallbackSchedulerState(
            staging_depth=0,
            native_callback_batch_size=1,
            result_in_flight_limit=None,
            dosage_buffer_limit=None,
        )
    with pytest.raises(ValueError, match="native_callback_batch_size must be positive"):
        _core.NativeCallbackSchedulerState(
            staging_depth=1,
            native_callback_batch_size=0,
            result_in_flight_limit=None,
            dosage_buffer_limit=None,
        )
    with pytest.raises(ValueError, match="result_in_flight_limit must be positive"):
        _core.NativeCallbackSchedulerState(
            staging_depth=1,
            native_callback_batch_size=1,
            result_in_flight_limit=0,
            dosage_buffer_limit=None,
        )
    with pytest.raises(ValueError, match="dosage_buffer_limit must be positive"):
        _core.NativeCallbackSchedulerState(
            staging_depth=1,
            native_callback_batch_size=1,
            result_in_flight_limit=None,
            dosage_buffer_limit=0,
        )
    with pytest.raises(ValueError, match="effective dosage_buffer_limit"):
        _core.NativeCallbackSchedulerState(
            staging_depth=1,
            native_callback_batch_size=3,
            result_in_flight_limit=None,
            dosage_buffer_limit=2,
        )


def test_plan_callback_queue_stage_observation_uses_native_timing_policy() -> None:
    scheduler_state = _core.NativeCallbackSchedulerState(
        staging_depth=3,
        native_callback_batch_size=2,
        result_in_flight_limit=7,
        dosage_buffer_limit=8,
    )

    queue_observation_plan = scheduler_state.plan_queue_stage_observation(
        queue_name="dosage_queue",
        operation_name="put",
        elapsed_seconds=0.25,
        blocked=False,
    )
    assert queue_observation_plan.queue_name == "dosage_queue"
    assert queue_observation_plan.operation_name == "put"
    assert queue_observation_plan.stage_name == "callback_queue_put"
    assert queue_observation_plan.blocked_seconds == 0.0

    queue_backpressure_observation = scheduler_state.plan_queue_stage_backpressure_observation(
        queue_name="dosage_queue",
        operation_name="put",
        queue_depth=2,
        queue_capacity=3,
        elapsed_seconds=0.25,
        blocked=False,
    )
    assert queue_backpressure_observation.queue_name == "dosage_queue"
    assert queue_backpressure_observation.operation_name == "put"
    assert queue_backpressure_observation.stage_name == "callback_queue_put"
    assert queue_backpressure_observation.queue_depth == 2
    assert queue_backpressure_observation.queue_capacity == 3
    assert queue_backpressure_observation.elapsed_seconds == 0.25
    assert queue_backpressure_observation.blocked_seconds == 0.0

    blocked_observation_plan = scheduler_state.plan_queue_stage_observation(
        queue_name="result_in_flight_slots",
        operation_name="producer_blocking",
        elapsed_seconds=0.5,
        blocked=True,
    )
    assert blocked_observation_plan.stage_name == "result_in_flight_producer_blocking"
    assert blocked_observation_plan.blocked_seconds == 0.5

    with pytest.raises(ValueError, match="Unsupported callback queue stage operation"):
        scheduler_state.plan_queue_stage_observation(
            queue_name="unknown_queue",
            operation_name="put",
            elapsed_seconds=0.25,
            blocked=False,
        )


def test_native_callback_scheduler_state_plans_current_queue_observations() -> None:
    scheduler_state = _core.NativeCallbackSchedulerState(
        staging_depth=3,
        native_callback_batch_size=2,
        result_in_flight_limit=7,
        dosage_buffer_limit=8,
    )

    assert scheduler_state.acquire_dosage_queue_slot() is True
    dosage_queue_observation = scheduler_state.plan_current_queue_stage_backpressure_observation(
        queue_name="dosage_queue",
        operation_name="producer_blocking",
        elapsed_seconds=0.5,
        blocked=True,
    )
    assert dosage_queue_observation.queue_name == "dosage_queue"
    assert dosage_queue_observation.operation_name == "producer_blocking"
    assert dosage_queue_observation.stage_name == "callback_queue_producer_blocking"
    assert dosage_queue_observation.queue_depth == 1
    assert dosage_queue_observation.queue_capacity == 3
    assert dosage_queue_observation.elapsed_seconds == 0.5
    assert dosage_queue_observation.blocked_seconds == 0.5

    assert scheduler_state.acquire_result_in_flight_slot() is True
    result_slot_observation = scheduler_state.plan_current_queue_backpressure_observation(
        queue_name="result_in_flight_slots",
        operation_name="release",
        elapsed_seconds=0.25,
        blocked=False,
    )
    assert result_slot_observation.queue_name == "result_in_flight_slots"
    assert result_slot_observation.operation_name == "release"
    assert result_slot_observation.queue_depth == 1
    assert result_slot_observation.queue_capacity == 7
    assert result_slot_observation.elapsed_seconds == 0.25
    assert result_slot_observation.blocked_seconds == 0.0

    dosage_buffer_observation = scheduler_state.plan_dosage_buffer_pool_backpressure_observation(
        operation_name="reuse",
        free_buffer_count=4,
        elapsed_seconds=0.25,
        blocked=False,
    )
    assert dosage_buffer_observation.queue_name == "dosage_buffer_pool"
    assert dosage_buffer_observation.operation_name == "reuse"
    assert dosage_buffer_observation.queue_depth == 4
    assert dosage_buffer_observation.queue_capacity == 8
    assert dosage_buffer_observation.elapsed_seconds == 0.25
    assert dosage_buffer_observation.blocked_seconds == 0.0

    dosage_buffer_stage_observation = scheduler_state.plan_dosage_buffer_pool_stage_backpressure_observation(
        operation_name="consumer_wait",
        free_buffer_count=2,
        elapsed_seconds=0.5,
        blocked=True,
    )
    assert dosage_buffer_stage_observation.queue_name == "dosage_buffer_pool"
    assert dosage_buffer_stage_observation.operation_name == "consumer_wait"
    assert dosage_buffer_stage_observation.stage_name == "dosage_buffer_pool_consumer_wait"
    assert dosage_buffer_stage_observation.queue_depth == 2
    assert dosage_buffer_stage_observation.queue_capacity == 8
    assert dosage_buffer_stage_observation.elapsed_seconds == 0.5
    assert dosage_buffer_stage_observation.blocked_seconds == 0.5


def test_plan_callback_queue_operation_observation_uses_native_timing_policy() -> None:
    scheduler_state = _core.NativeCallbackSchedulerState(
        staging_depth=3,
        native_callback_batch_size=2,
        result_in_flight_limit=7,
        dosage_buffer_limit=8,
    )

    pool_observation_plan = scheduler_state.plan_queue_operation_observation(
        queue_name="dosage_buffer_pool",
        operation_name="reuse",
        elapsed_seconds=0.25,
        blocked=False,
    )
    assert pool_observation_plan.queue_name == "dosage_buffer_pool"
    assert pool_observation_plan.operation_name == "reuse"
    assert pool_observation_plan.blocked_seconds == 0.0

    pool_backpressure_observation = scheduler_state.plan_queue_backpressure_observation(
        queue_name="dosage_buffer_pool",
        operation_name="reuse",
        queue_depth=1,
        queue_capacity=2,
        elapsed_seconds=0.25,
        blocked=False,
    )
    assert pool_backpressure_observation.queue_name == "dosage_buffer_pool"
    assert pool_backpressure_observation.operation_name == "reuse"
    assert pool_backpressure_observation.queue_depth == 1
    assert pool_backpressure_observation.queue_capacity == 2
    assert pool_backpressure_observation.elapsed_seconds == 0.25
    assert pool_backpressure_observation.blocked_seconds == 0.0

    blocked_observation_plan = scheduler_state.plan_queue_operation_observation(
        queue_name="result_in_flight_slots",
        operation_name="release",
        elapsed_seconds=0.5,
        blocked=True,
    )
    assert blocked_observation_plan.queue_name == "result_in_flight_slots"
    assert blocked_observation_plan.operation_name == "release"
    assert blocked_observation_plan.blocked_seconds == 0.5

    with pytest.raises(ValueError, match="Unsupported callback queue operation"):
        scheduler_state.plan_queue_operation_observation(
            queue_name="dosage_buffer_pool",
            operation_name="unknown_operation",
            elapsed_seconds=0.25,
            blocked=False,
        )


def test_plan_dosage_buffer_reuse_uses_native_shape_policy() -> None:
    scheduler_state = _core.NativeCallbackSchedulerState(
        staging_depth=1,
        native_callback_batch_size=1,
        result_in_flight_limit=None,
        dosage_buffer_limit=None,
    )

    exact_reuse_plan = scheduler_state.plan_dosage_buffer_reuse(
        buffered_shape=(2, 3),
        expected_shape=(2, 3),
    )
    assert exact_reuse_plan is not None
    assert exact_reuse_plan.requires_slice is False
    assert exact_reuse_plan.slice_dimensions == [2, 3]

    sliced_reuse_plan = scheduler_state.plan_dosage_buffer_reuse(
        buffered_shape=(4, 5),
        expected_shape=(2, 3),
    )
    assert sliced_reuse_plan is not None
    assert sliced_reuse_plan.requires_slice is True
    assert sliced_reuse_plan.slice_dimensions == [2, 3]

    assert scheduler_state.plan_dosage_buffer_reuse(buffered_shape=(2, 3), expected_shape=(2, 3, 1)) is None
    assert scheduler_state.plan_dosage_buffer_reuse(buffered_shape=(2, 3), expected_shape=(3, 2)) is None


def test_plan_variant_major_dosage_batch_handoff_uses_native_batch_policy() -> None:
    scheduler_state = _core.NativeCallbackSchedulerState(
        staging_depth=1,
        native_callback_batch_size=1,
        result_in_flight_limit=None,
        dosage_buffer_limit=None,
    )

    batch_handoff_plan = scheduler_state.plan_variant_major_dosage_batch_handoff(
        metadata_count=2,
        genotype_matrix_by_variant_count=2,
        chunk_stats_count=2,
    )
    assert batch_handoff_plan.chunk_count == 2

    with pytest.raises(ValueError, match="identical lengths"):
        scheduler_state.plan_variant_major_dosage_batch_handoff(
            metadata_count=2,
            genotype_matrix_by_variant_count=1,
            chunk_stats_count=2,
        )
    with pytest.raises(ValueError, match="at least one chunk"):
        scheduler_state.plan_variant_major_dosage_batch_handoff(
            metadata_count=0,
            genotype_matrix_by_variant_count=0,
            chunk_stats_count=0,
        )


def test_plan_dosage_work_handoff_uses_native_policy() -> None:
    scheduler_state = _core.NativeCallbackSchedulerState(
        staging_depth=1,
        native_callback_batch_size=1,
        result_in_flight_limit=None,
        dosage_buffer_limit=None,
    )
    handoff_plan = scheduler_state.plan_dosage_work_handoff(chunk_count=2)
    assert handoff_plan.chunk_count == 2

    scheduler_handoff_plan = scheduler_state.plan_dosage_work_handoff(chunk_count=1)
    assert scheduler_handoff_plan.chunk_count == 1

    with pytest.raises(ValueError, match="at least one chunk"):
        scheduler_state.plan_dosage_work_handoff(chunk_count=0)


def test_plan_result_write_handoff_uses_native_policy() -> None:
    scheduler_state = _core.NativeCallbackSchedulerState(
        staging_depth=1,
        native_callback_batch_size=1,
        result_in_flight_limit=None,
        dosage_buffer_limit=None,
    )
    result_handoff_plan = scheduler_state.plan_result_write_handoff(has_result_work_item=True)
    assert result_handoff_plan.should_enqueue is True
    assert result_handoff_plan.has_result_work_item is True
    assert result_handoff_plan.is_stop_signal is False

    stop_handoff_plan = scheduler_state.plan_result_write_handoff(has_result_work_item=False)
    assert stop_handoff_plan.should_enqueue is True
    assert stop_handoff_plan.has_result_work_item is False
    assert stop_handoff_plan.is_stop_signal is True


def test_native_dosage_buffer_pool_state_tracks_capacity_and_ownership() -> None:
    buffer_pool_state = _core.NativeDosageBufferPoolState(buffer_limit=2)

    assert buffer_pool_state.buffer_limit == 2
    assert buffer_pool_state.allocated_count == 0
    assert buffer_pool_state.buffer_identifiers == []
    assert buffer_pool_state.has_available_slot() is True
    assert buffer_pool_state.register_buffer(11) is True
    assert buffer_pool_state.owns_buffer(11) is True
    assert buffer_pool_state.register_buffer(11) is False
    assert buffer_pool_state.register_buffer(7) is True
    assert buffer_pool_state.allocated_count == 2
    assert buffer_pool_state.buffer_identifiers == [7, 11]
    assert buffer_pool_state.has_available_slot() is False
    assert buffer_pool_state.register_buffer(13) is False
    assert buffer_pool_state.discard_buffer(11) is True
    assert buffer_pool_state.owns_buffer(11) is False
    assert buffer_pool_state.has_available_slot() is True
    assert buffer_pool_state.discard_buffer(99) is False


def test_native_result_in_flight_slot_state_tracks_capacity() -> None:
    slot_state = _core.NativeResultInFlightSlotState(slot_limit=2)

    assert slot_state.slot_limit == 2
    assert slot_state.occupied_count == 0
    assert slot_state.has_available_slot() is True
    assert slot_state.acquire_slot() is True
    assert slot_state.occupied_count == 1
    assert slot_state.acquire_slot() is True
    assert slot_state.occupied_count == 2
    assert slot_state.has_available_slot() is False
    assert slot_state.acquire_slot() is False
    assert slot_state.release_slot() is True
    assert slot_state.occupied_count == 1
    assert slot_state.release_slot() is True
    assert slot_state.occupied_count == 0
    assert slot_state.release_slot() is False


def test_resolve_bgen_delivery_method_uses_native_alignment_precedence() -> None:
    assert (
        _core.plan_bgen_delivery_invocation(
            callback_batch_size=None,
            variant_major_packed8_probability_pairs=False,
            has_native_multi_aligned_sample_data=True,
            has_native_aligned_sample_data=True,
        ).delivery_method
        == "dosage_native_multi_aligned_samples"
    )
    assert (
        _core.plan_bgen_delivery_invocation(
            callback_batch_size=None,
            variant_major_packed8_probability_pairs=False,
            has_native_multi_aligned_sample_data=False,
            has_native_aligned_sample_data=True,
        ).delivery_method
        == "dosage_native_aligned_samples"
    )
    assert (
        _core.plan_bgen_delivery_invocation(
            callback_batch_size=None,
            variant_major_packed8_probability_pairs=False,
            has_native_multi_aligned_sample_data=False,
            has_native_aligned_sample_data=False,
        ).delivery_method
        == "dosage_sample_indices"
    )
    assert (
        _core.plan_bgen_delivery_invocation(
            callback_batch_size=None,
            variant_major_packed8_probability_pairs=True,
            has_native_multi_aligned_sample_data=True,
            has_native_aligned_sample_data=True,
        ).delivery_method
        == "packed8_native_multi_aligned_samples"
    )
    assert (
        _core.plan_bgen_delivery_invocation(
            callback_batch_size=None,
            variant_major_packed8_probability_pairs=True,
            has_native_multi_aligned_sample_data=False,
            has_native_aligned_sample_data=True,
        ).delivery_method
        == "packed8_native_aligned_samples"
    )
    assert (
        _core.plan_bgen_delivery_invocation(
            callback_batch_size=None,
            variant_major_packed8_probability_pairs=True,
            has_native_multi_aligned_sample_data=False,
            has_native_aligned_sample_data=False,
        ).delivery_method
        == "packed8_sample_indices"
    )


def test_resolve_writer_finish_thread_count_enforces_native_cleanup_policy() -> None:
    assert _core.resolve_writer_finish_thread_count(0, 0) == 0
    assert _core.resolve_writer_finish_thread_count(3, 2) == 2
    assert _core.resolve_writer_finish_thread_count(3, 5) == 3
    with pytest.raises(ValueError, match="Writer finish thread count must be positive"):
        _core.resolve_writer_finish_thread_count(1, 0)


def test_plan_writer_finish_execution_uses_native_cleanup_policy() -> None:
    empty_finish_plan = _core.plan_writer_finish_execution(writer_session_count=0, requested_thread_count=0)
    assert empty_finish_plan.writer_session_count == 0
    assert empty_finish_plan.thread_count == 0
    assert empty_finish_plan.has_writer_sessions is False
    assert empty_finish_plan.uses_parallel_finish is False

    serial_finish_plan = _core.plan_writer_finish_execution(writer_session_count=1, requested_thread_count=1)
    assert serial_finish_plan.writer_session_count == 1
    assert serial_finish_plan.thread_count == 1
    assert serial_finish_plan.has_writer_sessions is True
    assert serial_finish_plan.uses_parallel_finish is False

    parallel_finish_plan = _core.plan_writer_finish_execution(writer_session_count=3, requested_thread_count=2)
    assert parallel_finish_plan.writer_session_count == 3
    assert parallel_finish_plan.thread_count == 2
    assert parallel_finish_plan.has_writer_sessions is True
    assert parallel_finish_plan.uses_parallel_finish is True

    with pytest.raises(ValueError, match="Writer finish thread count must be positive"):
        _core.plan_writer_finish_execution(writer_session_count=1, requested_thread_count=0)


def test_plan_bgen_delivery_cleanup_uses_native_lifecycle_policy() -> None:
    success_plan = _core.plan_bgen_delivery_cleanup(cleanup_outcome="success", callback_finished=False)
    assert success_plan.cleanup_actions == [
        "drain_callback",
        "finish_writer_sessions",
        "write_stage_timing_snapshot",
    ]
    assert success_plan.drain_callback is True
    assert success_plan.finish_writer_sessions is True
    assert success_plan.finish_interrupted_writer_sessions is False
    assert success_plan.abort_callback is False
    assert success_plan.abort_writer_sessions is False
    assert success_plan.write_stage_timing_snapshot is True

    interrupted_pending_callback_plan = _core.plan_bgen_delivery_cleanup(
        cleanup_outcome="interrupted",
        callback_finished=False,
    )
    assert interrupted_pending_callback_plan.drain_callback is True
    assert interrupted_pending_callback_plan.finish_writer_sessions is False
    assert interrupted_pending_callback_plan.finish_interrupted_writer_sessions is True
    assert interrupted_pending_callback_plan.abort_callback is False
    assert interrupted_pending_callback_plan.abort_writer_sessions is False
    assert interrupted_pending_callback_plan.cleanup_actions == [
        "drain_callback",
        "finish_interrupted_writer_sessions",
        "write_stage_timing_snapshot",
    ]

    interrupted_finished_callback_plan = _core.plan_bgen_delivery_cleanup(
        cleanup_outcome="interrupted",
        callback_finished=True,
    )
    assert interrupted_finished_callback_plan.drain_callback is False
    assert interrupted_finished_callback_plan.finish_interrupted_writer_sessions is True
    assert interrupted_finished_callback_plan.cleanup_actions == [
        "finish_interrupted_writer_sessions",
        "write_stage_timing_snapshot",
    ]

    failure_plan = _core.plan_bgen_delivery_cleanup(cleanup_outcome="failure", callback_finished=False)
    assert failure_plan.drain_callback is False
    assert failure_plan.finish_writer_sessions is False
    assert failure_plan.finish_interrupted_writer_sessions is False
    assert failure_plan.abort_callback is True
    assert failure_plan.abort_writer_sessions is True
    assert failure_plan.write_stage_timing_snapshot is True
    assert failure_plan.cleanup_actions == [
        "abort_callback",
        "abort_writer_sessions",
        "write_stage_timing_snapshot",
    ]

    cleanup_failure_plan = _core.plan_bgen_delivery_cleanup(
        cleanup_outcome="interrupted_cleanup_failure",
        callback_finished=False,
    )
    assert cleanup_failure_plan.abort_callback is True
    assert cleanup_failure_plan.abort_writer_sessions is True

    with pytest.raises(ValueError, match="Unsupported BGEN delivery cleanup outcome"):
        _core.plan_bgen_delivery_cleanup(cleanup_outcome="unknown", callback_finished=False)


def test_plan_output_write_methods_use_native_dtype_policy() -> None:
    native_float64_write_plan = _core.plan_single_trait_output_write(
        is_native_writer_session=True,
        output_statistic_dtype="float64",
    )
    assert native_float64_write_plan.method_name == "write_regenie2_native_chunk_f64"
    assert native_float64_write_plan.uses_float64_native_writer is True

    fallback_float64_write_plan = _core.plan_single_trait_output_write(
        is_native_writer_session=False,
        output_statistic_dtype="float64",
    )
    assert fallback_float64_write_plan.method_name == "write_regenie2_native_chunk"
    assert fallback_float64_write_plan.uses_float64_native_writer is False

    native_multi_write_plan = _core.plan_multi_trait_output_write(
        active_trait_count=2,
        all_writer_sessions_native=True,
        output_statistic_dtype="float64",
    )
    assert native_multi_write_plan.active_trait_count == 2
    assert native_multi_write_plan.use_native_multi_writer is True
    assert native_multi_write_plan.uses_float64_native_writer is True

    fallback_multi_write_plan = _core.plan_multi_trait_output_write(
        active_trait_count=2,
        all_writer_sessions_native=False,
        output_statistic_dtype="float64",
    )
    assert fallback_multi_write_plan.use_native_multi_writer is False
    assert fallback_multi_write_plan.uses_float64_native_writer is False

    with pytest.raises(ValueError, match="Unsupported public statistic output dtype"):
        _core.plan_single_trait_output_write(is_native_writer_session=True, output_statistic_dtype="float16")
    with pytest.raises(ValueError, match="Unsupported public statistic output dtype"):
        _core.plan_multi_trait_output_write(
            active_trait_count=1,
            all_writer_sessions_native=True,
            output_statistic_dtype="float16",
        )


def test_regenie2_run_engine_required_chromosomes_returns_boundary_labels() -> None:
    engine = _core.Regenie2RunEngine(str(HAPLOTYPES_BGEN_PATH), chunk_size=2)

    assert engine.required_chromosomes() == ["1"]
    assert engine.required_chromosomes(variant_limit=1) == ["1"]
    assert engine.required_chromosomes(variant_limit=0) == []


def test_regenie2_run_engine_buffered_chunks_deliver_preprocessed_variant_major_dosage_chunks() -> None:
    class RecordingCallback:
        def __init__(self) -> None:
            self.chunk_shapes: list[tuple[int, int, int]] = []
            self.free_buffers: list[np.ndarray] = []

        def acquire_variant_major_dosage_buffer(self, variant_count: int, sample_count: int) -> np.ndarray:
            if self.free_buffers:
                return self.free_buffers.pop()
            return np.empty((variant_count, sample_count), dtype=np.float32, order="C")

        def compute_preprocessed_variant_major_dosage_chunk(
            self,
            metadata: _core.VariantMetadata,
            genotype_matrix: np.ndarray,
            chunk_stats: _core.ChunkStats,
        ) -> None:
            self.chunk_shapes.append(
                (
                    metadata.variant_start_index,
                    genotype_matrix.shape[0],
                    genotype_matrix.shape[1],
                )
            )
            assert metadata.chromosome_label == "1"
            assert not np.isnan(genotype_matrix).any()
            np.testing.assert_allclose(chunk_stats.allele_one_frequency, genotype_matrix.mean(axis=1) / 2.0)
            np.testing.assert_array_equal(chunk_stats.observation_count, np.full(genotype_matrix.shape[0], 4))
            self.free_buffers.append(genotype_matrix)

    callback = RecordingCallback()
    engine = _core.Regenie2RunEngine(str(HAPLOTYPES_BGEN_PATH), chunk_size=2)

    processed_chunk_count = engine.run_bgen_variant_major_dosage_buffered_chunks(
        np.arange(4, dtype=np.int64),
        callback,
    )

    assert processed_chunk_count == 2
    assert callback.chunk_shapes == [(0, 2, 4), (2, 2, 4)]


def test_regenie2_run_engine_buffered_chunks_deliver_preprocessed_variant_major_dosage_batches() -> None:
    class RecordingBatchCallback:
        def __init__(self) -> None:
            self.chunk_batches: list[tuple[int, ...]] = []
            self.free_buffers: list[np.ndarray] = []

        def acquire_variant_major_dosage_buffer(self, variant_count: int, sample_count: int) -> np.ndarray:
            if self.free_buffers:
                return self.free_buffers.pop()
            return np.empty((variant_count, sample_count), dtype=np.float32, order="C")

        def compute_preprocessed_variant_major_dosage_chunk_batch(
            self,
            metadata_batch: list[_core.VariantMetadata],
            genotype_matrix_batch: list[np.ndarray],
            chunk_stats_batch: list[_core.ChunkStats],
        ) -> None:
            self.chunk_batches.append(tuple(metadata.variant_start_index for metadata in metadata_batch))
            for metadata, genotype_matrix, chunk_stats in zip(metadata_batch, genotype_matrix_batch, chunk_stats_batch):
                assert metadata.chromosome_label == "1"
                assert genotype_matrix.shape == (1, 4)
                assert not np.isnan(genotype_matrix).any()
                np.testing.assert_allclose(chunk_stats.allele_one_frequency, genotype_matrix.mean(axis=1) / 2.0)
                np.testing.assert_array_equal(chunk_stats.observation_count, np.full(genotype_matrix.shape[0], 4))
                self.free_buffers.append(genotype_matrix)

    callback = RecordingBatchCallback()
    engine = _core.Regenie2RunEngine(str(HAPLOTYPES_BGEN_PATH), chunk_size=1)

    processed_chunk_count = engine.run_bgen_variant_major_dosage_buffered_chunks(
        np.arange(4, dtype=np.int64),
        callback,
        callback_batch_size=2,
    )

    assert processed_chunk_count == 4
    assert callback.chunk_batches == [(0, 1), (2, 3)]


def test_regenie2_run_engine_buffered_chunks_deliver_preprocessed_variant_major_packed8_chunks() -> None:
    class RecordingPackedCallback:
        def __init__(self) -> None:
            self.chunk_shapes: list[tuple[int, int, int, int]] = []
            self.free_buffers: list[np.ndarray] = []

        def acquire_variant_major_packed8_probability_pair_buffer(
            self,
            variant_count: int,
            sample_count: int,
        ) -> np.ndarray:
            if self.free_buffers:
                return self.free_buffers.pop()
            return np.empty((variant_count, sample_count, 2), dtype=np.uint8, order="C")

        def compute_preprocessed_variant_major_packed8_probability_pair_chunk(
            self,
            metadata: _core.VariantMetadata,
            packed_probability_pairs: np.ndarray,
            chunk_stats: _core.ChunkStats,
        ) -> None:
            self.chunk_shapes.append(
                (
                    metadata.variant_start_index,
                    packed_probability_pairs.shape[0],
                    packed_probability_pairs.shape[1],
                    packed_probability_pairs.shape[2],
                )
            )
            assert metadata.chromosome_label == "1"
            assert packed_probability_pairs.dtype == np.uint8
            assert not chunk_stats.has_missing_values
            np.testing.assert_array_equal(
                chunk_stats.observation_count,
                np.full(packed_probability_pairs.shape[0], 4),
            )
            self.free_buffers.append(packed_probability_pairs)

    callback = RecordingPackedCallback()
    engine = _core.Regenie2RunEngine(
        str(TRUSTED_PACKED8_BGEN_PATH),
        chunk_size=2,
        variant_limit=4,
        trusted_no_missing_diploid=True,
    )
    engine.validate_trusted_no_missing_diploid()

    processed_chunk_count = engine.run_bgen_variant_major_packed8_probability_pair_buffered_chunks(
        np.arange(4, dtype=np.int64),
        callback,
    )

    assert processed_chunk_count == 2
    assert callback.chunk_shapes == [(0, 2, 4, 2), (2, 2, 4, 2)]


def test_regenie2_run_engine_variant_major_chunks_support_untrusted_bgen() -> None:
    class RecordingCallback:
        def __init__(self) -> None:
            self.chunk_shapes: list[tuple[int, int, int]] = []
            self.free_buffers: list[np.ndarray] = []

        def acquire_variant_major_dosage_buffer(self, variant_count: int, sample_count: int) -> np.ndarray:
            if self.free_buffers:
                return self.free_buffers.pop()
            return np.empty((variant_count, sample_count), dtype=np.float32, order="C")

        def compute_preprocessed_variant_major_dosage_chunk(
            self,
            metadata: _core.VariantMetadata,
            genotype_matrix_by_variant: np.ndarray,
            chunk_stats: _core.ChunkStats,
        ) -> None:
            self.chunk_shapes.append(
                (
                    metadata.variant_start_index,
                    genotype_matrix_by_variant.shape[0],
                    genotype_matrix_by_variant.shape[1],
                )
            )
            assert not np.isnan(genotype_matrix_by_variant).any()
            np.testing.assert_allclose(chunk_stats.allele_one_frequency, genotype_matrix_by_variant.mean(axis=1) / 2.0)
            np.testing.assert_array_equal(
                chunk_stats.observation_count,
                np.full(genotype_matrix_by_variant.shape[0], 4),
            )
            np.testing.assert_allclose(chunk_stats.dosage_sum, genotype_matrix_by_variant.sum(axis=1))
            self.free_buffers.append(genotype_matrix_by_variant)

    callback = RecordingCallback()
    engine = _core.Regenie2RunEngine(str(HAPLOTYPES_BGEN_PATH), chunk_size=2, trusted_no_missing_diploid=False)

    processed_chunk_count = engine.run_bgen_variant_major_dosage_buffered_chunks(
        np.arange(4, dtype=np.int64),
        callback,
    )

    assert processed_chunk_count == 2
    assert callback.chunk_shapes == [(0, 2, 4), (2, 2, 4)]


def test_regenie_prediction_source_loads_aligned_loco_predictions(tmp_path: Path) -> None:
    loco_path = tmp_path / "trait.loco"
    loco_path.write_text("FID_IID 0_A 0_B 0_C\nchr22 0.1 0.2 0.3\n01 1.0 2.0 3.0\n")
    prediction_list_path = tmp_path / "trait_pred.list"
    prediction_list_path.write_text(f"trait {loco_path}\n")

    prediction_source = _core.RegeniePredictionSource(
        str(prediction_list_path),
        "trait",
        ["0", "0"],
        ["C", "A"],
    )

    assert prediction_source.get_chromosome_predictions("22").dtype == np.float32
    np.testing.assert_allclose(prediction_source.get_chromosome_predictions("chr22"), [0.3, 0.1], atol=1e-6)
    np.testing.assert_allclose(prediction_source.get_chromosome_predictions("1"), [3.0, 1.0], atol=1e-6)


def test_regenie_prediction_source_loads_from_native_aligned_sample_data(tmp_path: Path) -> None:
    loco_path = tmp_path / "trait.loco"
    loco_path.write_text("FID_IID 0_A 0_B 0_C\nchr22 0.1 0.2 0.3\n")
    prediction_list_path = tmp_path / "trait_pred.list"
    prediction_list_path.write_text(f"trait {loco_path}\n")
    phenotype_path = tmp_path / "phenotypes.tsv"
    phenotype_path.write_text("IID\ttrait\nC\t1.0\nA\t2.0\n")
    native_aligned_sample_data = _core.align_sample_data(
        np.asarray([0, 1], dtype=np.int64),
        ["0", "0"],
        ["C", "A"],
        str(phenotype_path),
        "trait",
    )

    prediction_source = _core.RegeniePredictionSource.from_native_aligned_sample_data(
        str(prediction_list_path),
        "trait",
        native_aligned_sample_data,
    )

    np.testing.assert_allclose(prediction_source.get_chromosome_predictions("chr22"), [0.3, 0.1], atol=1e-6)


def test_multi_regenie_prediction_source_returns_trait_major_loco_matrix(tmp_path: Path) -> None:
    trait_a_loco_path = tmp_path / "trait_a.loco"
    trait_a_loco_path.write_text("FID_IID 0_A 0_B 0_C\nchr22 0.1 0.2 0.3\n")
    trait_b_loco_path = tmp_path / "trait_b.loco"
    trait_b_loco_path.write_text("FID_IID 0_A 0_B 0_C\nchr22 1.1 1.2 1.3\n")
    prediction_list_path = tmp_path / "pred.list"
    prediction_list_path.write_text(f"trait_a {trait_a_loco_path}\ntrait_b {trait_b_loco_path}\n")
    phenotype_path = tmp_path / "phenotypes.tsv"
    phenotype_path.write_text("IID\ttrait_a\ttrait_b\nC\t1.0\t2.0\nA\t3.0\t4.0\n")
    native_multi_aligned_sample_data = _core.align_multi_sample_data(
        np.asarray([0, 1], dtype=np.int64),
        ["0", "0"],
        ["C", "A"],
        str(phenotype_path),
        ["trait_a", "trait_b"],
    )

    prediction_source = _core.MultiRegeniePredictionSource.from_native_multi_aligned_sample_data(
        str(prediction_list_path),
        native_multi_aligned_sample_data,
    )

    np.testing.assert_allclose(
        prediction_source.get_chromosome_predictions("chr22"),
        np.asarray([[0.3, 0.1], [1.3, 1.1]], dtype=np.float32),
        atol=1e-6,
    )


def test_multi_regenie_prediction_source_reports_missing_phenotype(tmp_path: Path) -> None:
    trait_a_loco_path = tmp_path / "trait_a.loco"
    trait_a_loco_path.write_text("FID_IID 0_A\n22 0.1\n")
    prediction_list_path = tmp_path / "pred.list"
    prediction_list_path.write_text(f"trait_a {trait_a_loco_path}\n")

    with np.testing.assert_raises_regex(ValueError, "Phenotype 'trait_b' not found"):
        _core.MultiRegeniePredictionSource(
            str(prediction_list_path),
            ["trait_a", "trait_b"],
            ["0"],
            ["A"],
        )


def test_multi_regenie_prediction_source_reports_missing_chromosome(tmp_path: Path) -> None:
    trait_a_loco_path = tmp_path / "trait_a.loco"
    trait_a_loco_path.write_text("FID_IID 0_A\n22 0.1\n")
    trait_b_loco_path = tmp_path / "trait_b.loco"
    trait_b_loco_path.write_text("FID_IID 0_A\n22 1.1\n")
    prediction_list_path = tmp_path / "pred.list"
    prediction_list_path.write_text(f"trait_a {trait_a_loco_path}\ntrait_b {trait_b_loco_path}\n")
    prediction_source = _core.MultiRegeniePredictionSource(
        str(prediction_list_path),
        ["trait_a", "trait_b"],
        ["0"],
        ["A"],
    )

    with np.testing.assert_raises_regex(ValueError, "Chromosome '1'"):
        prediction_source.get_chromosome_predictions("1")


def test_regenie_prediction_source_reports_missing_samples(tmp_path: Path) -> None:
    loco_path = tmp_path / "trait.loco"
    loco_path.write_text("FID_IID 0_A\n22 0.1\n")
    prediction_list_path = tmp_path / "trait_pred.list"
    prediction_list_path.write_text(f"trait {loco_path}\n")

    with np.testing.assert_raises_regex(ValueError, "Target samples not found in LOCO file"):
        _core.RegeniePredictionSource(
            str(prediction_list_path),
            "trait",
            ["0"],
            ["missing"],
        )


def test_regenie_prediction_source_rejects_duplicate_loco_iid_by_default(tmp_path: Path) -> None:
    loco_path = tmp_path / "trait.loco"
    loco_path.write_text("FID_IID f1_s1 f2_s1\n22 0.1 0.2\n")
    prediction_list_path = tmp_path / "trait_pred.list"
    prediction_list_path.write_text(f"trait {loco_path}\n")

    with np.testing.assert_raises_regex(ValueError, "Duplicate LOCO IID 's1'"):
        _core.RegeniePredictionSource(
            str(prediction_list_path),
            "trait",
            ["f1"],
            ["s1"],
        )


def test_regenie_prediction_source_rejects_duplicate_target_iid_by_default(tmp_path: Path) -> None:
    loco_path = tmp_path / "trait.loco"
    loco_path.write_text("FID_IID f1_s1\n22 0.1\n")
    prediction_list_path = tmp_path / "trait_pred.list"
    prediction_list_path.write_text(f"trait {loco_path}\n")

    with np.testing.assert_raises_regex(ValueError, "Duplicate target IID 's1'"):
        _core.RegeniePredictionSource(
            str(prediction_list_path),
            "trait",
            ["f1", "f2"],
            ["s1", "s1"],
        )


def test_regenie_prediction_source_rejects_duplicate_exact_loco_key(tmp_path: Path) -> None:
    loco_path = tmp_path / "trait.loco"
    loco_path.write_text("FID_IID f1_s1 f1_s1\n22 0.1 0.2\n")
    prediction_list_path = tmp_path / "trait_pred.list"
    prediction_list_path.write_text(f"trait {loco_path}\n")

    with np.testing.assert_raises_regex(ValueError, "Duplicate LOCO sample key: f1_s1"):
        _core.RegeniePredictionSource(
            str(prediction_list_path),
            "trait",
            ["f1"],
            ["s1"],
        )


def test_regenie_prediction_source_fid_iid_mode_aligns_repeated_iid(tmp_path: Path) -> None:
    loco_path = tmp_path / "trait.loco"
    loco_path.write_text("FID_IID f1_s1 f2_s1\n22 0.1 0.2\n")
    prediction_list_path = tmp_path / "trait_pred.list"
    prediction_list_path.write_text(f"trait {loco_path}\n")

    prediction_source = _core.RegeniePredictionSource(
        str(prediction_list_path),
        "trait",
        ["f2", "f1"],
        ["s1", "s1"],
        sample_key_mode="fid_iid",
    )

    np.testing.assert_allclose(prediction_source.get_chromosome_predictions("22"), [0.2, 0.1], atol=1e-6)
