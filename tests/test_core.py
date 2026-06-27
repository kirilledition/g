from __future__ import annotations

import json
import subprocess
import sys
import typing
from pathlib import Path

import numpy as np
import pytest

from g import _core

TEST_DATA_DIRECTORY = Path(__file__).parent / "data" / "bgen"
HAPLOTYPES_BGEN_PATH = TEST_DATA_DIRECTORY / "haplotypes.bgen"


def run_logging_subprocess(script: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
    )


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


def test_native_null_logistic_nonconvergence_policy() -> None:
    continue_plan = _core.plan_null_logistic_nonconvergence(
        chromosome="22",
        convergence_flags=(True,),
        scalar_convergence=True,
        phenotype_names=None,
        policy="fail",
    )
    assert continue_plan.action == "continue"
    assert continue_plan.failed_trait_indices == []
    assert continue_plan.message is None
    assert continue_plan.warning_message is None

    fail_plan = _core.plan_null_logistic_nonconvergence(
        chromosome="22",
        convergence_flags=(False,),
        scalar_convergence=True,
        phenotype_names=None,
        policy="fail",
    )
    assert fail_plan.action == "fail"
    assert fail_plan.failed_trait_indices == [0]
    assert fail_plan.message == "Binary null logistic model did not converge for chromosome 22."
    assert fail_plan.warning_message is None

    warn_plan = _core.plan_null_logistic_nonconvergence(
        chromosome="22",
        convergence_flags=(True, False),
        scalar_convergence=False,
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

    with pytest.raises(ValueError, match="Unsupported null logistic nonconvergence policy"):
        _core.plan_null_logistic_nonconvergence(
            chromosome="22",
            convergence_flags=(False,),
            scalar_convergence=True,
            phenotype_names=None,
            policy="ignore",
        )


def test_native_jax_runtime_setup_diagnostic_payloads() -> None:
    diagnostic_payloads = _core.build_jax_runtime_setup_diagnostic_payloads(
        requested_device="gpu",
        platform_name="cuda",
        cache_directory="/tmp/g-cache",
        matmul_precision="float32",
        persistent_cache_enabled=True,
        persistent_cache_min_entry_size_bytes=1024,
        persistent_cache_min_compile_time_seconds=5,
        xla_auxiliary_cache_mode="xla_gpu_per_fusion_autotune_cache_dir",
        xla_auxiliary_cache_reason="XLA auxiliary cache was requested",
        transfer_guard_enabled=True,
        gpu_validation_status="failed",
        gpu_validation_message="no gpu",
    )

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

    second_update = progress_state.record_processed_chunk(_core.build_callback_chunk_identity("chr2", 8, 10))
    assert second_update.processed_chunk_count == 2
    assert second_update.completed_chromosome == "chr1"
    assert second_update.completed_processed_chunk_count == 1
    assert second_update.started_chromosome == "chr2"
    assert progress_state.current_progress_chromosome == "chr2"

    progress_completion = progress_state.finish_progress()
    assert progress_completion is not None
    assert progress_completion.chromosome == "chr2"
    assert progress_completion.processed_chunk_count == 2
    assert progress_state.finish_progress() is None

    progress_state.record_processed_chunk_without_progress()
    assert progress_state.processed_chunk_count == 3
    assert progress_state.current_progress_chromosome is None
    assert progress_state.finish_progress() is None


def test_resolve_native_callback_worker_shutdown_timeouts_returns_native_defaults() -> None:
    worker_shutdown_timeouts = _core.resolve_native_callback_worker_shutdown_timeouts()

    assert worker_shutdown_timeouts.dosage_worker_join_timeout_seconds == 60.0
    assert worker_shutdown_timeouts.result_worker_join_timeout_seconds == 60.0
    assert worker_shutdown_timeouts.graceful_dosage_worker_join_timeout_seconds == 300.0
    assert worker_shutdown_timeouts.graceful_result_worker_join_timeout_seconds == 300.0
    assert worker_shutdown_timeouts.worker_abort_stop_timeout_seconds == 1.0


def test_resolve_callback_worker_backpressure_poll_timeout_seconds_returns_native_default() -> None:
    assert _core.resolve_callback_worker_backpressure_poll_timeout_seconds() == 0.1


def test_resolve_callback_worker_stop_poll_timeout_seconds_caps_deadline_remaining_time() -> None:
    assert _core.resolve_callback_worker_stop_poll_timeout_seconds(1.0) == 0.1
    assert _core.resolve_callback_worker_stop_poll_timeout_seconds(0.05) == 0.05
    assert _core.resolve_callback_worker_stop_poll_timeout_seconds(0.0) == 0.0
    assert _core.resolve_callback_worker_stop_poll_timeout_seconds(-1.0) == 0.0


def test_should_attempt_callback_worker_stop_uses_native_lifecycle_policy() -> None:
    assert _core.should_attempt_callback_worker_stop(
        has_started=True,
        has_worker_error=False,
        is_worker_alive=True,
    )
    assert not _core.should_attempt_callback_worker_stop(
        has_started=False,
        has_worker_error=False,
        is_worker_alive=True,
    )
    assert not _core.should_attempt_callback_worker_stop(
        has_started=True,
        has_worker_error=True,
        is_worker_alive=True,
    )
    assert not _core.should_attempt_callback_worker_stop(
        has_started=True,
        has_worker_error=False,
        is_worker_alive=False,
    )


def test_plan_callback_worker_join_uses_native_timeout_policy() -> None:
    dosage_join_plan = _core.plan_dosage_callback_worker_join(
        timeout_seconds=None,
        has_started=True,
    )
    assert dosage_join_plan.should_join is True
    assert dosage_join_plan.timeout_seconds == 60.0

    result_join_plan = _core.plan_result_callback_worker_join(
        timeout_seconds=0.25,
        has_started=True,
    )
    assert result_join_plan.should_join is True
    assert result_join_plan.timeout_seconds == 0.25

    unstarted_join_plan = _core.plan_result_callback_worker_join(
        timeout_seconds=None,
        has_started=False,
    )
    assert unstarted_join_plan.should_join is False
    assert unstarted_join_plan.timeout_seconds == 60.0


def test_plan_callback_worker_stop_uses_native_timeout_policy() -> None:
    dosage_stop_plan = _core.plan_dosage_callback_worker_stop(
        timeout_seconds=None,
        has_started=True,
        has_worker_error=False,
        is_worker_alive=True,
    )
    assert dosage_stop_plan.should_stop is True
    assert dosage_stop_plan.timeout_seconds == 60.0

    result_stop_plan = _core.plan_result_callback_worker_stop(
        timeout_seconds=0.25,
        has_started=True,
        has_worker_error=False,
        is_worker_alive=True,
    )
    assert result_stop_plan.should_stop is True
    assert result_stop_plan.timeout_seconds == 0.25

    failed_worker_stop_plan = _core.plan_result_callback_worker_stop(
        timeout_seconds=None,
        has_started=True,
        has_worker_error=True,
        is_worker_alive=True,
    )
    assert failed_worker_stop_plan.should_stop is False
    assert failed_worker_stop_plan.timeout_seconds == 60.0


def test_plan_callback_worker_finish_and_abort_use_native_timeout_policy() -> None:
    finish_plan = _core.plan_callback_worker_finish()
    assert finish_plan.dosage_stop_timeout_seconds == 60.0
    assert finish_plan.dosage_join_timeout_seconds == 300.0
    assert finish_plan.result_stop_timeout_seconds == 60.0
    assert finish_plan.result_join_timeout_seconds == 300.0

    abort_plan = _core.plan_callback_worker_abort()
    assert abort_plan.dosage_stop_timeout_seconds == 1.0
    assert abort_plan.result_stop_timeout_seconds == 1.0


def test_plan_callback_worker_stop_poll_uses_native_loop_policy() -> None:
    active_poll_plan = _core.plan_callback_worker_stop_poll(
        remaining_timeout_seconds=1.0,
        has_started=True,
        has_worker_error=False,
        is_worker_alive=True,
    )
    assert active_poll_plan.should_stop is True
    assert active_poll_plan.poll_timeout_seconds == 0.1

    failed_poll_plan = _core.plan_callback_worker_stop_poll(
        remaining_timeout_seconds=0.05,
        has_started=True,
        has_worker_error=True,
        is_worker_alive=True,
    )
    assert failed_poll_plan.should_stop is False
    assert failed_poll_plan.poll_timeout_seconds == 0.05

    expired_poll_plan = _core.plan_callback_worker_stop_poll(
        remaining_timeout_seconds=-1.0,
        has_started=True,
        has_worker_error=False,
        is_worker_alive=True,
    )
    assert expired_poll_plan.should_stop is True
    assert expired_poll_plan.poll_timeout_seconds == 0.0


def test_format_callback_worker_error_messages_uses_native_policy() -> None:
    assert (
        _core.format_dosage_callback_worker_error_message("dosage failed")
        == "native pipeline callback worker failed: dosage failed"
    )
    assert (
        _core.format_result_callback_worker_error_message("writer failed")
        == "native pipeline result writer worker failed: writer failed"
    )


def test_resolve_native_callback_queue_limits_uses_native_capacity_policy() -> None:
    queue_limits = _core.resolve_native_callback_queue_limits(
        staging_depth=3,
        native_callback_batch_size=1,
        result_in_flight_limit=None,
        dosage_buffer_limit=None,
    )
    assert queue_limits.dosage_queue_depth == 3
    assert queue_limits.result_queue_depth == 3
    assert queue_limits.result_in_flight_limit == 4
    assert queue_limits.dosage_buffer_limit == 4

    explicit_queue_limits = _core.resolve_native_callback_queue_limits(
        staging_depth=3,
        native_callback_batch_size=2,
        result_in_flight_limit=7,
        dosage_buffer_limit=8,
    )
    assert explicit_queue_limits.result_in_flight_limit == 7
    assert explicit_queue_limits.dosage_buffer_limit == 8

    with pytest.raises(ValueError, match="staging_depth must be positive"):
        _core.resolve_native_callback_queue_limits(
            staging_depth=0,
            native_callback_batch_size=1,
            result_in_flight_limit=None,
            dosage_buffer_limit=None,
        )
    with pytest.raises(ValueError, match="native_callback_batch_size must be positive"):
        _core.resolve_native_callback_queue_limits(
            staging_depth=1,
            native_callback_batch_size=0,
            result_in_flight_limit=None,
            dosage_buffer_limit=None,
        )
    with pytest.raises(ValueError, match="result_in_flight_limit must be positive"):
        _core.resolve_native_callback_queue_limits(
            staging_depth=1,
            native_callback_batch_size=1,
            result_in_flight_limit=0,
            dosage_buffer_limit=None,
        )
    with pytest.raises(ValueError, match="dosage_buffer_limit must be positive"):
        _core.resolve_native_callback_queue_limits(
            staging_depth=1,
            native_callback_batch_size=1,
            result_in_flight_limit=None,
            dosage_buffer_limit=0,
        )
    with pytest.raises(ValueError, match="effective dosage_buffer_limit"):
        _core.resolve_native_callback_queue_limits(
            staging_depth=1,
            native_callback_batch_size=3,
            result_in_flight_limit=None,
            dosage_buffer_limit=2,
        )


def test_plan_callback_queue_stage_observation_uses_native_timing_policy() -> None:
    queue_observation_plan = _core.plan_callback_queue_stage_observation(
        queue_name="dosage_queue",
        operation_name="put",
        elapsed_seconds=0.25,
        blocked=False,
    )
    assert queue_observation_plan.queue_name == "dosage_queue"
    assert queue_observation_plan.operation_name == "put"
    assert queue_observation_plan.stage_name == "callback_queue_put"
    assert queue_observation_plan.blocked_seconds == 0.0

    blocked_observation_plan = _core.plan_callback_queue_stage_observation(
        queue_name="result_in_flight_slots",
        operation_name="producer_blocking",
        elapsed_seconds=0.5,
        blocked=True,
    )
    assert blocked_observation_plan.stage_name == "result_in_flight_producer_blocking"
    assert blocked_observation_plan.blocked_seconds == 0.5

    with pytest.raises(ValueError, match="Unsupported callback queue stage operation"):
        _core.plan_callback_queue_stage_observation(
            queue_name="unknown_queue",
            operation_name="put",
            elapsed_seconds=0.25,
            blocked=False,
        )


def test_plan_callback_queue_operation_observation_uses_native_timing_policy() -> None:
    pool_observation_plan = _core.plan_callback_queue_operation_observation(
        queue_name="dosage_buffer_pool",
        operation_name="reuse",
        elapsed_seconds=0.25,
        blocked=False,
    )
    assert pool_observation_plan.queue_name == "dosage_buffer_pool"
    assert pool_observation_plan.operation_name == "reuse"
    assert pool_observation_plan.blocked_seconds == 0.0

    blocked_observation_plan = _core.plan_callback_queue_operation_observation(
        queue_name="result_in_flight_slots",
        operation_name="release",
        elapsed_seconds=0.5,
        blocked=True,
    )
    assert blocked_observation_plan.queue_name == "result_in_flight_slots"
    assert blocked_observation_plan.operation_name == "release"
    assert blocked_observation_plan.blocked_seconds == 0.5

    with pytest.raises(ValueError, match="Unsupported callback queue operation"):
        _core.plan_callback_queue_operation_observation(
            queue_name="dosage_buffer_pool",
            operation_name="unknown_operation",
            elapsed_seconds=0.25,
            blocked=False,
        )


def test_plan_dosage_buffer_reuse_uses_native_shape_policy() -> None:
    exact_reuse_plan = _core.plan_dosage_buffer_reuse(
        buffered_shape=(2, 3),
        expected_shape=(2, 3),
    )
    assert exact_reuse_plan is not None
    assert exact_reuse_plan.requires_slice is False
    assert exact_reuse_plan.slice_dimensions == [2, 3]

    sliced_reuse_plan = _core.plan_dosage_buffer_reuse(
        buffered_shape=(4, 5),
        expected_shape=(2, 3),
    )
    assert sliced_reuse_plan is not None
    assert sliced_reuse_plan.requires_slice is True
    assert sliced_reuse_plan.slice_dimensions == [2, 3]

    assert _core.plan_dosage_buffer_reuse(buffered_shape=(2, 3), expected_shape=(2, 3, 1)) is None
    assert _core.plan_dosage_buffer_reuse(buffered_shape=(2, 3), expected_shape=(3, 2)) is None


def test_plan_variant_major_dosage_batch_handoff_uses_native_batch_policy() -> None:
    batch_handoff_plan = _core.plan_variant_major_dosage_batch_handoff(
        metadata_count=2,
        genotype_matrix_by_variant_count=2,
        chunk_stats_count=2,
    )
    assert batch_handoff_plan.chunk_count == 2

    with pytest.raises(ValueError, match="identical lengths"):
        _core.plan_variant_major_dosage_batch_handoff(
            metadata_count=2,
            genotype_matrix_by_variant_count=1,
            chunk_stats_count=2,
        )
    with pytest.raises(ValueError, match="at least one chunk"):
        _core.plan_variant_major_dosage_batch_handoff(
            metadata_count=0,
            genotype_matrix_by_variant_count=0,
            chunk_stats_count=0,
        )


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
        _core.resolve_bgen_delivery_method_value(
            variant_major_packed8_probability_pairs=False,
            has_native_multi_aligned_sample_data=True,
            has_native_aligned_sample_data=True,
        )
        == "dosage_native_multi_aligned_samples"
    )
    assert (
        _core.resolve_bgen_delivery_method_value(
            variant_major_packed8_probability_pairs=False,
            has_native_multi_aligned_sample_data=False,
            has_native_aligned_sample_data=True,
        )
        == "dosage_native_aligned_samples"
    )
    assert (
        _core.resolve_bgen_delivery_method_value(
            variant_major_packed8_probability_pairs=False,
            has_native_multi_aligned_sample_data=False,
            has_native_aligned_sample_data=False,
        )
        == "dosage_sample_indices"
    )
    assert (
        _core.resolve_bgen_delivery_method_value(
            variant_major_packed8_probability_pairs=True,
            has_native_multi_aligned_sample_data=True,
            has_native_aligned_sample_data=True,
        )
        == "packed8_native_multi_aligned_samples"
    )
    assert (
        _core.resolve_bgen_delivery_method_value(
            variant_major_packed8_probability_pairs=True,
            has_native_multi_aligned_sample_data=False,
            has_native_aligned_sample_data=True,
        )
        == "packed8_native_aligned_samples"
    )
    assert (
        _core.resolve_bgen_delivery_method_value(
            variant_major_packed8_probability_pairs=True,
            has_native_multi_aligned_sample_data=False,
            has_native_aligned_sample_data=False,
        )
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
