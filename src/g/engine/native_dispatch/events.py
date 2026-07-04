"""Native-dispatch diagnostic event helpers."""

from __future__ import annotations

import typing

from g import _core

if typing.TYPE_CHECKING:
    from pathlib import Path

    from g import types
    from g.runner import lifecycle


def native_dispatch_diagnostic_policy() -> _core.NativeDispatchDiagnosticPolicy:
    """Build the native-dispatch diagnostic policy handle."""
    return _core.NativeDispatchDiagnosticPolicy()


def record_bgen_engine_constructing(
    *,
    chunk_size: int,
    source_path: Path,
    trusted_no_missing_diploid: bool,
    variant_limit: int | None,
) -> None:
    """Record native BGEN engine construction."""
    native_dispatch_diagnostic_policy().record_native_dispatch_bgen_engine_constructing_diagnostic_event(
        chunk_size=chunk_size,
        source_path=str(source_path),
        trusted_no_missing_diploid=trusted_no_missing_diploid,
        variant_limit=variant_limit,
    )


def record_trusted_bgen_validation_started(
    *,
    source_path: Path,
    trusted_bgen_validation_mode: types.TrustedBgenValidationMode,
) -> None:
    """Record trusted BGEN validation start."""
    native_dispatch_diagnostic_policy().record_native_dispatch_trusted_bgen_validation_started_diagnostic_event(
        source_path=str(source_path),
        trusted_bgen_validation_mode=trusted_bgen_validation_mode.value,
    )


def record_callback_drain_started() -> None:
    """Record callback drain start."""
    native_dispatch_diagnostic_policy().record_native_dispatch_callback_drain_started_diagnostic_event()


def record_writer_session_finish_started() -> None:
    """Record single writer-session finish start."""
    native_dispatch_diagnostic_policy().record_native_dispatch_writer_session_finish_started_diagnostic_event()


def record_writer_sessions_finish_started(
    *,
    requested_thread_count: int,
    writer_session_count: int,
) -> None:
    """Record multi-writer finish start."""
    native_dispatch_diagnostic_policy().record_native_dispatch_writer_sessions_finish_started_diagnostic_event(
        requested_thread_count=requested_thread_count,
        writer_session_count=writer_session_count,
    )


def record_writer_session_interrupted_flush_started(
    shutdown_request: lifecycle.GracefulShutdownRequested,
) -> None:
    """Record interrupted single-writer flush start."""
    native_dispatch_diagnostic_policy().record_native_dispatch_writer_session_interrupted_flush_started_diagnostic_event(
        signal_exit_code=shutdown_request.exit_code,
        signal_name=shutdown_request.signal_name,
        signal_number=shutdown_request.shutdown_signal.number,
    )


def record_writer_sessions_interrupted_flush_started(
    shutdown_request: lifecycle.GracefulShutdownRequested,
    *,
    requested_thread_count: int,
    writer_session_count: int,
) -> None:
    """Record interrupted multi-writer flush start."""
    native_dispatch_diagnostic_policy().record_native_dispatch_writer_sessions_interrupted_flush_started_diagnostic_event(
        requested_thread_count=requested_thread_count,
        signal_exit_code=shutdown_request.exit_code,
        signal_name=shutdown_request.signal_name,
        signal_number=shutdown_request.shutdown_signal.number,
        writer_session_count=writer_session_count,
    )


def record_delivery_started(
    *,
    committed_chunk_count: int,
    pipeline_label: str,
    variant_major_packed8_probability_pairs: bool,
) -> None:
    """Record native BGEN delivery start."""
    native_dispatch_diagnostic_policy().record_native_dispatch_delivery_started_diagnostic_event(
        committed_chunk_count=committed_chunk_count,
        pipeline_label=pipeline_label,
        variant_major_packed8_probability_pairs=variant_major_packed8_probability_pairs,
    )


def record_delivery_finished(
    *,
    pipeline_label: str,
    processed_chunk_count: int,
) -> None:
    """Record native BGEN delivery completion."""
    native_dispatch_diagnostic_policy().record_native_dispatch_delivery_finished_diagnostic_event(
        pipeline_label=pipeline_label,
        processed_chunk_count=processed_chunk_count,
    )


def record_delivery_interrupted(
    *,
    pipeline_label: str,
    shutdown_request: lifecycle.GracefulShutdownRequested,
) -> None:
    """Record native BGEN delivery interruption."""
    native_dispatch_diagnostic_policy().record_native_dispatch_delivery_interrupted_diagnostic_event(
        pipeline_label=pipeline_label,
        signal_exit_code=shutdown_request.exit_code,
        signal_name=shutdown_request.signal_name,
        signal_number=shutdown_request.shutdown_signal.number,
    )


def record_delivery_failed(
    *,
    exception: BaseException,
    pipeline_label: str,
) -> None:
    """Record native BGEN delivery failure."""
    native_dispatch_diagnostic_policy().record_native_dispatch_delivery_failed_diagnostic_event(
        exception_message=str(exception),
        exception_type=type(exception).__name__,
        pipeline_label=pipeline_label,
    )


def record_pipeline_finished(
    *,
    final_parquet_path_count: int,
    pipeline_label: str,
) -> None:
    """Record native-dispatch pipeline completion."""
    native_dispatch_diagnostic_policy().record_native_dispatch_pipeline_finished_diagnostic_event(
        final_parquet_path_count=final_parquet_path_count,
        pipeline_label=pipeline_label,
    )
