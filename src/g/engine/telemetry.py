"""Structured run telemetry helpers."""

from __future__ import annotations

import contextlib
import typing
from dataclasses import dataclass
from pathlib import Path

from g import _core, types

TelemetryCounterValue = bool | float | int | None
TelemetryWriterCounters = dict[str, TelemetryCounterValue]
TelemetryCloseMetadata = dict[str, TelemetryWriterCounters]

if typing.TYPE_CHECKING:
    from g.engine import run_events
    from g.interface import config


class TelemetryCloseableSession(typing.Protocol):
    """Telemetry session shape accepted by the close helper."""

    def log_event(self, event: str, level: str, **fields: object) -> None:
        """Write one telemetry event."""

    def writer_counters(self) -> object:
        """Return writer counters for close telemetry."""

    def close(self) -> object:
        """Close the telemetry resources."""


@dataclass(frozen=True)
class TelemetryPaths:
    """Resolved telemetry output paths for one run.

    Attributes:
        log_dir: Directory containing telemetry streams.
        stream_file: Unified JSONL event stream.
        profile_summary_json: Optional aggregate profile summary path.
        stage_timings_json: Optional detailed synchronized stage timings path.

    """

    log_dir: Path | None
    stream_file: Path | None
    profile_summary_json: Path | None
    stage_timings_json: Path | None


class TelemetrySession:
    """Run-scoped structured telemetry writer."""

    def __init__(
        self,
        *,
        mode: types.TelemetryMode,
        paths: TelemetryPaths,
        progress_interval_seconds: float,
        progress_interval_chunks: int,
        queue_size: int,
        lossy: bool,
        trace_event_cap: int,
        run_id: str | None,
    ) -> None:
        """Initialize a run telemetry session."""
        self.mode = mode
        self.paths = paths
        self.native_session_handle = _core.NativeTelemetryRunSession(
            telemetry_mode=mode.value,
            stream_file=None if paths.stream_file is None else str(paths.stream_file),
            progress_interval_seconds=progress_interval_seconds,
            progress_interval_chunks=progress_interval_chunks,
            queue_size=queue_size,
            lossy=lossy,
            trace_event_cap=trace_event_cap,
            run_id=run_id,
        )

    @property
    def enabled(self) -> bool:
        """Return whether this session writes telemetry."""
        return self.native_session_handle.enabled

    @property
    def profile_enabled(self) -> bool:
        """Return whether profiling-grade telemetry is enabled."""
        return self.native_session_handle.profile_enabled

    @property
    def run_id(self) -> str:
        """Return the native run identifier."""
        return self.native_session_handle.run_id

    @property
    def native_session_policy(self) -> _core.NativeTelemetryRunSession:
        """Return the native session handle for policy compatibility."""
        return self.native_session_handle

    @property
    def native_progress_throttle(self) -> _core.NativeTelemetryRunSession:
        """Return the native session handle for progress compatibility."""
        return self.native_session_handle

    @property
    def native_telemetry_session(self) -> _core.NativeTelemetryRunSession | None:
        """Return the native session handle when a writer is configured."""
        if not self.native_session_handle.has_native_telemetry_session:
            return None
        return self.native_session_handle

    @property
    def close_metadata(self) -> TelemetryCloseMetadata | None:
        """Return close metadata captured by the native telemetry handle."""
        metadata = self.native_session_handle.close_metadata()
        if metadata is None:
            return None
        return typing.cast("TelemetryCloseMetadata", dict(metadata))

    def log_event(self, event: str, level: str, **fields: object) -> None:
        """Write one structured lifecycle or profile event."""
        self.native_session_handle.emit_current_event(
            event,
            level,
            fields,
        )

    def log_run_completed(self, event: run_events.RunCompletedEvent) -> None:
        """Write the canonical run completion event."""
        self.native_session_handle.emit_run_completed_event(event)

    def log_run_interrupted(self, event: run_events.RunInterruptedEvent) -> None:
        """Write the canonical graceful-interruption event."""
        self.native_session_handle.emit_run_interrupted_event(event)

    def log_run_failed(self, event: run_events.RunFailedEvent) -> None:
        """Write the canonical run failure event."""
        self.native_session_handle.emit_run_failed_event(event)

    def log_run_started(
        self,
        *,
        association_mode: types.AssociationMode,
        trait_type: types.RegenieTraitType,
        phenotype_count: int,
        output_run_root: Path,
    ) -> None:
        """Write the canonical run start event."""
        self.native_session_handle.emit_run_started_event(
            association_mode.value,
            trait_type.value,
            phenotype_count,
            str(output_run_root),
        )

    def log_execution_plan_prepared(
        self,
        *,
        association_mode: types.AssociationMode,
        trait_type: types.RegenieTraitType,
        phenotype_count: int,
        chunk_size: int,
        variant_limit: int | None,
        device: types.Device,
    ) -> None:
        """Write the canonical execution-plan preparation event."""
        self.native_session_handle.emit_execution_plan_prepared_event(
            association_mode.value,
            trait_type.value,
            phenotype_count,
            chunk_size,
            variant_limit,
            device.value,
        )

    def log_effective_config_written(
        self,
        *,
        association_mode: types.AssociationMode,
        phenotype: str,
        effective_config: Path,
        output_run_directory: Path,
    ) -> None:
        """Write the canonical effective-config metadata event."""
        self.native_session_handle.emit_effective_config_written_event(
            association_mode.value,
            phenotype,
            str(effective_config),
            str(output_run_directory),
        )

    def log_writer_finished(
        self,
        *,
        association_mode: types.AssociationMode,
        phenotype: str,
        final_output_path: Path | None,
    ) -> None:
        """Write the canonical single-phenotype writer completion event."""
        self.native_session_handle.emit_phenotype_writer_finished_event(
            association_mode.value,
            phenotype,
            None if final_output_path is None else str(final_output_path),
        )

    def log_multi_writer_finished(
        self,
        *,
        association_mode: types.AssociationMode,
        phenotype_count: int,
        final_output_paths: tuple[Path | None, ...],
    ) -> None:
        """Write the canonical multi-phenotype writer completion event."""
        self.native_session_handle.emit_multi_phenotype_writer_finished_event(
            association_mode.value,
            phenotype_count,
            tuple(None if path is None else str(path) for path in final_output_paths),
        )

    def log_single_trait_preflight_completed(
        self,
        *,
        association_mode: types.AssociationMode,
        phenotype: str,
        sample_count: int,
        covariate_count: int,
        chromosome_count: int,
    ) -> None:
        """Write the canonical single-trait preflight completion event."""
        self.native_session_handle.emit_single_trait_preflight_completed_event(
            association_mode.value,
            phenotype,
            sample_count,
            covariate_count,
            chromosome_count,
        )

    def log_multi_phenotype_preflight_completed(
        self,
        *,
        association_mode: types.AssociationMode,
        phenotype_count: int,
        sample_count: int,
    ) -> None:
        """Write the canonical multi-phenotype preflight completion event."""
        self.native_session_handle.emit_multi_phenotype_preflight_completed_event(
            association_mode.value,
            phenotype_count,
            sample_count,
        )

    def log_sample_alignment_completed(
        self,
        *,
        association_mode: types.AssociationMode,
        phenotype: str | None,
        phenotype_count: int | None,
        sample_count: int | None,
        covariate_count: int | None,
        phenotype_group_count: int | None,
    ) -> None:
        """Write the canonical sample-alignment completion event."""
        self.native_session_handle.emit_sample_alignment_completed_event(
            association_mode.value,
            phenotype,
            phenotype_count,
            sample_count,
            covariate_count,
            phenotype_group_count,
        )

    def log_prediction_source_loaded(
        self,
        *,
        association_mode: types.AssociationMode,
        phenotype: str | None,
        phenotype_count: int | None,
    ) -> None:
        """Write the canonical prediction-source loading event."""
        self.native_session_handle.emit_prediction_source_loaded_event(
            association_mode.value,
            phenotype,
            phenotype_count,
        )

    def log_multi_phenotype_sample_summary(
        self,
        *,
        association_mode: types.AssociationMode,
        sample_mode: types.MultiPhenotypeSampleMode,
        sample_counts: tuple[int, ...],
        sample_set_fingerprints: tuple[str | None, ...],
        phenotype_group_count: int,
    ) -> None:
        """Write the canonical multi-phenotype sample summary event."""
        self.native_session_handle.emit_multi_phenotype_sample_summary_event(
            association_mode.value,
            sample_mode.value,
            sample_counts,
            sample_set_fingerprints,
            phenotype_group_count,
        )

    def log_gpu_genotype_format_resolved(
        self,
        *,
        requested_gpu_genotype_format: types.GpuGenotypeFormat,
        resolved_gpu_genotype_format: types.GpuGenotypeFormat,
        resolution_reason: str,
        fallback_error: str | None,
    ) -> None:
        """Write the canonical GPU genotype-format resolution event."""
        self.native_session_handle.emit_gpu_genotype_format_resolved_event(
            requested_gpu_genotype_format.value,
            resolved_gpu_genotype_format.value,
            resolution_reason,
            fallback_error,
        )

    def log_association_backend_selected(
        self,
        *,
        association_mode: types.AssociationMode,
        association_backend_kind: types.AssociationBackendKind,
        device: types.Device,
        genotype_format: types.GpuGenotypeFormat,
        phenotype: str | None,
        phenotype_count: int | None,
    ) -> None:
        """Write the canonical association-backend selection event."""
        self.native_session_handle.emit_association_backend_selected_event(
            association_mode.value,
            association_backend_kind.value,
            device.value,
            genotype_format.value,
            phenotype,
            phenotype_count,
        )

    def log_bgen_engine_opened(
        self,
        *,
        association_mode: types.AssociationMode,
        association_backend_kind: types.AssociationBackendKind,
        sample_count: int,
        variant_count: int,
        phenotype: str | None,
        phenotype_count: int | None,
    ) -> None:
        """Write the canonical BGEN engine-opened event."""
        self.native_session_handle.emit_bgen_engine_opened_event(
            association_mode.value,
            association_backend_kind.value,
            sample_count,
            variant_count,
            phenotype,
            phenotype_count,
        )

    def log_progress(self, *, processed_chunk_count: int, **fields: object) -> None:
        """Write throttled progress telemetry."""
        self.native_session_handle.emit_progress(
            processed_chunk_count,
            fields,
        )

    def should_emit_progress(self, processed_chunk_count: int) -> bool:
        """Return whether a progress event should be emitted now."""
        return self.native_session_handle.should_emit_progress(processed_chunk_count)

    def build_event_payload(self, *, event: str, level: str, **fields: object) -> dict[str, object]:
        """Build a schema-versioned telemetry event payload."""
        return dict(self.native_session_handle.build_current_event_payload(event, level, fields))

    def write_json_line(self, payload: dict[str, object]) -> None:
        """Append one JSON line when the destination path is configured."""
        self.native_session_handle.emit_payload(payload)

    def writer_counters(self) -> TelemetryWriterCounters:
        """Return the current native telemetry writer counters."""
        return typing.cast("TelemetryWriterCounters", dict(self.native_session_handle.counters()))

    def close(self) -> TelemetryCloseMetadata | None:
        """Flush buffered telemetry resources."""
        metadata = self.native_session_handle.finish_close_metadata()
        if metadata is None:
            return None
        return typing.cast("TelemetryCloseMetadata", dict(metadata))

    def close_with_event(self) -> TelemetryCloseMetadata | None:
        """Emit the close event and flush buffered telemetry resources."""
        metadata = self.native_session_handle.finish_with_current_close_event_metadata()
        if metadata is None:
            return None
        return typing.cast("TelemetryCloseMetadata", dict(metadata))


def format_timestamp(timestamp_seconds: float) -> str:
    """Format a Unix timestamp as an RFC 3339 UTC timestamp."""
    return _core.format_telemetry_timestamp_value(timestamp_seconds)


def resolve_output_run_root(regenie_config: config.RegenieConfig) -> Path:
    """Resolve the shared output run root for telemetry defaults."""
    output_prefix = typing.cast("Path", regenie_config.g_output.out)
    output_run_directory = regenie_config.g_output.output_run_directory
    return Path(
        _core.resolve_telemetry_output_run_root_value(
            str(output_prefix),
            None if output_run_directory is None else str(output_run_directory),
        )
    )


def resolve_telemetry_paths(regenie_config: config.RegenieConfig) -> TelemetryPaths:
    """Resolve diagnostics paths using documented log_dir defaults."""
    diagnostics_config = regenie_config.g_diagnostics
    output_prefix = typing.cast("Path", regenie_config.g_output.out)
    output_run_directory = regenie_config.g_output.output_run_directory
    return telemetry_paths_from_native_payload(
        _core.resolve_telemetry_paths_payload(
            str(output_prefix),
            None if output_run_directory is None else str(output_run_directory),
            diagnostics_config.telemetry.value,
            None if diagnostics_config.log_dir is None else str(diagnostics_config.log_dir),
            None if diagnostics_config.log_file is None else str(diagnostics_config.log_file),
            None if diagnostics_config.trace_file is None else str(diagnostics_config.trace_file),
            None if diagnostics_config.profile_summary_json is None else str(diagnostics_config.profile_summary_json),
            None if diagnostics_config.stage_timings_json is None else str(diagnostics_config.stage_timings_json),
        )
    )


def telemetry_paths_from_native_payload(payload: object) -> TelemetryPaths:
    """Adapt a native telemetry path payload to the public Python dataclass."""
    telemetry_paths_payload = native_mapping_payload(payload)
    return TelemetryPaths(
        log_dir=optional_path_from_native_payload(telemetry_paths_payload["log_dir"]),
        stream_file=optional_path_from_native_payload(telemetry_paths_payload["stream_file"]),
        profile_summary_json=optional_path_from_native_payload(telemetry_paths_payload["profile_summary_json"]),
        stage_timings_json=optional_path_from_native_payload(telemetry_paths_payload["stage_timings_json"]),
    )


def optional_path_from_native_payload(path_payload: object) -> Path | None:
    """Adapt an optional native path string to a Python path."""
    if path_payload is None:
        return None
    return Path(typing.cast("str", path_payload))


def native_mapping_payload(payload: object) -> dict[str, typing.Any]:
    """Adapt a native mapping payload to a mutable Python dictionary."""
    return dict(typing.cast("typing.Mapping[str, typing.Any]", payload))


def resolve_telemetry_stream_file(
    *,
    telemetry_mode: types.TelemetryMode,
    log_dir: Path | None,
    log_file: Path | None,
    trace_file: Path | None,
) -> Path | None:
    """Resolve the unified telemetry stream file."""
    stream_file = _core.resolve_telemetry_stream_file_value(
        telemetry_mode.value,
        None if log_dir is None else str(log_dir),
        None if log_file is None else str(log_file),
        None if trace_file is None else str(trace_file),
    )
    return None if stream_file is None else Path(stream_file)


def paths_refer_to_same_file(first_path: Path, second_path: Path) -> bool:
    """Return whether two paths resolve to the same filesystem target."""
    return _core.paths_refer_to_same_file_value(str(first_path), str(second_path))


def build_telemetry_session(regenie_config: config.RegenieConfig) -> TelemetrySession:
    """Build a telemetry session from normalized diagnostics config."""
    diagnostics_config = regenie_config.g_diagnostics
    return TelemetrySession(
        mode=diagnostics_config.telemetry,
        paths=resolve_telemetry_paths(regenie_config),
        progress_interval_seconds=diagnostics_config.progress_interval_seconds,
        progress_interval_chunks=diagnostics_config.progress_interval_chunks,
        queue_size=diagnostics_config.log_queue_size,
        lossy=diagnostics_config.log_lossy,
        trace_event_cap=diagnostics_config.trace_event_cap,
        run_id=None,
    )


def build_empty_writer_counters() -> TelemetryWriterCounters:
    """Return a zeroed telemetry writer counter snapshot."""
    return typing.cast(
        "TelemetryWriterCounters",
        native_mapping_payload(_core.build_empty_telemetry_writer_counters_payload()),
    )


def close_telemetry_session(telemetry_session: TelemetryCloseableSession | None) -> None:
    """Flush telemetry teardown hooks and preserve close failures."""
    close_plan = _core.plan_telemetry_close(
        has_telemetry_session=telemetry_session is not None,
        is_native_telemetry_session=isinstance(telemetry_session, TelemetrySession),
    )
    if not close_plan.should_close:
        return
    active_telemetry_session = typing.cast("TelemetryCloseableSession", telemetry_session)
    if close_plan.use_native_close_with_event:
        native_telemetry_session = typing.cast("TelemetrySession", active_telemetry_session)
        native_telemetry_session.close_with_event()
        return
    if close_plan.should_emit_legacy_close_event:
        with contextlib.suppress(Exception):
            active_telemetry_session.log_event(
                close_plan.legacy_close_event_name,
                level=close_plan.legacy_close_event_level,
                writer_counters=active_telemetry_session.writer_counters(),
            )
    active_telemetry_session.close()
