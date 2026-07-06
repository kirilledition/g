"""Runner-local run-event and telemetry helpers."""

from __future__ import annotations

import typing
from dataclasses import dataclass
from pathlib import Path

from g import _core, types


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
    def native_telemetry_session(self) -> _core.NativeTelemetryRunSession | None:
        """Return the native session handle when a writer is configured."""
        if not self.native_session_handle.has_native_telemetry_session:
            return None
        return self.native_session_handle


def native_telemetry_session_handle(
    telemetry_session: TelemetrySession | None,
) -> _core.NativeTelemetryRunSession | None:
    """Return the active native telemetry session handle, if telemetry writes are enabled."""
    if telemetry_session is None:
        return None
    return telemetry_session.native_telemetry_session


def record_phenotype_writer_finished_telemetry(
    telemetry_session: TelemetrySession | None,
    association_mode: str,
    phenotype: str,
    final_output_path: str | None,
) -> None:
    """Record single-phenotype writer completion telemetry."""
    native_telemetry_session = native_telemetry_session_handle(telemetry_session)
    if native_telemetry_session is None:
        return
    native_telemetry_session.emit_phenotype_writer_finished_event(
        association_mode,
        phenotype,
        final_output_path,
    )


def record_multi_phenotype_writer_finished_telemetry(
    telemetry_session: TelemetrySession | None,
    association_mode: str,
    phenotype_count: int,
    final_output_paths: tuple[str | None, ...],
) -> None:
    """Record multi-phenotype writer completion telemetry."""
    native_telemetry_session = native_telemetry_session_handle(telemetry_session)
    if native_telemetry_session is None:
        return
    native_telemetry_session.emit_multi_phenotype_writer_finished_event(
        association_mode,
        phenotype_count,
        final_output_paths,
    )


def record_sample_alignment_completed_telemetry(
    telemetry_session: TelemetrySession | None,
    association_mode: str,
    phenotype: str | None,
    phenotype_count: int | None,
    sample_count: int | None,
    covariate_count: int | None,
    phenotype_group_count: int | None,
) -> None:
    """Record sample-alignment completion telemetry."""
    native_telemetry_session = native_telemetry_session_handle(telemetry_session)
    if native_telemetry_session is None:
        return
    native_telemetry_session.emit_sample_alignment_completed_event(
        association_mode,
        phenotype,
        phenotype_count,
        sample_count,
        covariate_count,
        phenotype_group_count,
    )


def record_prediction_source_loaded_telemetry(
    telemetry_session: TelemetrySession | None,
    association_mode: str,
    phenotype: str | None,
    phenotype_count: int | None,
) -> None:
    """Record prediction source load telemetry."""
    native_telemetry_session = native_telemetry_session_handle(telemetry_session)
    if native_telemetry_session is None:
        return
    native_telemetry_session.emit_prediction_source_loaded_event(
        association_mode,
        phenotype,
        phenotype_count,
    )


def record_single_trait_preflight_completed_telemetry(
    telemetry_session: TelemetrySession | None,
    association_mode: str,
    phenotype: str,
    sample_count: int,
    covariate_count: int,
    chromosome_count: int,
) -> None:
    """Record single-trait preflight completion telemetry."""
    native_telemetry_session = native_telemetry_session_handle(telemetry_session)
    if native_telemetry_session is None:
        return
    native_telemetry_session.emit_single_trait_preflight_completed_event(
        association_mode,
        phenotype,
        sample_count,
        covariate_count,
        chromosome_count,
    )


def record_multi_phenotype_preflight_completed_telemetry(
    telemetry_session: TelemetrySession | None,
    association_mode: str,
    phenotype_count: int,
    sample_count: int,
) -> None:
    """Record multi-phenotype preflight completion telemetry."""
    native_telemetry_session = native_telemetry_session_handle(telemetry_session)
    if native_telemetry_session is None:
        return
    native_telemetry_session.emit_multi_phenotype_preflight_completed_event(
        association_mode,
        phenotype_count,
        sample_count,
    )


def record_multi_phenotype_sample_summary_telemetry(
    telemetry_session: TelemetrySession | None,
    association_mode: str,
    multi_phenotype_sample_mode: str,
    sample_counts: tuple[int, ...],
    sample_set_fingerprints: tuple[str | None, ...],
    phenotype_group_count: int,
) -> None:
    """Record multi-phenotype sample summary telemetry."""
    native_telemetry_session = native_telemetry_session_handle(telemetry_session)
    if native_telemetry_session is None:
        return
    native_telemetry_session.emit_multi_phenotype_sample_summary_event(
        association_mode,
        multi_phenotype_sample_mode,
        sample_counts,
        sample_set_fingerprints,
        phenotype_group_count,
    )


def record_association_backend_selected_telemetry(
    telemetry_session: TelemetrySession | None,
    association_mode: str,
    association_backend_kind: str,
    device: str,
    genotype_format: str,
    phenotype: str | None,
    phenotype_count: int | None,
) -> None:
    """Record association backend selection telemetry."""
    native_telemetry_session = native_telemetry_session_handle(telemetry_session)
    if native_telemetry_session is None:
        return
    native_telemetry_session.emit_association_backend_selected_event(
        association_mode,
        association_backend_kind,
        device,
        genotype_format,
        phenotype,
        phenotype_count,
    )


def record_bgen_engine_opened_telemetry(
    telemetry_session: TelemetrySession | None,
    association_mode: str,
    association_backend_kind: str,
    sample_count: int,
    variant_count: int,
    phenotype: str | None,
    phenotype_count: int | None,
) -> None:
    """Record BGEN engine open telemetry."""
    native_telemetry_session = native_telemetry_session_handle(telemetry_session)
    if native_telemetry_session is None:
        return
    native_telemetry_session.emit_bgen_engine_opened_event(
        association_mode,
        association_backend_kind,
        sample_count,
        variant_count,
        phenotype,
        phenotype_count,
    )


def record_callback_progress_update_telemetry(
    telemetry_session: TelemetrySession | None,
    progress_update: _core.NativeCallbackProgressUpdate,
) -> None:
    """Record callback progress telemetry update."""
    native_telemetry_session = native_telemetry_session_handle(telemetry_session)
    if native_telemetry_session is None:
        return
    telemetry_plan = progress_update.telemetry_plan
    for progress_event in telemetry_plan.events:
        native_telemetry_session.emit_callback_progress_event(progress_event)
    progress_record = telemetry_plan.progress
    native_telemetry_session.emit_progress(
        progress_record.processed_chunk_count,
        {
            "chromosome": progress_record.chromosome,
            "chunk_identifier": progress_record.chunk_identifier,
            "variant_start_index": progress_record.variant_start_index,
            "variant_stop_index": progress_record.variant_stop_index,
            "variant_count": progress_record.variant_count,
        },
    )


def record_callback_progress_event_telemetry(
    telemetry_session: TelemetrySession | None,
    progress_event: _core.NativeCallbackProgressTelemetryEvent | None,
) -> None:
    """Record callback progress event telemetry."""
    if progress_event is None:
        return
    native_telemetry_session = native_telemetry_session_handle(telemetry_session)
    if native_telemetry_session is None:
        return
    native_telemetry_session.emit_callback_progress_event(progress_event)


def record_binary_correction_summary_telemetry(
    telemetry_session: TelemetrySession | None,
    summary_payload: dict[str, int] | None,
) -> None:
    """Record binary-correction summary telemetry."""
    if summary_payload is None:
        return
    native_telemetry_session = native_telemetry_session_handle(telemetry_session)
    if native_telemetry_session is None:
        return
    native_telemetry_session.emit_binary_correction_summary_event(summary_payload)


@dataclass(frozen=True)
class RunArtifacts:
    """Immutable pointers to generated output files.

    Attributes:
        output_run_directory: Chunked output run directory.
        final_dataset: Parquet dataset directory for part-based output.
        final_parquet: Finalized Parquet output path.
        final_regenie: Finalized REGENIE-compatible text output path.
        effective_config: Written effective TOML config path.
        phenotype_artifacts: Per-phenotype artifacts for multi-phenotype runs.
        phenotype_name: Phenotype column represented by this artifact set.
        association_mode: Statistical association engine used by the run.
        phenotype_count: Number of phenotypes included in the run.
        run_id: Diagnostics run identifier when telemetry created one.

    """

    output_run_directory: Path | None
    final_dataset: Path | None
    final_parquet: Path | None
    final_regenie: Path | None
    effective_config: Path | None
    phenotype_artifacts: tuple[RunArtifacts, ...]
    phenotype_name: str | None
    association_mode: types.AssociationMode | None
    phenotype_count: int | None
    run_id: str | None


def build_telemetry_session(regenie_config: _core.RegenieConfig) -> TelemetrySession:
    """Build the run telemetry session for one runner invocation."""
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


def resolve_telemetry_paths(regenie_config: _core.RegenieConfig) -> TelemetryPaths:
    """Resolve diagnostics paths using documented log-dir defaults."""
    diagnostics_config = regenie_config.g_diagnostics
    output_prefix = typing.cast("Path", regenie_config.g_output.out)
    output_run_directory = regenie_config.g_output.output_run_directory
    native_paths = _core.resolve_telemetry_paths(
        str(output_prefix),
        None if output_run_directory is None else str(output_run_directory),
        diagnostics_config.telemetry.value,
        None if diagnostics_config.log_dir is None else str(diagnostics_config.log_dir),
        None if diagnostics_config.log_file is None else str(diagnostics_config.log_file),
        None if diagnostics_config.trace_file is None else str(diagnostics_config.trace_file),
        None if diagnostics_config.profile_summary_json is None else str(diagnostics_config.profile_summary_json),
        None if diagnostics_config.stage_timings_json is None else str(diagnostics_config.stage_timings_json),
    )
    return TelemetryPaths(
        log_dir=optional_path_from_native_value(native_paths.log_dir),
        stream_file=optional_path_from_native_value(native_paths.stream_file),
        profile_summary_json=optional_path_from_native_value(native_paths.profile_summary_json),
        stage_timings_json=optional_path_from_native_value(native_paths.stage_timings_json),
    )


def run_artifacts_from_native_artifacts(native_artifacts: _core.NativeRunArtifacts) -> RunArtifacts:
    """Adapt a native artifact tree to the public Python dataclass."""
    association_mode_payload = native_artifacts.association_mode
    return RunArtifacts(
        output_run_directory=optional_path_from_native_value(native_artifacts.output_run_directory),
        final_dataset=optional_path_from_native_value(native_artifacts.final_dataset),
        final_parquet=optional_path_from_native_value(native_artifacts.final_parquet),
        final_regenie=optional_path_from_native_value(native_artifacts.final_regenie),
        effective_config=optional_path_from_native_value(native_artifacts.effective_config),
        phenotype_artifacts=tuple(
            run_artifacts_from_native_artifacts(phenotype_artifact)
            for phenotype_artifact in native_artifacts.phenotype_artifacts
        ),
        phenotype_name=native_artifacts.phenotype_name,
        association_mode=None if association_mode_payload is None else types.AssociationMode(association_mode_payload),
        phenotype_count=native_artifacts.phenotype_count,
        run_id=native_artifacts.run_id,
    )


def optional_path_from_native_value(path_payload: object) -> Path | None:
    """Adapt an optional native path string."""
    if path_payload is None:
        return None
    return Path(typing.cast("str", path_payload))
