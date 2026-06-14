"""Structured REGENIE run lifecycle events and terminal rendering."""

from __future__ import annotations

import dataclasses
import typing
from dataclasses import dataclass

import g._core

if typing.TYPE_CHECKING:
    from pathlib import Path

    from g import types
    from g.engine import shutdown


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


@dataclass(frozen=True)
class RunArtifactPayload:
    """Structured payload for one phenotype's user-visible artifacts.

    Attributes:
        phenotype_name: Phenotype column represented by these artifacts.
        output_run_directory: Chunked output run directory.
        final_dataset: Parquet dataset directory for part-based output.
        final_parquet: Finalized Parquet output path.
        final_regenie: Finalized REGENIE-compatible text output path.
        effective_config: Written effective TOML config path.

    """

    phenotype_name: str | None
    output_run_directory: Path | None
    final_dataset: Path | None
    final_parquet: Path | None
    final_regenie: Path | None
    effective_config: Path | None


@dataclass(frozen=True)
class RunCompletedEvent:
    """Canonical completion event for telemetry and terminal rendering.

    Attributes:
        run_id: Diagnostics run identifier when available.
        association_mode: Statistical association engine used by the run.
        phenotype_count: Number of phenotypes included in the run.
        artifacts: User-visible artifacts produced by the run.

    """

    run_id: str | None
    association_mode: types.AssociationMode | None
    phenotype_count: int | None
    artifacts: tuple[RunArtifactPayload, ...]


@dataclass(frozen=True)
class RunInterruptedEvent:
    """Canonical graceful-interruption event.

    Attributes:
        signal_number: POSIX signal number.
        signal_name: POSIX signal name.
        exit_code: Conventional process exit code for the signal.
        flushed_for_resume: Whether committed output was flushed for resume.

    """

    signal_number: int
    signal_name: str
    exit_code: int
    flushed_for_resume: bool


@dataclass(frozen=True)
class RunFailedEvent:
    """Canonical non-graceful failure event.

    Attributes:
        error_type: Exception class name.
        error_message: Exception message.

    """

    error_type: str
    error_message: str


def attach_run_metadata(
    artifacts: RunArtifacts,
    *,
    run_id: str | None,
    association_mode: types.AssociationMode,
    phenotype_count: int,
) -> RunArtifacts:
    """Attach lifecycle metadata to returned run artifacts."""
    phenotype_artifacts = tuple(
        attach_run_metadata(
            phenotype_artifact,
            run_id=run_id,
            association_mode=association_mode,
            phenotype_count=phenotype_count,
        )
        for phenotype_artifact in artifacts.phenotype_artifacts
    )
    return dataclasses.replace(
        artifacts,
        phenotype_artifacts=phenotype_artifacts,
        run_id=run_id,
        association_mode=association_mode,
        phenotype_count=phenotype_count,
    )


def build_run_completed_event(artifacts: RunArtifacts) -> RunCompletedEvent:
    """Build a structured completion event from run artifacts."""
    artifact_payloads = flatten_artifact_payloads(artifacts)
    phenotype_count = artifacts.phenotype_count
    if phenotype_count is None and len(artifact_payloads) > 1:
        phenotype_count = len(artifact_payloads)
    return RunCompletedEvent(
        run_id=artifacts.run_id,
        association_mode=artifacts.association_mode,
        phenotype_count=phenotype_count,
        artifacts=artifact_payloads,
    )


def build_run_interrupted_event(shutdown_request: shutdown.GracefulShutdownRequested) -> RunInterruptedEvent:
    """Build a structured interruption event from a graceful shutdown request."""
    shutdown_signal = shutdown_request.shutdown_signal
    return RunInterruptedEvent(
        signal_number=shutdown_signal.number,
        signal_name=shutdown_signal.name,
        exit_code=shutdown_signal.exit_code,
        flushed_for_resume=True,
    )


def build_run_failed_event(error: Exception) -> RunFailedEvent:
    """Build a structured failure event from an exception."""
    return RunFailedEvent(error_type=type(error).__name__, error_message=str(error))


def flatten_artifact_payloads(artifacts: RunArtifacts) -> tuple[RunArtifactPayload, ...]:
    """Return per-phenotype artifact payloads for a run artifact tree."""
    if artifacts.phenotype_artifacts:
        return tuple(
            artifact_payload
            for phenotype_artifact in artifacts.phenotype_artifacts
            for artifact_payload in flatten_artifact_payloads(phenotype_artifact)
        )
    return (
        RunArtifactPayload(
            phenotype_name=artifacts.phenotype_name,
            output_run_directory=artifacts.output_run_directory,
            final_dataset=artifacts.final_dataset,
            final_parquet=artifacts.final_parquet,
            final_regenie=artifacts.final_regenie,
            effective_config=artifacts.effective_config,
        ),
    )


def run_completed_telemetry_fields(event: RunCompletedEvent) -> dict[str, object]:
    """Return JSON-serializable completion fields for telemetry."""
    native_fields_builder = getattr(g._core, "build_run_completed_telemetry_fields", None)
    if native_fields_builder is not None:
        return typing.cast("dict[str, object]", dict(native_fields_builder(event)))
    fields: dict[str, object] = {
        "artifact_count": len(event.artifacts),
        "phenotype_artifacts": tuple(artifact_payload_to_mapping(artifact) for artifact in event.artifacts),
    }
    if event.run_id is not None:
        fields["run_id"] = event.run_id
    if event.association_mode is not None:
        fields["association_mode"] = event.association_mode.value
    if event.phenotype_count is not None:
        fields["phenotype_count"] = event.phenotype_count
    if len(event.artifacts) == 1:
        fields.update(artifact_payload_to_mapping(event.artifacts[0]))
    return fields


def run_interrupted_telemetry_fields(event: RunInterruptedEvent) -> dict[str, object]:
    """Return JSON-serializable graceful-interruption fields for telemetry."""
    native_fields_builder = getattr(g._core, "build_run_interrupted_telemetry_fields", None)
    if native_fields_builder is not None:
        return typing.cast("dict[str, object]", dict(native_fields_builder(event)))
    return {
        "failure_kind": "graceful_shutdown",
        "signal_number": event.signal_number,
        "signal_name": event.signal_name,
        "exit_code": event.exit_code,
        "flushed_for_resume": event.flushed_for_resume,
    }


def run_failed_telemetry_fields(event: RunFailedEvent) -> dict[str, object]:
    """Return JSON-serializable failure fields for telemetry."""
    native_fields_builder = getattr(g._core, "build_run_failed_telemetry_fields", None)
    if native_fields_builder is not None:
        return typing.cast("dict[str, object]", dict(native_fields_builder(event)))
    return {
        "failure_kind": "exception",
        "error_type": event.error_type,
        "error_message": event.error_message,
    }


def artifact_payload_to_mapping(artifact: RunArtifactPayload) -> dict[str, str]:
    """Return a compact artifact mapping for telemetry."""
    payload: dict[str, str | None] = {
        "phenotype": artifact.phenotype_name,
        "output_run_directory": optional_path_string(artifact.output_run_directory),
        "final_dataset": optional_path_string(artifact.final_dataset),
        "final_parquet": optional_path_string(artifact.final_parquet),
        "final_regenie": optional_path_string(artifact.final_regenie),
        "effective_config": optional_path_string(artifact.effective_config),
    }
    return {key: value for key, value in payload.items() if value is not None}


def optional_path_string(path: Path | None) -> str | None:
    """Return a path as text when present."""
    return None if path is None else str(path)


def render_run_completed_lines(event: RunCompletedEvent) -> tuple[str, ...]:
    """Render concise terminal lines for a completed run."""
    native_renderer = getattr(g._core, "render_run_completed_lines", None)
    if native_renderer is not None:
        return typing.cast("tuple[str, ...]", tuple(native_renderer(event)))
    if not event.artifacts:
        return ("Success. Run completed.",)
    lines: list[str] = []
    for artifact in event.artifacts:
        lines.extend(render_artifact_lines(artifact))
    return tuple(lines) if lines else ("Success. Run completed.",)


def render_artifact_lines(artifact: RunArtifactPayload) -> tuple[str, ...]:
    """Render terminal lines for one artifact payload."""
    lines: list[str] = []
    if artifact.output_run_directory is not None:
        lines.append(f"Success. Chunked run saved to {artifact.output_run_directory}")
    else:
        lines.append("Success. Run completed.")
    if artifact.final_dataset is not None:
        lines.append(f"Parquet dataset saved to {artifact.final_dataset}")
    if artifact.final_parquet is not None:
        lines.append(f"Finalized Parquet saved to {artifact.final_parquet}")
    if artifact.final_regenie is not None:
        lines.append(f"REGENIE text output saved to {artifact.final_regenie}")
    return tuple(lines)


def render_run_interrupted_lines(event: RunInterruptedEvent) -> tuple[str, ...]:
    """Render concise terminal lines for a gracefully interrupted run."""
    native_renderer = getattr(g._core, "render_run_interrupted_lines", None)
    if native_renderer is not None:
        return typing.cast("tuple[str, ...]", tuple(native_renderer(event)))
    return (f"Interrupted by {event.signal_name}. Flushed queued chunks and saved committed output for --resume.",)


def render_run_failed_lines(event: RunFailedEvent) -> tuple[str, ...]:
    """Render concise terminal lines for a failed run."""
    native_renderer = getattr(g._core, "render_run_failed_lines", None)
    if native_renderer is not None:
        return typing.cast("tuple[str, ...]", tuple(native_renderer(event)))
    if event.error_message:
        return (f"Error: {event.error_message}",)
    return (f"Error: {event.error_type}",)
