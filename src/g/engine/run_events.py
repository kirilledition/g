"""Structured REGENIE run lifecycle events and terminal rendering."""

from __future__ import annotations

import typing
from dataclasses import dataclass
from pathlib import Path

import g._core
from g import types

if typing.TYPE_CHECKING:
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
    return run_artifacts_from_native_payload(
        native_run_event_payload_policy().attach_run_metadata_payload(
            artifacts,
            run_id,
            association_mode.value,
            phenotype_count,
        )
    )


def build_run_completed_event(artifacts: RunArtifacts) -> RunCompletedEvent:
    """Build a structured completion event from run artifacts."""
    return run_completed_event_from_native_payload(
        native_run_event_payload_policy().build_run_completed_event_payload(artifacts)
    )


def build_run_interrupted_event(shutdown_request: shutdown.GracefulShutdownRequested) -> RunInterruptedEvent:
    """Build a structured interruption event from a graceful shutdown request."""
    return run_interrupted_event_from_native_payload(
        native_run_event_payload_policy().build_run_interrupted_event_payload(shutdown_request)
    )


def build_run_failed_event(error: Exception) -> RunFailedEvent:
    """Build a structured failure event from an exception."""
    return run_failed_event_from_native_payload(native_run_event_payload_policy().build_run_failed_event_payload(error))


def native_run_event_payload_policy() -> g._core.NativeRunEventPayloadPolicy:
    """Build the native run-event payload policy handle."""
    return g._core.NativeRunEventPayloadPolicy()


def native_run_event_telemetry_policy() -> g._core.NativeRunEventTelemetryPolicy:
    """Build the native run-event telemetry policy handle."""
    return g._core.NativeRunEventTelemetryPolicy()


def native_runner_diagnostic_policy() -> g._core.NativeRunnerDiagnosticPolicy:
    """Build the native runner diagnostic policy handle."""
    return g._core.NativeRunnerDiagnosticPolicy()


def native_output_preflight_diagnostic_policy() -> g._core.NativeOutputPreflightDiagnosticPolicy:
    """Build the native output/preflight diagnostic policy handle."""
    return g._core.NativeOutputPreflightDiagnosticPolicy()


def native_pipeline_diagnostic_policy() -> g._core.NativePipelineDiagnosticPolicy:
    """Build the native pipeline diagnostic policy handle."""
    return g._core.NativePipelineDiagnosticPolicy()


def native_dispatch_diagnostic_policy() -> g._core.NativeDispatchDiagnosticPolicy:
    """Build the native dispatch diagnostic policy handle."""
    return g._core.NativeDispatchDiagnosticPolicy()


def run_completed_event_from_native_payload(payload: object) -> RunCompletedEvent:
    """Adapt a native completed-run event payload to the public Python dataclass."""
    event_payload = native_mapping_payload(payload)
    association_mode_payload = typing.cast("str | None", event_payload["association_mode"])
    return RunCompletedEvent(
        run_id=typing.cast("str | None", event_payload["run_id"]),
        association_mode=None if association_mode_payload is None else types.AssociationMode(association_mode_payload),
        phenotype_count=typing.cast("int | None", event_payload["phenotype_count"]),
        artifacts=tuple(
            run_artifact_payload_from_native_payload(artifact_payload)
            for artifact_payload in typing.cast("typing.Sequence[object]", event_payload["artifacts"])
        ),
    )


def run_artifact_payload_from_native_payload(payload: object) -> RunArtifactPayload:
    """Adapt one native completed-run artifact payload."""
    artifact_payload = native_mapping_payload(payload)
    return RunArtifactPayload(
        phenotype_name=typing.cast("str | None", artifact_payload["phenotype_name"]),
        output_run_directory=optional_path_from_native_payload(artifact_payload["output_run_directory"]),
        final_dataset=optional_path_from_native_payload(artifact_payload["final_dataset"]),
        final_parquet=optional_path_from_native_payload(artifact_payload["final_parquet"]),
        final_regenie=optional_path_from_native_payload(artifact_payload["final_regenie"]),
        effective_config=optional_path_from_native_payload(artifact_payload["effective_config"]),
    )


def run_artifacts_from_native_payload(payload: object) -> RunArtifacts:
    """Adapt a native artifact tree payload to the public Python dataclass."""
    artifacts_payload = native_mapping_payload(payload)
    association_mode_payload = typing.cast("str | None", artifacts_payload["association_mode"])
    return RunArtifacts(
        output_run_directory=optional_path_from_native_payload(artifacts_payload["output_run_directory"]),
        final_dataset=optional_path_from_native_payload(artifacts_payload["final_dataset"]),
        final_parquet=optional_path_from_native_payload(artifacts_payload["final_parquet"]),
        final_regenie=optional_path_from_native_payload(artifacts_payload["final_regenie"]),
        effective_config=optional_path_from_native_payload(artifacts_payload["effective_config"]),
        phenotype_artifacts=tuple(
            run_artifacts_from_native_payload(phenotype_artifact_payload)
            for phenotype_artifact_payload in typing.cast(
                "typing.Sequence[object]",
                artifacts_payload["phenotype_artifacts"],
            )
        ),
        phenotype_name=typing.cast("str | None", artifacts_payload["phenotype_name"]),
        association_mode=None if association_mode_payload is None else types.AssociationMode(association_mode_payload),
        phenotype_count=typing.cast("int | None", artifacts_payload["phenotype_count"]),
        run_id=typing.cast("str | None", artifacts_payload["run_id"]),
    )


def run_interrupted_event_from_native_payload(payload: object) -> RunInterruptedEvent:
    """Adapt a native interrupted-run event payload."""
    event_payload = native_mapping_payload(payload)
    return RunInterruptedEvent(
        signal_number=typing.cast("int", event_payload["signal_number"]),
        signal_name=typing.cast("str", event_payload["signal_name"]),
        exit_code=typing.cast("int", event_payload["exit_code"]),
        flushed_for_resume=typing.cast("bool", event_payload["flushed_for_resume"]),
    )


def run_failed_event_from_native_payload(payload: object) -> RunFailedEvent:
    """Adapt a native failed-run event payload."""
    event_payload = native_mapping_payload(payload)
    return RunFailedEvent(
        error_type=typing.cast("str", event_payload["error_type"]),
        error_message=typing.cast("str", event_payload["error_message"]),
    )


def optional_path_from_native_payload(path_payload: object) -> Path | None:
    """Adapt an optional native path string."""
    if path_payload is None:
        return None
    return Path(typing.cast("str", path_payload))


def native_mapping_payload(payload: object) -> dict[str, typing.Any]:
    """Adapt a native mapping payload to a mutable Python dictionary."""
    return dict(typing.cast("typing.Mapping[str, typing.Any]", payload))


def render_run_completed_lines(event: RunCompletedEvent) -> tuple[str, ...]:
    """Render concise terminal lines for a completed run."""
    return tuple(native_run_event_payload_policy().render_run_completed_lines(event))


def render_run_interrupted_lines(event: RunInterruptedEvent) -> tuple[str, ...]:
    """Render concise terminal lines for a gracefully interrupted run."""
    return tuple(native_run_event_payload_policy().render_run_interrupted_lines(event))


def render_run_failed_lines(event: RunFailedEvent) -> tuple[str, ...]:
    """Render concise terminal lines for a failed run."""
    return tuple(native_run_event_payload_policy().render_run_failed_lines(event))
