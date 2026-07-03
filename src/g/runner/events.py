"""Runner-local run-event and telemetry helpers."""

from __future__ import annotations

import typing

from g.engine import run_events, telemetry

if typing.TYPE_CHECKING:
    from pathlib import Path

    from g import _core, types
    from g.interface import config
    from g.runner import lifecycle

type RunArtifacts = run_events.RunArtifacts
type RunCompletedEvent = run_events.RunCompletedEvent
type RunFailedEvent = run_events.RunFailedEvent
type RunInterruptedEvent = run_events.RunInterruptedEvent
type TelemetryPaths = telemetry.TelemetryPaths
type TelemetrySession = telemetry.TelemetrySession


def build_telemetry_session(regenie_config: config.RegenieConfig) -> TelemetrySession:
    """Build the run telemetry session for one runner invocation."""
    return telemetry.build_telemetry_session(regenie_config)


def close_telemetry_session(telemetry_session: TelemetrySession | None) -> None:
    """Close a runner telemetry session through the engine telemetry adapter."""
    telemetry.close_telemetry_session(telemetry_session)


def resolve_output_run_root(regenie_config: config.RegenieConfig) -> Path:
    """Return the output root used for run-start telemetry."""
    return telemetry.resolve_output_run_root(regenie_config)


def native_run_event_telemetry_policy() -> _core.NativeRunEventTelemetryPolicy:
    """Build the native run-event telemetry policy handle."""
    return run_events.native_run_event_telemetry_policy()


def native_runner_diagnostic_policy() -> _core.NativeRunnerDiagnosticPolicy:
    """Build the native runner diagnostic policy handle."""
    return run_events.native_runner_diagnostic_policy()


def build_run_interrupted_event(shutdown_request: lifecycle.GracefulShutdownRequested) -> RunInterruptedEvent:
    """Build a structured interruption event from a graceful shutdown request."""
    return run_events.build_run_interrupted_event(shutdown_request)


def build_run_failed_event(error: Exception) -> RunFailedEvent:
    """Build a structured failure event from an exception."""
    return run_events.build_run_failed_event(error)


def attach_run_metadata(
    artifacts: RunArtifacts,
    *,
    run_id: str | None,
    association_mode: types.AssociationMode,
    phenotype_count: int,
) -> RunArtifacts:
    """Attach lifecycle metadata to returned run artifacts."""
    return run_events.attach_run_metadata(
        artifacts,
        run_id=run_id,
        association_mode=association_mode,
        phenotype_count=phenotype_count,
    )


def build_run_completed_event(artifacts: RunArtifacts) -> RunCompletedEvent:
    """Build a structured completion event from run artifacts."""
    return run_events.build_run_completed_event(artifacts)


def run_artifacts_from_native_payload(payload: object) -> RunArtifacts:
    """Adapt a native artifact tree payload to the public Python dataclass."""
    return run_events.run_artifacts_from_native_payload(payload)
