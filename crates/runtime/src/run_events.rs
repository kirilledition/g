//! Runtime-owned run lifecycle event payloads and rendering policy.

mod diagnostics;
mod lifecycle;
mod names;
mod native_cli_diagnostics;
mod native_dispatch_diagnostics;
mod runner_diagnostics;
mod runtime_diagnostics;
mod telemetry;

pub use diagnostics::{
    RunDiagnosticEventPayload, RunDiagnosticFieldPayload, RunDiagnosticFieldValue, emit_diagnostic_event,
    emit_run_diagnostic_event,
};
pub(crate) use lifecycle::build_run_failed_telemetry_fields;
pub use lifecycle::{
    PhenotypeRunArtifacts, RunFailedEventPayload, RunInterruptedEventPayload, build_run_failed_event_payload,
    build_run_interrupted_event_payload, render_run_completed_lines, render_run_failed_lines,
    render_run_interrupted_lines,
};
pub use native_cli_diagnostics::{
    build_native_cli_completed_line_diagnostic_payload, build_native_cli_failed_line_diagnostic_payload,
    build_native_cli_interrupted_line_diagnostic_payload,
};
pub use native_dispatch_diagnostics::build_native_dispatch_delivery_finished_diagnostic_payload;
pub use runner_diagnostics::{
    build_runner_execution_plan_build_started_diagnostic_payload,
    build_runner_execution_plan_dispatch_started_diagnostic_payload,
    build_runner_execution_plan_prepared_diagnostic_payload,
    build_runner_metadata_artifacts_completed_diagnostic_payload,
};
pub use runtime_diagnostics::build_native_runtime_knobs_configured_diagnostic_payload;
pub(crate) use telemetry::{
    RunTelemetryEventKind, build_association_backend_selected_telemetry_fields,
    build_execution_plan_prepared_telemetry_fields, build_multi_phenotype_writer_finished_telemetry_fields,
    build_phenotype_writer_finished_telemetry_fields,
};
