#![warn(clippy::pedantic)]

pub mod jax_runtime;
pub mod run_events;
pub mod run_metadata;
pub mod runtime_policy;
pub mod runtime_state;
pub mod shutdown;
pub mod telemetry_policy;
pub mod telemetry_session;
pub mod timing;

pub use jax_runtime::{
    JaxDeviceObservation, JaxGpuValidationPlan, JaxRuntimeConfigUpdatePayload, JaxRuntimeConfigValue,
    JaxRuntimeDiagnosticEventPayload, JaxRuntimeDiagnosticFieldPayload, JaxRuntimeDiagnosticValue,
    JaxRuntimeSetupPayload, build_jax_runtime_setup_diagnostic_events, plan_jax_gpu_validation,
    plan_jax_runtime_config_updates, resolve_jax_runtime_setup,
};
pub use run_events::{
    RunArtifactPayload, RunArtifactTelemetryFields, RunCompletedEventPayload, RunCompletedTelemetryFields,
    RunFailedEventPayload, RunFailedTelemetryFields, RunInterruptedEventPayload, RunInterruptedTelemetryFields,
    RunTelemetryStringField, build_artifact_telemetry_fields, build_run_completed_telemetry_fields,
    build_run_failed_telemetry_fields, build_run_interrupted_telemetry_fields, render_artifact_lines,
    render_run_completed_lines, render_run_failed_lines, render_run_interrupted_lines,
};
pub use run_metadata::{
    PhenotypeRunArtifactsInput, RunArtifactsPayload, RunManifestCommandPayload, RunManifestExtensionInput,
    RunManifestExtensionPayload, RunManifestRuntimePayload, build_multi_run_artifacts, build_phenotype_run_artifacts,
    build_run_manifest_extension,
};
pub use runtime_policy::{LoggingRuntimePolicyPayload, build_logging_runtime_policy, describe_logging_runtime_policy};
pub use runtime_state::{
    JaxRuntimePolicyPayload, ProcessRuntimeState, RuntimeCompatibilityError, RuntimeCompatibilityToken,
    describe_jax_runtime_policy,
};
pub use shutdown::{
    ShutdownControllerState, ShutdownRequestAction, ShutdownRequestDecisionPayload, ShutdownSignalPayload,
    build_shutdown_signal,
};
pub use telemetry_policy::{
    TelemetryPathsPayload, TelemetryWriterCountersPayload, build_empty_writer_counters, format_timestamp,
    paths_refer_to_same_file, resolve_output_run_root, resolve_telemetry_paths, resolve_telemetry_stream_file,
};
pub use telemetry_session::{
    TelemetryCapAction, TelemetryEventCapState, TelemetryEventEnvelope, TelemetryProgressThrottleState,
    TelemetryWriterCounterSnapshot, build_telemetry_event_envelope, generate_run_id,
};
pub use timing::{
    ChunkStageSummary, ChunkStageTiming, NullLogisticDiagnosticValue, NullLogisticSummary, NumericDiagnosticValue,
    ProfileSummaryPayload, QueueBackpressureAccumulator, QueueBackpressureKey, QueueBackpressureSnapshot,
    StageTimingSnapshotPayload, StageTimingState, TimingFileError, TransferMetadataAccumulator, TransferMetadataError,
    TransferMetadataKey, TransferMetadataObservation, TransferMetadataSnapshot, build_transfer_metadata_observation,
    should_collect_exact_stage_timings, write_profile_summary_payload, write_stage_timing_snapshot_payload,
};
