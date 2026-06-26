#![warn(clippy::pedantic)]

pub mod run_metadata;
pub mod runtime_policy;
pub mod runtime_state;
pub mod shutdown;
pub mod telemetry_policy;
pub mod timing;

pub use run_metadata::{
    PhenotypeRunArtifactsInput, RunArtifactsPayload, RunManifestCommandPayload, RunManifestExtensionInput,
    RunManifestExtensionPayload, RunManifestRuntimePayload, build_multi_run_artifacts, build_phenotype_run_artifacts,
    build_run_manifest_extension,
};
pub use runtime_policy::{LoggingRuntimePolicyPayload, build_logging_runtime_policy, describe_logging_runtime_policy};
pub use runtime_state::{ProcessRuntimeState, RuntimeCompatibilityError};
pub use shutdown::{
    ShutdownControllerState, ShutdownRequestAction, ShutdownRequestDecisionPayload, ShutdownSignalPayload,
    build_shutdown_signal,
};
pub use telemetry_policy::{
    TelemetryPathsPayload, TelemetryWriterCountersPayload, build_empty_writer_counters, format_timestamp,
    paths_refer_to_same_file, resolve_output_run_root, resolve_telemetry_paths, resolve_telemetry_stream_file,
};
pub use timing::{
    ChunkStageSummary, ChunkStageTiming, NullLogisticDiagnosticValue, NullLogisticSummary, NumericDiagnosticValue,
    ProfileSummaryPayload, QueueBackpressureAccumulator, QueueBackpressureKey, QueueBackpressureSnapshot,
    StageTimingState, TransferMetadataAccumulator, TransferMetadataKey, TransferMetadataSnapshot,
};
