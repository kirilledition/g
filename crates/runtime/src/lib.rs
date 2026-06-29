#![warn(clippy::pedantic)]

pub mod jax_runtime;
pub mod rayon_runtime;
pub mod run_events;
pub mod run_metadata;
pub mod runtime_paths;
pub mod runtime_policy;
pub mod runtime_state;
pub mod shutdown;
pub mod telemetry_policy;
pub mod telemetry_session;
pub mod timing;

pub use jax_runtime::{
    JaxDeviceObservation, JaxGpuValidationPlan, JaxRuntimeConfigUpdatePayload, JaxRuntimeConfigValue,
    JaxRuntimeDiagnosticEventPayload, JaxRuntimeDiagnosticFieldPayload, JaxRuntimeDiagnosticRecordPlan,
    JaxRuntimeDiagnosticValue, JaxRuntimeSetupPayload, JaxRuntimeSetupSideEffectPlan,
    build_jax_runtime_setup_diagnostic_events, complete_jax_runtime_setup_validation, nvidia_driver_files_are_visible,
    plan_jax_gpu_validation, plan_jax_runtime_config_updates, plan_jax_runtime_diagnostic_record,
    plan_jax_runtime_setup_side_effects, resolve_jax_runtime_setup,
};
pub use rayon_runtime::{
    RayonRuntimeError, configure_global_rayon_thread_pool, format_global_rayon_thread_pool_configuration_error,
};
pub use run_events::{
    EFFECTIVE_CONFIG_WRITTEN_EVENT_NAME, EXECUTION_PLAN_PREPARED_EVENT_NAME, EffectiveConfigWrittenTelemetryFields,
    ExecutionPlanPreparedTelemetryFields, MultiPhenotypePreflightCompletedTelemetryFields,
    MultiPhenotypeWriterFinishedTelemetryFields, PREFLIGHT_COMPLETED_EVENT_NAME,
    PhenotypeWriterFinishedTelemetryFields, RUN_COMPLETED_EVENT_NAME, RUN_FAILED_EVENT_NAME, RUN_LIFECYCLE_ERROR_LEVEL,
    RUN_LIFECYCLE_INFO_LEVEL, RUN_LIFECYCLE_WARN_LEVEL, RUN_STARTED_EVENT_NAME, RunArtifactPayload,
    RunArtifactTelemetryFields, RunCompletedEventPayload, RunCompletedTelemetryFields, RunFailedEventPayload,
    RunFailedTelemetryFields, RunInterruptedEventPayload, RunInterruptedTelemetryFields, RunStartedTelemetryFields,
    RunTelemetryStringField, SingleTraitPreflightCompletedTelemetryFields, WRITER_FINISHED_EVENT_NAME,
    build_artifact_telemetry_fields, build_effective_config_written_telemetry_fields,
    build_execution_plan_prepared_telemetry_fields, build_multi_phenotype_preflight_completed_telemetry_fields,
    build_multi_phenotype_writer_finished_telemetry_fields, build_phenotype_writer_finished_telemetry_fields,
    build_run_completed_telemetry_fields, build_run_failed_telemetry_fields, build_run_interrupted_telemetry_fields,
    build_run_started_telemetry_fields, build_single_trait_preflight_completed_telemetry_fields, render_artifact_lines,
    render_run_completed_lines, render_run_failed_lines, render_run_interrupted_lines,
};
pub use run_metadata::{
    ExecutionRunArtifactsInput, PhenotypeRunArtifactsInput, RunArtifactsPayload, RunManifestCommandPayload,
    RunManifestExtensionInput, RunManifestExtensionPayload, RunManifestRuntimePayload, build_execution_run_artifacts,
    build_multi_run_artifacts, build_phenotype_run_artifacts, build_run_manifest_extension,
};
pub use runtime_paths::build_default_local_cache_directory;
pub use runtime_policy::{LoggingRuntimePolicyPayload, build_logging_runtime_policy, describe_logging_runtime_policy};
pub use runtime_state::{
    JaxRuntimePolicyPayload, JaxRuntimeSetupLifecyclePlan, ProcessRuntimeState, RayonThreadPoolConfigurationPlan,
    RunRuntime, RuntimeCompatibilityError, RuntimeCompatibilityToken, RuntimePolicyPayload,
    build_jax_runtime_policy_payload, describe_jax_runtime_policy,
};
pub use shutdown::{
    SecondSignalExceptionPlan, ShutdownControllerState, ShutdownRequestAction, ShutdownRequestDecisionPayload,
    ShutdownSignalPayload, build_shutdown_signal, plan_second_signal_exception,
};
pub use telemetry_policy::{
    TelemetryPathsPayload, TelemetryWriterCountersPayload, build_empty_writer_counters, format_timestamp,
    paths_refer_to_same_file, resolve_output_run_root, resolve_telemetry_paths, resolve_telemetry_stream_file,
};
pub use telemetry_session::{
    TelemetryCapAction, TelemetryCloseMetadataPayload, TelemetryClosePlan, TelemetryEventCapState,
    TelemetryEventEmissionPlan, TelemetryEventEnvelope, TelemetryProgressEmissionPlan, TelemetryProgressThrottleState,
    TelemetryWriterCounterSnapshot, build_telemetry_close_metadata, build_telemetry_event_envelope, generate_run_id,
    plan_telemetry_close, plan_telemetry_event_emission, plan_telemetry_progress_emission,
};
pub use timing::{
    ChunkStageSummary, ChunkStageTiming, NullLogisticDiagnosticValue, NullLogisticSummary, NumericDiagnosticValue,
    ProfileSummaryPayload, QueueBackpressureAccumulator, QueueBackpressureKey, QueueBackpressureSnapshot,
    StageTimingRecorderPlan, StageTimingSnapshotPayload, StageTimingState, TimingFileError, TimingFileWritePlan,
    TransferMetadataAccumulator, TransferMetadataError, TransferMetadataKey, TransferMetadataObservation,
    TransferMetadataSnapshot, build_transfer_metadata_observation, plan_stage_timing_recorder, plan_timing_file_write,
    should_collect_exact_stage_timings, write_profile_summary_payload, write_stage_timing_snapshot_payload,
};
