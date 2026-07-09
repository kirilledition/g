//! Public runtime crate facade.

pub use crate::cli_runtime::{
    CLI_RUNTIME_FAILURE_EXIT_CODE, CliExitCodeRangeError, CliOutputBuffer, CliOutputChunks,
    CliRunFailedTelemetryEmissionPlan, CliRunFailureTelemetryPlan, CliRunLifecycleState, CliTelemetryCloseFailurePlan,
    CliTerminalResult, NATIVE_CLI_OUTPUT_LOG_LIMIT, build_completed_cli_terminal_result,
    build_failed_cli_terminal_result, build_interrupted_cli_terminal_result,
    build_telemetry_close_failure_cli_terminal_result, plan_cli_run_failed_telemetry_emission,
    plan_cli_telemetry_close_failure,
};
pub use crate::error::{RuntimeCompatibilityError, RuntimeError, RuntimeResult};
pub use crate::events::{
    AssociationBackendSelectedTelemetryFields, BgenEngineOpenedTelemetryFields, EffectiveConfigWrittenTelemetryFields,
    ExecutionPlanPreparedTelemetryFields, GpuGenotypeFormatResolvedTelemetryFields,
    MultiPhenotypePreflightCompletedTelemetryFields, MultiPhenotypeSampleSummaryTelemetryFields,
    MultiPhenotypeWriterFinishedTelemetryFields, PhenotypeWriterFinishedTelemetryFields,
    PredictionSourceLoadedTelemetryFields, RunArtifactPayload, RunArtifactTelemetryFields, RunArtifactsPayload,
    RunCompletedEventPayload, RunCompletedTelemetryFields, RunDiagnosticEventPayload, RunDiagnosticFieldPayload,
    RunDiagnosticFieldValue, RunFailedEventPayload, RunFailedTelemetryFields, RunInterruptedEventPayload,
    RunInterruptedTelemetryFields, RunStartedTelemetryFields, RunTelemetryEventKind, RunTelemetryStringField,
    SampleAlignmentCompletedTelemetryFields, SingleTraitPreflightCompletedTelemetryFields,
    attach_run_metadata_to_artifacts, build_artifact_telemetry_fields,
    build_association_backend_selected_telemetry_fields, build_bgen_engine_opened_telemetry_fields,
    build_callback_null_logistic_nonconvergence_warning_diagnostic_payload,
    build_effective_config_written_telemetry_fields, build_execution_plan_prepared_telemetry_fields,
    build_gpu_genotype_format_resolved_telemetry_fields, build_io_output_resume_committed_chunks_diagnostic_payload,
    build_multi_phenotype_preflight_completed_telemetry_fields, build_multi_phenotype_sample_summary_telemetry_fields,
    build_multi_phenotype_writer_finished_telemetry_fields, build_native_cli_completed_line_diagnostic_payload,
    build_native_cli_failed_line_diagnostic_payload, build_native_cli_interrupted_line_diagnostic_payload,
    build_native_cli_stderr_diagnostic_payload, build_native_cli_stdout_diagnostic_payload,
    build_native_dispatch_bgen_engine_constructing_diagnostic_payload,
    build_native_dispatch_callback_drain_started_diagnostic_payload,
    build_native_dispatch_delivery_failed_diagnostic_payload,
    build_native_dispatch_delivery_finished_diagnostic_payload,
    build_native_dispatch_delivery_interrupted_diagnostic_payload,
    build_native_dispatch_delivery_started_diagnostic_payload,
    build_native_dispatch_pipeline_finished_diagnostic_payload,
    build_native_dispatch_trusted_bgen_validation_started_diagnostic_payload,
    build_native_dispatch_writer_session_finish_started_diagnostic_payload,
    build_native_dispatch_writer_session_interrupted_flush_started_diagnostic_payload,
    build_native_dispatch_writer_sessions_finish_started_diagnostic_payload,
    build_native_dispatch_writer_sessions_interrupted_flush_started_diagnostic_payload,
    build_native_runtime_knobs_configured_diagnostic_payload, build_phenotype_writer_finished_telemetry_fields,
    build_pipeline_bgen_engine_open_started_diagnostic_payload, build_pipeline_bgen_engine_opened_diagnostic_payload,
    build_pipeline_gpu_genotype_format_resolved_diagnostic_payload,
    build_pipeline_grouped_per_phenotype_groups_prepared_diagnostic_payload,
    build_pipeline_grouped_per_phenotype_started_diagnostic_payload,
    build_pipeline_grouped_union_delivery_selected_diagnostic_payload,
    build_pipeline_multi_group_preflight_completed_diagnostic_payload,
    build_pipeline_multi_group_preflight_started_diagnostic_payload,
    build_pipeline_multi_phenotype_sample_summary_diagnostic_payload,
    build_pipeline_multi_trait_input_aligned_diagnostic_payload,
    build_pipeline_multi_trait_input_load_started_diagnostic_payload,
    build_pipeline_multi_trait_prediction_source_load_started_diagnostic_payload,
    build_pipeline_multi_trait_started_diagnostic_payload,
    build_pipeline_output_resume_committed_chunks_diagnostic_payload,
    build_pipeline_output_writer_sessions_create_started_diagnostic_payload,
    build_pipeline_prevalidated_bgen_engine_used_diagnostic_payload,
    build_pipeline_single_trait_input_aligned_diagnostic_payload,
    build_pipeline_single_trait_input_load_started_diagnostic_payload,
    build_pipeline_single_trait_prediction_source_load_started_diagnostic_payload,
    build_pipeline_single_trait_preflight_completed_diagnostic_payload,
    build_pipeline_single_trait_preflight_started_diagnostic_payload,
    build_pipeline_single_trait_started_diagnostic_payload, build_prediction_source_loaded_telemetry_fields,
    build_preflight_warning_diagnostic_payload, build_run_completed_event_from_artifacts,
    build_run_completed_telemetry_fields, build_run_failed_event_payload, build_run_failed_telemetry_fields,
    build_run_interrupted_event_payload, build_run_interrupted_telemetry_fields, build_run_started_telemetry_fields,
    build_runner_binary_engine_dispatch_started_diagnostic_payload,
    build_runner_execution_plan_build_started_diagnostic_payload,
    build_runner_execution_plan_dispatch_started_diagnostic_payload,
    build_runner_execution_plan_finalization_started_diagnostic_payload,
    build_runner_execution_plan_prepared_diagnostic_payload,
    build_runner_jax_runtime_configuration_started_diagnostic_payload,
    build_runner_linear_engine_dispatch_started_diagnostic_payload,
    build_runner_metadata_artifacts_finalized_diagnostic_payload,
    build_runner_multi_phenotype_binary_engine_dispatch_started_diagnostic_payload,
    build_runner_multi_phenotype_dispatch_started_diagnostic_payload,
    build_runner_multi_phenotype_linear_engine_dispatch_started_diagnostic_payload,
    build_runner_run_completed_diagnostic_payload, build_runner_run_failed_diagnostic_payload,
    build_runner_run_interrupted_diagnostic_payload, build_runner_run_started_diagnostic_payload,
    build_runner_single_phenotype_dispatch_started_diagnostic_payload,
    build_sample_alignment_completed_telemetry_fields, build_single_trait_preflight_completed_telemetry_fields,
    flatten_run_artifact_payloads, render_artifact_lines, render_run_completed_lines, render_run_failed_lines,
    render_run_interrupted_lines, serialize_run_diagnostic_fields_json,
};
pub use crate::jax_runtime::{
    JaxDeviceObservation, JaxGpuValidationPlan, JaxRuntimeConfigUpdatePayload, JaxRuntimeConfigValue,
    JaxRuntimeDiagnosticEventPayload, JaxRuntimeDiagnosticFieldPayload, JaxRuntimeDiagnosticRecordPlan,
    JaxRuntimeDiagnosticValue, JaxRuntimeSetupPayload, JaxRuntimeSetupSession, JaxRuntimeSetupSideEffectPlan,
    NvidiaDriverProbePathsPayload, build_jax_runtime_setup_diagnostic_events, complete_jax_runtime_setup_validation,
    default_nvidia_driver_probe_paths, nvidia_driver_files_are_visible, plan_jax_gpu_validation,
    plan_jax_runtime_config_updates, plan_jax_runtime_diagnostic_record, plan_jax_runtime_setup_side_effects,
    resolve_jax_runtime_setup, serialize_jax_runtime_diagnostic_fields_json,
};
pub use crate::logging_sink::{
    LoggingSinkConfig, LoggingSinkError, LoggingSinkInitializationError, initialize_logging_sinks,
    shutdown_logging_sinks,
};
pub use crate::rayon_runtime::{
    RayonRuntimeError, configure_global_rayon_thread_pool, format_global_rayon_thread_pool_configuration_error,
};
pub use crate::run_metadata::{
    ExecutionRunArtifactsInput, ExecutionRunArtifactsSequenceInput, PhenotypeRunArtifactsInput,
    RunManifestCommandPayload, RunManifestExtensionInput, RunManifestExtensionPayload, RunManifestRuntimePayload,
    RunMetadataError, build_execution_run_artifacts, build_execution_run_artifacts_from_sequences,
    build_multi_run_artifacts, build_phenotype_run_artifacts, build_run_manifest_extension,
};
pub use crate::runtime_paths::{build_default_local_cache_directory, default_local_cache_directory};
pub use crate::runtime_policy::{
    LoggingRuntimePolicyPayload, build_logging_runtime_policy, describe_logging_runtime_policy,
};
pub use crate::runtime_state::{
    JaxRuntimePolicyPayload, JaxRuntimeSetupLifecyclePlan, ProcessRuntimeState, RayonThreadPoolConfigurationError,
    RayonThreadPoolConfigurationPlan, RunRuntime, RuntimeCompatibilityToken, RuntimePolicyPayload,
    RuntimeStateSnapshotPayload, build_jax_runtime_policy_payload, describe_jax_runtime_policy,
    resolve_jax_runtime_cache_directory,
};
pub use crate::shutdown::{
    SecondSignalExceptionPlan, ShutdownControllerState, ShutdownError, ShutdownHandlerInstallPlan,
    ShutdownHandlerRestorePlan, ShutdownHandlerSession, ShutdownRequestAction, ShutdownRequestDecisionPayload,
    ShutdownSignalPayload, build_shutdown_signal, default_shutdown_signal_numbers, plan_second_signal_exception,
};
pub use crate::telemetry_policy::{
    TelemetryMode, TelemetryPathError, TelemetryPathsPayload, TelemetrySessionPolicyPayload,
    TelemetryWriterCountersPayload, build_empty_writer_counters, format_timestamp, paths_refer_to_same_file,
    resolve_output_run_root, resolve_telemetry_paths, resolve_telemetry_session_policy, resolve_telemetry_stream_file,
};
pub use crate::telemetry_session::{
    TelemetryCapAction, TelemetryCloseEventPayload, TelemetryCloseMetadataPayload, TelemetryClosePlan,
    TelemetryEventCapState, TelemetryEventEmissionPlan, TelemetryEventEnvelope, TelemetryProgressEmissionPlan,
    TelemetryProgressThrottleState, TelemetryRunSessionState, TelemetryRunSessionWriterPlan,
    TelemetryWriterCounterSnapshot, build_current_telemetry_event_envelope, build_telemetry_close_event_payload,
    build_telemetry_close_metadata, build_telemetry_event_envelope, generate_run_id, plan_telemetry_close,
    plan_telemetry_event_emission, plan_telemetry_progress_emission, serialize_telemetry_payload_json_line,
};
pub use crate::telemetry_writer::{
    TelemetryLineWriter, TelemetrySessionWriter, TelemetryWriterFactory, TelemetryWriterGuard, build_log_file_writer,
    build_non_blocking_writer, build_shared_or_log_file_writer, build_telemetry_file_writer,
    clear_shared_telemetry_writer, normalize_event_cap, replace_shared_telemetry_writer,
    shared_telemetry_writer_for_path,
};
pub use crate::timing::diagnostics::{
    TimingDiagnosticError, build_multi_null_logistic_diagnostics, build_scalar_null_logistic_diagnostics,
};
pub use crate::timing::{
    ChunkStageSummary, ChunkStageTiming, FinalTimingOutputContext, FinalTimingOutputsWriteResultPayload,
    FinalTimingOutputsWriteStartedDiagnosticPayload, NullLogisticDiagnosticValue, NullLogisticSummary,
    NumericDiagnosticValue, ProfileSummaryPayload, QueueBackpressureAccumulator, QueueBackpressureKey,
    QueueBackpressureSnapshot, StageTimingRecorder, StageTimingRecorderPlan, StageTimingSnapshotPayload,
    StageTimingState, TimingFileError, TimingFileWritePlan, TransferMetadataAccumulator, TransferMetadataError,
    TransferMetadataKey, TransferMetadataObservation, TransferMetadataSnapshot,
    build_final_timing_outputs_write_started_diagnostic_payload, build_transfer_metadata_observation,
    plan_stage_timing_recorder, plan_timing_file_write, resolve_final_timing_output_context,
    serialize_final_timing_outputs_write_started_diagnostic_fields_json, write_profile_summary_payload,
    write_stage_timing_snapshot_payload,
};
pub use crate::trusted_validation::{
    TrustedBgenValidationCacheDirectoryError, TrustedBgenValidationCacheLookupError,
    TrustedBgenValidationCacheLookupPlan, TrustedBgenValidationCachePayload, TrustedBgenValidationFingerprintInput,
    build_default_trusted_bgen_validation_cache_directory, build_trusted_bgen_validation_cache_path,
    build_trusted_bgen_validation_cache_payload, build_trusted_bgen_validation_fingerprint,
    default_trusted_bgen_validation_cache_directory, plan_trusted_bgen_validation_cache_lookup,
    require_cache_backed_trusted_bgen_validation_mode, serialize_trusted_bgen_validation_cache_payload,
    write_trusted_bgen_validation_cache_payload, write_trusted_bgen_validation_cache_payload_to_path,
};
