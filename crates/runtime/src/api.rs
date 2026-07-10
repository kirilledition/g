//! Public runtime crate facade.

pub use crate::cli_runtime::{
    CLI_RUNTIME_FAILURE_EXIT_CODE, CliOutputBuffer, CliTerminalResult, NATIVE_CLI_OUTPUT_LOG_LIMIT,
};
pub use crate::error::RuntimeCompatibilityError;
pub use crate::jax_runtime::{
    JaxDeviceObservation, JaxGpuValidationPlan, JaxRuntimeConfigUpdatePayload, JaxRuntimeConfigValue,
    JaxRuntimeDiagnosticEventPayload, JaxRuntimeDiagnosticFieldPayload, JaxRuntimeDiagnosticFields,
    JaxRuntimeDiagnosticRecordPlan, JaxRuntimeDiagnosticValue, JaxRuntimeSetupPayload, JaxRuntimeSetupSession,
    NvidiaDriverProbePathsPayload, default_nvidia_driver_probe_paths, nvidia_driver_files_are_visible,
    plan_jax_gpu_validation, plan_jax_runtime_diagnostic_record,
};
pub use crate::logging_sink::{
    LoggingSinkConfig, LoggingSinkError, LoggingSinkInitializationError, initialize_logging_sinks,
    shutdown_logging_sinks,
};
pub use crate::native_run_session::{NativeRunSession, NativeRunSessionError};
pub use crate::rayon_runtime::{
    RayonRuntimeError, configure_global_rayon_thread_pool, format_global_rayon_thread_pool_configuration_error,
};
pub use crate::run_events::{
    PhenotypeRunArtifacts, RunDiagnosticEventPayload, RunDiagnosticFieldPayload, RunDiagnosticFieldValue,
    RunFailedEventPayload, RunInterruptedEventPayload, build_native_cli_completed_line_diagnostic_payload,
    build_native_cli_failed_line_diagnostic_payload, build_native_cli_interrupted_line_diagnostic_payload,
    build_native_cli_stderr_diagnostic_payload, build_native_cli_stdout_diagnostic_payload,
    build_native_dispatch_delivery_finished_diagnostic_payload,
    build_native_runtime_knobs_configured_diagnostic_payload, build_run_failed_event_payload,
    build_run_interrupted_event_payload, build_runner_execution_plan_build_started_diagnostic_payload,
    build_runner_execution_plan_dispatch_started_diagnostic_payload,
    build_runner_execution_plan_prepared_diagnostic_payload,
    build_runner_metadata_artifacts_completed_diagnostic_payload, emit_diagnostic_event, emit_run_diagnostic_event,
    render_run_completed_lines, render_run_failed_lines, render_run_interrupted_lines,
};
pub use crate::runtime_policy::{LoggingRuntimePolicyPayload, describe_logging_runtime_policy};
pub use crate::runtime_state::{
    JaxRuntimePolicyPayload, ProcessRuntimeState, RayonThreadPoolConfigurationError, build_jax_runtime_policy_payload,
    describe_jax_runtime_policy, resolve_jax_runtime_cache_directory,
};
pub use crate::shutdown::{
    ShutdownError, ShutdownSignalPayload, SigtermShutdownScope, begin_sigterm_shutdown_scope, build_shutdown_signal,
    sigterm_shutdown_requested,
};
pub use crate::telemetry_policy::TelemetryPathError;
pub use crate::telemetry_session::{TelemetryRunError, TelemetryRunSession, generate_run_id};
pub use crate::timing::{
    FinalTimingOutputsWriteStartedDiagnosticPayload, StageTimingRecorder, TimingFileError,
    build_final_timing_outputs_write_started_diagnostic_payload,
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
pub use g_plan::TelemetryMode;
