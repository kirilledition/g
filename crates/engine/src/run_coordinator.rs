//! Coarse native run orchestration above the prepared engine.

use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::Arc;
use std::time::Instant;
use std::{error::Error, fmt};

use g_runtime::{StageTimingRecorder, TelemetryRunSession};
use serde::Serialize;

use crate::backend::AssociationBackend;
use crate::delivery_execution::AssociationDeliveryReport;
use crate::progress::RunProgressReporter;
use crate::run::{
    EnginePostSessionCleanup, RunActivationError, RunActivationOutcome, RunEngine, RunExecution, RunExecutionError,
    RunExecutionOutcome, RunHooks, RunPreparationError,
};

const ASSOCIATION_BACKEND_SELECTED_EVENT_NAME: &str = "association_backend_selected";
const ASSOCIATION_IMPLEMENTATION_SELECTED_EVENT_NAME: &str = "association_implementation_selected";
const EXECUTION_PLAN_PREPARED_EVENT_NAME: &str = "execution_plan_prepared";
const WRITER_FINISHED_EVENT_NAME: &str = "writer_finished";

#[derive(Serialize)]
struct ExecutionPlanPreparedTelemetryFields<'fields> {
    association_mode: &'fields str,
    trait_type: &'fields str,
    phenotype_count: u64,
    chunk_size: i64,
    device: &'fields str,
}

#[derive(Serialize)]
struct AssociationBackendSelectedTelemetryFields<'fields> {
    association_mode: &'fields str,
    association_backend_kind: &'fields str,
    device: &'fields str,
    genotype_format: &'fields str,
    #[serde(skip_serializing_if = "Option::is_none")]
    phenotype: Option<&'fields str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    phenotype_count: Option<u64>,
}

#[derive(Serialize)]
struct AssociationImplementationSelectedTelemetryFields<'fields> {
    association_mode: &'fields str,
    jax_version: &'fields str,
    jaxlib_version: &'fields str,
    #[serde(skip_serializing_if = "Option::is_none")]
    firth_components_requested: Option<&'fields str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    firth_components_effective: Option<&'fields str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    firth_components_fallback_reason: Option<&'fields str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    raw_cuda_ffi_target: Option<&'fields str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    raw_cuda_ffi_api_version: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    raw_cuda_handler_sha256: Option<&'fields str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    raw_cuda_ptx_sha256: Option<&'fields str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    raw_cuda_ptx_isa: Option<&'fields str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    raw_cuda_ptx_target: Option<&'fields str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    raw_cuda_minimum_cuda_driver_version: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    raw_cuda_minimum_compute_capability_major: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    raw_cuda_minimum_compute_capability_minor: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    cuda_driver_version: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    cuda_device_ordinal: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    cuda_compute_capability_major: Option<i32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    cuda_compute_capability_minor: Option<i32>,
}

#[derive(Serialize)]
struct PhenotypeWriterFinishedTelemetryFields<'fields> {
    association_mode: &'fields str,
    phenotype: &'fields str,
    parquet_dataset_path: &'fields str,
}

#[derive(Serialize)]
struct MultiPhenotypeWriterFinishedTelemetryFields<'fields> {
    association_mode: &'fields str,
    phenotype_count: u64,
    parquet_dataset_paths: &'fields [&'fields str],
}

#[derive(Serialize)]
struct EmptyDiagnosticFields {}

#[derive(Serialize)]
struct ExecutionPlanDispatchDiagnosticFields<'fields> {
    phenotype_count: u64,
    association_mode: &'fields str,
}

#[derive(Serialize)]
struct DeliveryFinishedDiagnosticFields {
    group_index: usize,
    processed_chunk_count: u64,
}

#[derive(Serialize)]
struct ArtifactsCompletedDiagnosticFields<'fields> {
    association_mode: &'fields str,
    phenotype_count: u64,
}

#[derive(Debug, thiserror::Error)]
pub(crate) enum CoordinatedRunDetailError<BackendError, HookError> {
    #[error("Native run activation failed: {0}")]
    Activation(#[source] RunPreparationError),
    #[error("Native run preparation failed: {0}")]
    Preparation(#[from] RunPreparationError),
    #[error("Native run execution failed: {0}")]
    Execution(#[from] RunExecutionError<BackendError, HookError>),
    #[error("Native run completed without a phenotype output.")]
    MissingPhenotypeOutput,
    #[error("Native run completed with {actual} phenotype outputs, but the run plan requires {expected}.")]
    PhenotypeOutputCountMismatch { expected: usize, actual: usize },
    #[error("Native run completed with a non-absolute or non-UTF-8 output path.")]
    InvalidCompletedOutputPath,
}

/// Failure reported by the coarse engine entry point.
#[derive(Debug, thiserror::Error)]
pub enum EngineRunError<HookError> {
    #[error("{message}")]
    Failure { message: String },
    #[error("Native run was interrupted.")]
    Interrupted(HookError),
}

/// Failure before an output claim can be returned to the runner.
#[derive(Debug)]
pub struct EngineClaimError {
    message: String,
}

impl fmt::Display for EngineClaimError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.message.fmt(formatter)
    }
}

impl Error for EngineClaimError {}

/// Prepared native inputs with exclusive output ownership and no attempt authority.
pub struct ClaimedCoordinatedRun {
    run: crate::run::ClaimedRun,
}

impl ClaimedCoordinatedRun {
    /// Return the canonical plan associated with this claim.
    #[must_use]
    pub fn run_plan(&self) -> &g_plan::RunPlan {
        self.run.run_plan()
    }

    /// Return the ownership-private diagnostics directory.
    ///
    /// # Errors
    ///
    /// Returns an error if claim staging is inconsistent.
    pub fn diagnostics_directory(&self) -> Result<&std::path::Path, EngineClaimError> {
        self.run.diagnostics_directory().map_err(|error| engine_claim_error(&error))
    }

    /// Remove the unpublished attempt reservation and release ownership.
    ///
    /// Run-scoped diagnostics must be closed first.
    ///
    /// # Errors
    ///
    /// Returns an output error when staging cleanup or claim release fails.
    pub fn abort_before_activation(self) -> Result<(), g_output::OutputError> {
        self.run.abort_before_activation()
    }
}

/// One phenotype output produced by a completed engine run.
#[derive(Debug, Eq, PartialEq)]
pub struct PhenotypeRunArtifact {
    pub output_run_directory: String,
    pub parquet_dataset_directory: String,
}

/// One fixed engine outcome plus optional cleanup deferred until session close.
#[must_use = "the primary engine result and any post-session cleanup must both be handled"]
pub struct EngineExecutionOutcome<HookError> {
    /// Completed artifacts or the fixed primary engine failure.
    pub result: Result<Vec<PhenotypeRunArtifact>, EngineRunError<HookError>>,
    /// Completed-noop cleanup that must run after diagnostics close.
    pub post_session_cleanup: Option<EnginePostSessionCleanup>,
}

struct CoordinatedRunDetailOutcome<BackendError, HookError, PostSessionCleanup = EnginePostSessionCleanup> {
    result: Result<Vec<PhenotypeRunArtifact>, CoordinatedRunDetailError<BackendError, HookError>>,
    post_session_cleanup: Option<PostSessionCleanup>,
}

/// Prepare read-only inputs and acquire exclusive output ownership.
///
/// # Errors
///
/// Returns an error before a native run session is opened when input
/// preparation, output policy validation, or claim acquisition fails.
pub fn claim_coordinated_run(
    run_plan: g_plan::RunPlan,
    effective_config_toml: String,
) -> Result<ClaimedCoordinatedRun, EngineClaimError> {
    let run = RunEngine::open(run_plan, effective_config_toml)
        .and_then(RunEngine::prepare)
        .map_err(|error| engine_claim_error(&error))?;
    Ok(ClaimedCoordinatedRun { run })
}

fn engine_claim_error(error: &impl ToString) -> EngineClaimError {
    EngineClaimError { message: error.to_string() }
}

/// Activate, execute, observe, and describe one claimed native association run.
///
/// The returned `result` contains any typed preparation, execution, or
/// required-artifact error. Observer emission failures are recorded as
/// warnings. Any post-session cleanup authority remains present independently
/// of that result.
pub fn execute_coordinated_run<Backend, Hooks>(
    claimed_run: ClaimedCoordinatedRun,
    backend: Arc<Backend>,
    hooks: &mut Hooks,
    telemetry_session: &TelemetryRunSession,
    thread_name: &str,
    stage_timing_recorder: Option<&mut StageTimingRecorder>,
) -> EngineExecutionOutcome<Hooks::Error>
where
    Backend: AssociationBackend + 'static,
    Hooks: RunHooks,
{
    let outcome = execute_coordinated_run_detail(
        claimed_run,
        backend,
        hooks,
        telemetry_session,
        thread_name,
        stage_timing_recorder,
    );
    EngineExecutionOutcome {
        result: outcome.result.map_err(engine_run_error_from_detail),
        post_session_cleanup: outcome.post_session_cleanup,
    }
}

fn engine_run_error_from_detail<BackendError, HookError>(
    error: CoordinatedRunDetailError<BackendError, HookError>,
) -> EngineRunError<HookError>
where
    BackendError: std::error::Error,
    HookError: std::error::Error,
{
    match error {
        CoordinatedRunDetailError::Execution(RunExecutionError::Interrupted(error)) => {
            EngineRunError::Interrupted(error)
        }
        error => EngineRunError::Failure { message: error.to_string() },
    }
}

fn execute_coordinated_run_detail<Backend, Hooks>(
    claimed_run: ClaimedCoordinatedRun,
    backend: Arc<Backend>,
    hooks: &mut Hooks,
    telemetry_session: &TelemetryRunSession,
    thread_name: &str,
    mut stage_timing_recorder: Option<&mut StageTimingRecorder>,
) -> CoordinatedRunDetailOutcome<Backend::Error, Hooks::Error>
where
    Backend: AssociationBackend + 'static,
    Hooks: RunHooks,
{
    let run_plan = claimed_run.run.run_plan();
    let phenotype_count = run_plan.phenotype_runs.len();
    let telemetry_phenotype_count = telemetry_counter_from_usize(phenotype_count);
    let association_mode = run_plan.association_mode;
    let trait_type = match association_mode {
        g_plan::AssociationMode::Regenie2Linear => "quantitative",
        g_plan::AssociationMode::Regenie2Binary => "binary",
    };
    let chunk_size = run_plan.chunk_size;
    let device = run_plan.compute.device;
    let single_phenotype_name =
        run_plan.phenotype_runs.first().filter(|_| phenotype_count == 1).map(|run| run.phenotype_name.clone());
    let progress_reporter = telemetry_session.is_enabled().then(|| {
        Arc::new(RunProgressReporter::new(telemetry_session.clone(), thread_name.to_string(), association_mode))
    });
    let association_implementation_state = backend.association_implementation_state();
    let association_implementation_fields =
        association_implementation_selected_fields(association_mode, &association_implementation_state);
    record_association_implementation_selection(telemetry_session, thread_name, &association_implementation_fields);

    record_execution_plan_build_started();
    let preparation_start_time = Instant::now();
    let activation_outcome = match association_implementation_state.output_compatibility() {
        Ok(association_implementation) => claimed_run.run.activate(association_implementation),
        Err(error) => claimed_run.run.reject_activation(error),
    };
    let RunActivationOutcome { result: activation_result, post_session_cleanup } = activation_outcome;
    let prepared_run = match activation_result {
        Ok(prepared_run) => prepared_run,
        Err(RunActivationError::Unpublished { source, rollback }) => {
            debug_assert!(post_session_cleanup.is_none());
            return CoordinatedRunDetailOutcome {
                result: Err(CoordinatedRunDetailError::Activation(RunPreparationError::Output(source))),
                post_session_cleanup: Some(EnginePostSessionCleanup::pre_activation_rollback(rollback)),
            };
        }
        Err(RunActivationError::Published(error)) => {
            return CoordinatedRunDetailOutcome {
                result: Err(CoordinatedRunDetailError::Activation(error)),
                post_session_cleanup: post_session_cleanup.map(EnginePostSessionCleanup::completed_noop),
            };
        }
    };
    debug_assert!(post_session_cleanup.is_none());
    let resolved_gpu_genotype_format = prepared_run.resolved_gpu_genotype_format();
    let association_backend_kind = match resolved_gpu_genotype_format {
        g_plan::GpuGenotypeFormat::Dosage => "jax_dosage",
        g_plan::GpuGenotypeFormat::Packed8 => "jax_packed8",
    };
    let execution_plan_fields = ExecutionPlanPreparedTelemetryFields {
        association_mode: association_mode.as_str(),
        trait_type,
        phenotype_count: telemetry_phenotype_count,
        chunk_size: i64::from(chunk_size),
        device: device.as_str(),
    };
    let association_backend_fields = AssociationBackendSelectedTelemetryFields {
        association_mode: association_mode.as_str(),
        association_backend_kind,
        device: device.as_str(),
        genotype_format: resolved_gpu_genotype_format.as_str(),
        phenotype: single_phenotype_name.as_deref(),
        phenotype_count: (phenotype_count > 1).then_some(telemetry_phenotype_count),
    };
    record_execution_plan_observations(
        telemetry_session,
        thread_name,
        &execution_plan_fields,
        &association_backend_fields,
    );
    record_stage_duration(stage_timing_recorder.as_deref_mut(), "native_run_preparation", preparation_start_time);

    record_execution_plan_dispatch_started(telemetry_phenotype_count, association_mode);
    let execution_start_time = Instant::now();
    let RunExecutionOutcome { result: execution_result, post_session_cleanup } =
        prepared_run.execute_with_progress(backend, hooks, progress_reporter.as_ref());
    let execution = match execution_result {
        Ok(execution) => execution,
        Err(error) => {
            return CoordinatedRunDetailOutcome {
                result: Err(error.into()),
                post_session_cleanup: post_session_cleanup.map(EnginePostSessionCleanup::completed_noop),
            };
        }
    };
    record_stage_duration(stage_timing_recorder, "native_run_execution", execution_start_time);

    if let Some(progress_reporter) = progress_reporter {
        progress_reporter.finish();
    }

    let RunExecution { completed_outputs, delivery_reports } = execution;
    record_delivery_reports(&delivery_reports);
    let result = complete_artifacts::<Backend::Error, Hooks::Error, _>(
        completed_outputs,
        telemetry_session,
        thread_name,
        association_mode,
        phenotype_count,
        single_phenotype_name.as_deref(),
        emit_artifacts_completed_diagnostic,
    );
    coordinated_run_detail_outcome(result, post_session_cleanup.map(EnginePostSessionCleanup::completed_noop))
}

fn record_execution_plan_build_started() {
    record_diagnostic_observation("runner_execution_plan_build_started", || {
        g_runtime::emit_diagnostic_event(
            "debug",
            "runner_execution_plan_build_started",
            "Building REGENIE execution plan.",
            &EmptyDiagnosticFields {},
        )
    });
}

fn record_execution_plan_dispatch_started(phenotype_count: u64, association_mode: g_plan::AssociationMode) {
    record_diagnostic_observation("runner_execution_plan_dispatch_started", || {
        g_runtime::emit_diagnostic_event(
            "debug",
            "runner_execution_plan_dispatch_started",
            "Dispatching REGENIE execution plan.",
            &ExecutionPlanDispatchDiagnosticFields { phenotype_count, association_mode: association_mode.as_str() },
        )
    });
}

fn association_implementation_selected_fields(
    association_mode: g_plan::AssociationMode,
    state: &crate::AssociationImplementationState,
) -> AssociationImplementationSelectedTelemetryFields<'_> {
    let runtime_versions = state.jax_runtime_versions();
    let firth_components = state.firth_components();
    let raw_cuda_artifact = firth_components.and_then(crate::FirthComponentsImplementationState::raw_cuda_artifact);
    let raw_cuda_observation =
        firth_components.and_then(crate::FirthComponentsImplementationState::raw_cuda_observation);
    AssociationImplementationSelectedTelemetryFields {
        association_mode: association_mode.as_str(),
        jax_version: runtime_versions.jax_version(),
        jaxlib_version: runtime_versions.jaxlib_version(),
        firth_components_requested: firth_components.map(|selection| selection.requested().stable_name()),
        firth_components_effective: firth_components.map(|selection| selection.effective().stable_name()),
        firth_components_fallback_reason: firth_components
            .and_then(crate::FirthComponentsImplementationState::fallback_reason)
            .map(crate::FirthComponentsFallbackReason::stable_name),
        raw_cuda_ffi_target: raw_cuda_artifact.map(crate::RawCudaFirthArtifactIdentity::ffi_target),
        raw_cuda_ffi_api_version: raw_cuda_artifact.map(crate::RawCudaFirthArtifactIdentity::ffi_api_version),
        raw_cuda_handler_sha256: raw_cuda_artifact.map(crate::RawCudaFirthArtifactIdentity::handler_sha256),
        raw_cuda_ptx_sha256: raw_cuda_artifact.map(crate::RawCudaFirthArtifactIdentity::ptx_sha256),
        raw_cuda_ptx_isa: raw_cuda_artifact.map(crate::RawCudaFirthArtifactIdentity::ptx_isa),
        raw_cuda_ptx_target: raw_cuda_artifact.map(crate::RawCudaFirthArtifactIdentity::ptx_target),
        raw_cuda_minimum_cuda_driver_version: raw_cuda_artifact
            .map(crate::RawCudaFirthArtifactIdentity::minimum_cuda_driver_version),
        raw_cuda_minimum_compute_capability_major: raw_cuda_artifact
            .map(crate::RawCudaFirthArtifactIdentity::minimum_compute_capability_major),
        raw_cuda_minimum_compute_capability_minor: raw_cuda_artifact
            .map(crate::RawCudaFirthArtifactIdentity::minimum_compute_capability_minor),
        cuda_driver_version: raw_cuda_observation.and_then(crate::RawCudaFirthRuntimeObservation::cuda_driver_version),
        cuda_device_ordinal: raw_cuda_observation.and_then(crate::RawCudaFirthRuntimeObservation::device_ordinal),
        cuda_compute_capability_major: raw_cuda_observation
            .and_then(crate::RawCudaFirthRuntimeObservation::compute_capability_major),
        cuda_compute_capability_minor: raw_cuda_observation
            .and_then(crate::RawCudaFirthRuntimeObservation::compute_capability_minor),
    }
}

fn record_association_implementation_selection(
    telemetry_session: &TelemetryRunSession,
    thread_name: &str,
    fields: &AssociationImplementationSelectedTelemetryFields<'_>,
) {
    if telemetry_session.is_enabled() {
        record_telemetry_observation(ASSOCIATION_IMPLEMENTATION_SELECTED_EVENT_NAME, || {
            telemetry_session.emit_current_event(
                thread_name,
                ASSOCIATION_IMPLEMENTATION_SELECTED_EVENT_NAME,
                "info",
                fields,
            )
        });
    } else {
        record_diagnostic_observation(ASSOCIATION_IMPLEMENTATION_SELECTED_EVENT_NAME, || {
            g_runtime::emit_diagnostic_event(
                "info",
                ASSOCIATION_IMPLEMENTATION_SELECTED_EVENT_NAME,
                "Selected association compute implementation.",
                fields,
            )
        });
    }
}

fn record_execution_plan_observations(
    telemetry_session: &TelemetryRunSession,
    thread_name: &str,
    execution_plan_fields: &ExecutionPlanPreparedTelemetryFields<'_>,
    association_backend_fields: &AssociationBackendSelectedTelemetryFields<'_>,
) {
    if telemetry_session.is_enabled() {
        record_telemetry_observation(EXECUTION_PLAN_PREPARED_EVENT_NAME, || {
            telemetry_session.emit_current_event(
                thread_name,
                EXECUTION_PLAN_PREPARED_EVENT_NAME,
                "info",
                execution_plan_fields,
            )
        });
    } else {
        record_diagnostic_observation("runner_execution_plan_prepared", || {
            g_runtime::emit_diagnostic_event(
                "info",
                "runner_execution_plan_prepared",
                "Prepared REGENIE execution plan.",
                execution_plan_fields,
            )
        });
    }
    record_telemetry_observation(ASSOCIATION_BACKEND_SELECTED_EVENT_NAME, || {
        telemetry_session.emit_current_event(
            thread_name,
            ASSOCIATION_BACKEND_SELECTED_EVENT_NAME,
            "info",
            association_backend_fields,
        )
    });
}

fn coordinated_run_detail_outcome<BackendError, HookError, PostSessionCleanup>(
    result: Result<Vec<PhenotypeRunArtifact>, CoordinatedRunDetailError<BackendError, HookError>>,
    post_session_cleanup: Option<PostSessionCleanup>,
) -> CoordinatedRunDetailOutcome<BackendError, HookError, PostSessionCleanup> {
    CoordinatedRunDetailOutcome { result, post_session_cleanup }
}

fn record_delivery_reports(delivery_reports: &[AssociationDeliveryReport]) {
    for (group_index, report) in delivery_reports.iter().enumerate() {
        let processed_chunk_count = telemetry_counter_from_usize(report.processed_chunk_count);
        record_diagnostic_observation("native_dispatch_delivery_finished", || {
            g_runtime::emit_diagnostic_event(
                "debug",
                "native_dispatch_delivery_finished",
                "Association group delivery finished.",
                &DeliveryFinishedDiagnosticFields { group_index, processed_chunk_count },
            )
        });
        for warning in &report.warnings {
            let nonconverged_count = telemetry_counter_from_usize(warning.nonconverged_count);
            let total_fit_count = telemetry_counter_from_usize(warning.total_fit_count);
            record_warning_observation(|| {
                tracing::warn!(
                    target: "g.engine",
                    g_event = "association_delivery_warning",
                    chromosome = warning.chromosome,
                    nonconverged_count,
                    total_fit_count,
                    "{}",
                    warning.message
                );
            });
        }
    }
}

fn complete_artifacts<BackendError, HookError, EmitDiagnostic>(
    completed_outputs: Vec<g_output::CompletedOutputRun>,
    telemetry_session: &TelemetryRunSession,
    thread_name: &str,
    association_mode: g_plan::AssociationMode,
    phenotype_count: usize,
    single_phenotype_name: Option<&str>,
    emit_diagnostic: EmitDiagnostic,
) -> Result<Vec<PhenotypeRunArtifact>, CoordinatedRunDetailError<BackendError, HookError>>
where
    EmitDiagnostic: FnOnce(&ArtifactsCompletedDiagnosticFields<'_>) -> Result<(), g_runtime::DiagnosticEventError>,
{
    if completed_outputs.len() != phenotype_count {
        return Err(CoordinatedRunDetailError::PhenotypeOutputCountMismatch {
            expected: phenotype_count,
            actual: completed_outputs.len(),
        });
    }
    let telemetry_phenotype_count = telemetry_counter_from_usize(phenotype_count);
    let artifacts = completed_outputs
        .into_iter()
        .map(|run| {
            if !run.run_directory.is_absolute() || !run.parts_directory.is_absolute() {
                return Err(CoordinatedRunDetailError::InvalidCompletedOutputPath);
            }
            Ok(PhenotypeRunArtifact {
                output_run_directory: run
                    .run_directory
                    .to_str()
                    .ok_or(CoordinatedRunDetailError::InvalidCompletedOutputPath)?
                    .to_string(),
                parquet_dataset_directory: run
                    .parts_directory
                    .to_str()
                    .ok_or(CoordinatedRunDetailError::InvalidCompletedOutputPath)?
                    .to_string(),
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    if phenotype_count == 1 {
        let phenotype_name = single_phenotype_name.ok_or(CoordinatedRunDetailError::MissingPhenotypeOutput)?;
        let artifact = artifacts.first().ok_or(CoordinatedRunDetailError::MissingPhenotypeOutput)?;
        record_telemetry_observation(WRITER_FINISHED_EVENT_NAME, || {
            telemetry_session.emit_current_event(
                thread_name,
                WRITER_FINISHED_EVENT_NAME,
                "info",
                &PhenotypeWriterFinishedTelemetryFields {
                    association_mode: association_mode.as_str(),
                    phenotype: phenotype_name,
                    parquet_dataset_path: &artifact.parquet_dataset_directory,
                },
            )
        });
    } else {
        let parquet_dataset_paths =
            artifacts.iter().map(|artifact| artifact.parquet_dataset_directory.as_str()).collect::<Vec<_>>();
        record_telemetry_observation(WRITER_FINISHED_EVENT_NAME, || {
            telemetry_session.emit_current_event(
                thread_name,
                WRITER_FINISHED_EVENT_NAME,
                "info",
                &MultiPhenotypeWriterFinishedTelemetryFields {
                    association_mode: association_mode.as_str(),
                    phenotype_count: telemetry_phenotype_count,
                    parquet_dataset_paths: &parquet_dataset_paths,
                },
            )
        });
    }
    record_diagnostic_observation("runner_metadata_artifacts_completed", || {
        emit_diagnostic(&ArtifactsCompletedDiagnosticFields {
            association_mode: association_mode.as_str(),
            phenotype_count: telemetry_phenotype_count,
        })
    });
    Ok(artifacts)
}

fn emit_artifacts_completed_diagnostic(
    fields: &ArtifactsCompletedDiagnosticFields<'_>,
) -> Result<(), g_runtime::DiagnosticEventError> {
    g_runtime::emit_diagnostic_event(
        "info",
        "runner_metadata_artifacts_completed",
        "Completed REGENIE run artifacts.",
        fields,
    )
}

fn record_diagnostic_observation<ObserveDiagnostic>(event_name: &str, observe_diagnostic: ObserveDiagnostic)
where
    ObserveDiagnostic: FnOnce() -> Result<(), g_runtime::DiagnosticEventError>,
{
    match catch_unwind(AssertUnwindSafe(observe_diagnostic)) {
        Ok(Ok(())) => {}
        Ok(Err(error)) => record_warning_observation(|| {
            tracing::warn!(
                target: "g.engine",
                error = %error,
                diagnostic_event = event_name,
                "Failed to emit native engine diagnostic event."
            );
        }),
        Err(_) => record_warning_observation(|| {
            tracing::warn!(
                target: "g.engine",
                diagnostic_event = event_name,
                "Native engine diagnostic observation panicked."
            );
        }),
    }
}

fn record_telemetry_observation<ObserveTelemetry>(event_name: &str, observe_telemetry: ObserveTelemetry)
where
    ObserveTelemetry: FnOnce() -> Result<(), g_runtime::TelemetryRunError>,
{
    match catch_unwind(AssertUnwindSafe(observe_telemetry)) {
        Ok(Ok(())) => {}
        Ok(Err(error)) => record_warning_observation(|| {
            tracing::warn!(
                target: "g.engine",
                error = %error,
                telemetry_event = event_name,
                "Failed to emit native engine telemetry event."
            );
        }),
        Err(_) => record_warning_observation(|| {
            tracing::warn!(
                target: "g.engine",
                telemetry_event = event_name,
                "Native engine telemetry observation panicked."
            );
        }),
    }
}

fn record_warning_observation<ObserveWarning>(observe_warning: ObserveWarning)
where
    ObserveWarning: FnOnce(),
{
    let _ = catch_unwind(AssertUnwindSafe(observe_warning));
}

fn telemetry_counter_from_usize(value: usize) -> u64 {
    u64::try_from(value).unwrap_or(u64::MAX)
}

fn record_stage_duration(recorder: Option<&mut StageTimingRecorder>, stage_name: &str, start_time: Instant) {
    if let Some(recorder) = recorder {
        recorder.add_stage_duration(stage_name, start_time.elapsed().as_secs_f64());
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicBool, Ordering};

    use serde::Serializer;
    use serde::ser::Error as _;

    use super::*;

    struct SerializationFailure;

    impl Serialize for SerializationFailure {
        fn serialize<SerializerType>(
            &self,
            _serializer: SerializerType,
        ) -> Result<SerializerType::Ok, SerializerType::Error>
        where
            SerializerType: Serializer,
        {
            Err(SerializerType::Error::custom("intentional engine diagnostic serialization failure"))
        }
    }

    #[derive(Debug, Eq, PartialEq, thiserror::Error)]
    #[error("test backend failure")]
    struct TestBackendError;

    #[derive(Debug, Eq, PartialEq, thiserror::Error)]
    #[error("test interruption")]
    struct TestInterruption;

    struct CleanupReacquisitionSentinel {
        owner_active: Arc<AtomicBool>,
    }

    impl CleanupReacquisitionSentinel {
        fn cleanup(self) {
            assert!(self.owner_active.swap(false, Ordering::SeqCst), "test cleanup releases an active owner");
        }

        fn reacquire(owner_active: &AtomicBool) {
            assert!(
                owner_active.compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst).is_ok(),
                "test cleanup permits exact owner reacquisition"
            );
        }
    }

    #[test]
    fn error_boundary_preserves_interruptions_and_renders_other_failures() {
        let interruption = engine_run_error_from_detail::<TestBackendError, TestInterruption>(
            CoordinatedRunDetailError::Execution(RunExecutionError::Interrupted(TestInterruption)),
        );
        assert!(matches!(interruption, EngineRunError::Interrupted(TestInterruption)));

        let failure = engine_run_error_from_detail::<TestBackendError, TestInterruption>(
            CoordinatedRunDetailError::MissingPhenotypeOutput,
        );
        assert!(matches!(
            failure,
            EngineRunError::Failure { message }
                if message == "Native run completed without a phenotype output."
        ));
    }

    #[test]
    fn diagnostic_serialization_failure_does_not_replace_completed_artifacts() {
        let serialization_error =
            serde_json::to_string(&SerializationFailure).expect_err("test serializer should fail");
        let runtime_error = g_runtime::DiagnosticEventError::from(serialization_error);
        let artifacts = complete_artifacts::<TestBackendError, TestInterruption, _>(
            vec![g_output::CompletedOutputRun {
                run_directory: "/tmp/durable-run".into(),
                parts_directory: "/tmp/durable-run/parts".into(),
            }],
            &TelemetryRunSession::default(),
            "test-thread",
            g_plan::AssociationMode::Regenie2Linear,
            1,
            Some("trait"),
            |_fields| Err(runtime_error),
        )
        .expect("diagnostic serialization failure must not replace completed artifacts");
        assert_eq!(
            artifacts,
            [PhenotypeRunArtifact {
                output_run_directory: "/tmp/durable-run".to_string(),
                parquet_dataset_directory: "/tmp/durable-run/parts".to_string(),
            }]
        );
    }

    #[test]
    fn association_selection_observation_is_stable_and_excludes_free_text() {
        let capability_requirements = crate::RawCudaFirthCapabilityRequirements::new(12_020, 7, 0)
            .expect("valid test raw-CUDA capability requirements");
        let artifact = crate::RawCudaFirthArtifactIdentity::new(
            "g.firth.components.test.v0",
            1,
            "abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789",
            "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
            "8.2",
            "sm_70",
            capability_requirements,
        )
        .expect("valid test raw-CUDA artifact");
        let observation = crate::RawCudaFirthRuntimeObservation::cuda_driver_too_old(12_010)
            .expect("valid test CUDA fallback observation");
        let firth_components = crate::FirthComponentsImplementationState::raw_cuda_fallback(
            artifact,
            observation,
            crate::FirthComponentsFallbackReason::CudaDriverTooOld,
            "host-specific driver diagnostic".to_string(),
        )
        .expect("reason-matched fallback observations are valid");
        let state = crate::AssociationImplementationState::jax(
            crate::JaxRuntimeVersions::new("0.11.0".to_string(), "0.11.0".to_string()).expect("valid JAX versions"),
            Some(firth_components),
        );

        let fields = association_implementation_selected_fields(g_plan::AssociationMode::Regenie2Binary, &state);
        let value = serde_json::to_value(fields).expect("selection fields serialize");

        assert_eq!(value["association_mode"], "regenie2_binary");
        assert_eq!(value["jax_version"], "0.11.0");
        assert_eq!(value["firth_components_requested"], "raw_cuda");
        assert_eq!(value["firth_components_effective"], "jax");
        assert_eq!(value["firth_components_fallback_reason"], "cuda_driver_too_old");
        assert_eq!(value["raw_cuda_ffi_target"], "g.firth.components.test.v0");
        assert_eq!(value["raw_cuda_ffi_api_version"], 1);
        assert_eq!(
            value["raw_cuda_handler_sha256"],
            "abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789"
        );
        assert_eq!(value["raw_cuda_ptx_sha256"], "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef");
        assert_eq!(value["raw_cuda_ptx_isa"], "8.2");
        assert_eq!(value["raw_cuda_ptx_target"], "sm_70");
        assert_eq!(value["raw_cuda_minimum_cuda_driver_version"], 12_020);
        assert_eq!(value["raw_cuda_minimum_compute_capability_major"], 7);
        assert_eq!(value["raw_cuda_minimum_compute_capability_minor"], 0);
        assert_eq!(value["cuda_driver_version"], 12_010);
        assert!(value.get("cuda_device_ordinal").is_none());
        assert!(value.get("cuda_compute_capability_major").is_none());
        assert!(value.get("cuda_compute_capability_minor").is_none());
        assert!(!value.to_string().contains("host-specific driver diagnostic"));
    }

    #[test]
    fn artifact_completion_preserves_output_order_and_validates_single_output() {
        let telemetry_session = TelemetryRunSession::default();
        let outputs = vec![
            g_output::CompletedOutputRun {
                run_directory: "/tmp/run-a".into(),
                parts_directory: "/tmp/run-a/parts".into(),
            },
            g_output::CompletedOutputRun {
                run_directory: "/tmp/run-b".into(),
                parts_directory: "/tmp/run-b/parts".into(),
            },
        ];
        let artifacts = complete_artifacts::<TestBackendError, TestInterruption, _>(
            outputs,
            &telemetry_session,
            "test-thread",
            g_plan::AssociationMode::Regenie2Binary,
            2,
            None,
            emit_artifacts_completed_diagnostic,
        )
        .expect("multi-phenotype artifacts complete");
        assert_eq!(
            artifacts,
            vec![
                PhenotypeRunArtifact {
                    output_run_directory: "/tmp/run-a".to_string(),
                    parquet_dataset_directory: "/tmp/run-a/parts".to_string(),
                },
                PhenotypeRunArtifact {
                    output_run_directory: "/tmp/run-b".to_string(),
                    parquet_dataset_directory: "/tmp/run-b/parts".to_string(),
                },
            ]
        );

        assert!(matches!(
            complete_artifacts::<TestBackendError, TestInterruption, _>(
                Vec::new(),
                &telemetry_session,
                "test-thread",
                g_plan::AssociationMode::Regenie2Linear,
                1,
                Some("trait"),
                emit_artifacts_completed_diagnostic,
            ),
            Err(CoordinatedRunDetailError::PhenotypeOutputCountMismatch { expected: 1, actual: 0 })
        ));

        let single_artifact = complete_artifacts::<TestBackendError, TestInterruption, _>(
            vec![g_output::CompletedOutputRun {
                run_directory: "/tmp/run-single".into(),
                parts_directory: "/tmp/run-single/parts".into(),
            }],
            &telemetry_session,
            "test-thread",
            g_plan::AssociationMode::Regenie2Linear,
            1,
            Some("trait"),
            emit_artifacts_completed_diagnostic,
        )
        .expect("single-phenotype artifact completes");
        assert_eq!(single_artifact[0].output_run_directory, "/tmp/run-single");

        assert!(matches!(
            complete_artifacts::<TestBackendError, TestInterruption, _>(
                vec![g_output::CompletedOutputRun {
                    run_directory: "/tmp/run-a".into(),
                    parts_directory: "/tmp/run-a/parts".into(),
                }],
                &telemetry_session,
                "test-thread",
                g_plan::AssociationMode::Regenie2Linear,
                2,
                None,
                emit_artifacts_completed_diagnostic,
            ),
            Err(CoordinatedRunDetailError::PhenotypeOutputCountMismatch { expected: 2, actual: 1 })
        ));

        assert!(matches!(
            complete_artifacts::<TestBackendError, TestInterruption, _>(
                Vec::new(),
                &telemetry_session,
                "test-thread",
                g_plan::AssociationMode::Regenie2Linear,
                usize::MAX,
                None,
                emit_artifacts_completed_diagnostic,
            ),
            Err(CoordinatedRunDetailError::PhenotypeOutputCountMismatch { expected: usize::MAX, actual: 0 })
        ));
    }

    #[test]
    fn output_count_conversion_failure_retains_post_session_cleanup() {
        // Paired with g-output's
        // `completed_claim_has_no_cleanup_authority_before_read_only_finalization`,
        // which proves the real token releases and permits reacquisition.
        let owner_active = Arc::new(AtomicBool::new(true));
        let conversion_result = complete_artifacts::<TestBackendError, TestInterruption, _>(
            Vec::new(),
            &TelemetryRunSession::default(),
            "test-thread",
            g_plan::AssociationMode::Regenie2Linear,
            1,
            Some("trait"),
            emit_artifacts_completed_diagnostic,
        );
        let outcome = coordinated_run_detail_outcome(
            conversion_result,
            Some(CleanupReacquisitionSentinel { owner_active: Arc::clone(&owner_active) }),
        );

        assert!(matches!(
            outcome.result,
            Err(CoordinatedRunDetailError::PhenotypeOutputCountMismatch { expected: 1, actual: 0 })
        ));
        outcome.post_session_cleanup.expect("cleanup authority survives count conversion failure").cleanup();
        CleanupReacquisitionSentinel::reacquire(&owner_active);
    }

    #[test]
    fn output_path_conversion_failure_retains_post_session_cleanup() {
        // This exercises the engine half of the paired real-token g-output
        // cleanup-and-reacquisition regression named above.
        let owner_active = Arc::new(AtomicBool::new(true));
        let conversion_result = complete_artifacts::<TestBackendError, TestInterruption, _>(
            vec![g_output::CompletedOutputRun {
                run_directory: "relative-run".into(),
                parts_directory: "relative-run/parts".into(),
            }],
            &TelemetryRunSession::default(),
            "test-thread",
            g_plan::AssociationMode::Regenie2Linear,
            1,
            Some("trait"),
            emit_artifacts_completed_diagnostic,
        );
        let outcome = coordinated_run_detail_outcome(
            conversion_result,
            Some(CleanupReacquisitionSentinel { owner_active: Arc::clone(&owner_active) }),
        );

        assert!(matches!(outcome.result, Err(CoordinatedRunDetailError::InvalidCompletedOutputPath)));
        outcome.post_session_cleanup.expect("cleanup authority survives path conversion failure").cleanup();
        CleanupReacquisitionSentinel::reacquire(&owner_active);
    }

    #[test]
    fn final_progress_failure_does_not_replace_completed_artifacts() {
        let progress_reporter = Arc::new(RunProgressReporter::new(
            TelemetryRunSession::default(),
            "test-thread".to_string(),
            g_plan::AssociationMode::Regenie2Linear,
        ));
        let totals = progress_reporter.totals_from_chunk_specs(&[]);
        let _delivery = progress_reporter.register_delivery("trait".to_string(), totals);
        progress_reporter.finish();

        let artifacts = complete_artifacts::<TestBackendError, TestInterruption, _>(
            vec![g_output::CompletedOutputRun {
                run_directory: "/tmp/run-after-progress-error".into(),
                parts_directory: "/tmp/run-after-progress-error/parts".into(),
            }],
            &TelemetryRunSession::default(),
            "test-thread",
            g_plan::AssociationMode::Regenie2Linear,
            1,
            Some("trait"),
            emit_artifacts_completed_diagnostic,
        )
        .expect("final progress failure must not replace completed artifacts");
        assert_eq!(artifacts[0].output_run_directory, "/tmp/run-after-progress-error");
    }

    #[test]
    fn delivery_report_observation_handles_usize_max_counters_infallibly() {
        record_delivery_reports(&[AssociationDeliveryReport {
            processed_chunk_count: usize::MAX,
            warnings: vec![crate::delivery_execution::DeliveryWarning {
                chromosome: "22".to_string(),
                message: "test warning".to_string(),
                nonconverged_count: usize::MAX,
                total_fit_count: usize::MAX,
            }],
        }]);

        assert_eq!(telemetry_counter_from_usize(usize::MAX), u64::MAX);
    }

    #[test]
    fn panicking_observers_and_warning_reporting_are_contained() {
        record_diagnostic_observation("panicking-diagnostic", || -> Result<(), g_runtime::DiagnosticEventError> {
            panic!("intentional diagnostic panic")
        });
        record_telemetry_observation("panicking-telemetry", || -> Result<(), g_runtime::TelemetryRunError> {
            panic!("intentional telemetry panic")
        });
        record_warning_observation(|| panic!("intentional warning panic"));
    }

    #[test]
    fn stage_duration_recording_is_optional_and_mutates_enabled_recorder() {
        let mut recorder = StageTimingRecorder::default();
        assert_eq!(recorder, StageTimingRecorder::default());
        record_stage_duration(Some(&mut recorder), "test-stage", Instant::now());
        assert_ne!(recorder, StageTimingRecorder::default());
        record_stage_duration(None, "ignored-stage", Instant::now());
    }
}
