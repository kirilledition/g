//! Coarse native run orchestration above the prepared engine.

use std::sync::Arc;
use std::time::Instant;

use g_runtime::{StageTimingRecorder, TelemetryRunError, TelemetryRunSession};
use serde::Serialize;

use crate::backend::AssociationBackend;
use crate::delivery_execution::AssociationDeliveryReport;
use crate::progress::{RunProgressError, RunProgressReporter};
use crate::run::{RunEngine, RunExecutionError, RunHooks, RunPreparationError};

const ASSOCIATION_BACKEND_SELECTED_EVENT_NAME: &str = "association_backend_selected";
const EXECUTION_PLAN_PREPARED_EVENT_NAME: &str = "execution_plan_prepared";
const WRITER_FINISHED_EVENT_NAME: &str = "writer_finished";

#[derive(Serialize)]
struct ExecutionPlanPreparedTelemetryFields<'fields> {
    association_mode: &'fields str,
    trait_type: &'fields str,
    phenotype_count: i64,
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
    phenotype_count: Option<i64>,
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
    phenotype_count: i64,
    parquet_dataset_paths: &'fields [&'fields str],
}

#[derive(Serialize)]
struct EmptyDiagnosticFields {}

#[derive(Serialize)]
struct ExecutionPlanDispatchDiagnosticFields<'fields> {
    phenotype_count: i64,
    association_mode: &'fields str,
}

#[derive(Serialize)]
struct DeliveryFinishedDiagnosticFields {
    group_index: usize,
    processed_chunk_count: i64,
}

#[derive(Serialize)]
struct ArtifactsCompletedDiagnosticFields<'fields> {
    association_mode: &'fields str,
    phenotype_count: i64,
}

#[derive(Debug, thiserror::Error)]
pub(crate) enum CoordinatedRunDetailError<BackendError, HookError> {
    #[error("Native run preparation failed: {0}")]
    Preparation(#[from] RunPreparationError),
    #[error("Native run execution failed: {0}")]
    Execution(#[from] RunExecutionError<BackendError, HookError>),
    #[error("Native run telemetry failed: {0}")]
    Telemetry(#[from] TelemetryRunError),
    #[error("Native run progress reporting failed: {0}")]
    Progress(#[from] RunProgressError),
    #[error("Native run diagnostic serialization failed: {0}")]
    Diagnostic(#[from] serde_json::Error),
    #[error("Phenotype count exceeds native int64 telemetry capacity.")]
    PhenotypeCountOutOfRange,
    #[error("Processed chunk count exceeds native int64 telemetry capacity.")]
    ProcessedChunkCountOutOfRange,
    #[error("Association warning count exceeds native uint64 telemetry capacity.")]
    AssociationWarningCountOutOfRange,
    #[error("Native run completed without a phenotype output.")]
    MissingPhenotypeOutput,
}

/// Failure reported by the coarse engine entry point.
#[derive(Debug, thiserror::Error)]
pub enum EngineRunError<HookError> {
    #[error("{message}")]
    Failure { message: String },
    #[error("Native run was interrupted.")]
    Interrupted(HookError),
}

/// One phenotype output produced by a completed engine run.
#[derive(Debug, Eq, PartialEq)]
pub struct PhenotypeRunArtifact {
    pub output_run_directory: String,
    pub parquet_dataset_directory: String,
}

/// Prepare, execute, observe, and describe one native association run.
///
/// # Errors
///
/// Returns a typed preparation, execution, telemetry, or diagnostic error.
pub fn execute_coordinated_run<Backend, Hooks>(
    run_plan: g_plan::RunPlan,
    effective_config_toml: String,
    backend: Arc<Backend>,
    hooks: &mut Hooks,
    telemetry_session: &TelemetryRunSession,
    thread_name: &str,
    stage_timing_recorder: Option<&mut StageTimingRecorder>,
) -> Result<Vec<PhenotypeRunArtifact>, EngineRunError<Hooks::Error>>
where
    Backend: AssociationBackend + 'static,
    Hooks: RunHooks,
{
    execute_coordinated_run_detail(
        run_plan,
        effective_config_toml,
        backend,
        hooks,
        telemetry_session,
        thread_name,
        stage_timing_recorder,
    )
    .map_err(engine_run_error_from_detail)
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
    run_plan: g_plan::RunPlan,
    effective_config_toml: String,
    backend: Arc<Backend>,
    hooks: &mut Hooks,
    telemetry_session: &TelemetryRunSession,
    thread_name: &str,
    mut stage_timing_recorder: Option<&mut StageTimingRecorder>,
) -> Result<Vec<PhenotypeRunArtifact>, CoordinatedRunDetailError<Backend::Error, Hooks::Error>>
where
    Backend: AssociationBackend + 'static,
    Hooks: RunHooks,
{
    let phenotype_count = i64::try_from(run_plan.phenotype_runs.len())
        .map_err(|_| CoordinatedRunDetailError::PhenotypeCountOutOfRange)?;
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

    g_runtime::emit_diagnostic_event(
        "debug",
        "runner_execution_plan_build_started",
        "Building REGENIE execution plan.",
        &EmptyDiagnosticFields {},
    )?;
    let preparation_start_time = Instant::now();
    let prepared_run = RunEngine::open(run_plan, effective_config_toml)?.prepare()?;
    let resolved_gpu_genotype_format = prepared_run.resolved_gpu_genotype_format();
    let association_backend_kind = match resolved_gpu_genotype_format {
        g_plan::GpuGenotypeFormat::Dosage => "jax_dosage",
        g_plan::GpuGenotypeFormat::Packed8 => "jax_packed8",
    };
    let execution_plan_fields = ExecutionPlanPreparedTelemetryFields {
        association_mode: association_mode.as_str(),
        trait_type,
        phenotype_count,
        chunk_size: i64::from(chunk_size),
        device: device.as_str(),
    };
    if telemetry_session.is_enabled() {
        telemetry_session.emit_current_event(
            thread_name,
            EXECUTION_PLAN_PREPARED_EVENT_NAME,
            "info",
            &execution_plan_fields,
        )?;
    } else {
        g_runtime::emit_diagnostic_event(
            "info",
            "runner_execution_plan_prepared",
            "Prepared REGENIE execution plan.",
            &execution_plan_fields,
        )?;
    }
    telemetry_session.emit_current_event(
        thread_name,
        ASSOCIATION_BACKEND_SELECTED_EVENT_NAME,
        "info",
        &AssociationBackendSelectedTelemetryFields {
            association_mode: association_mode.as_str(),
            association_backend_kind,
            device: device.as_str(),
            genotype_format: resolved_gpu_genotype_format.as_str(),
            phenotype: single_phenotype_name.as_deref(),
            phenotype_count: (phenotype_count > 1).then_some(phenotype_count),
        },
    )?;
    record_stage_duration(stage_timing_recorder.as_deref_mut(), "native_run_preparation", preparation_start_time);

    g_runtime::emit_diagnostic_event(
        "debug",
        "runner_execution_plan_dispatch_started",
        "Dispatching REGENIE execution plan.",
        &ExecutionPlanDispatchDiagnosticFields { phenotype_count, association_mode: association_mode.as_str() },
    )?;
    let execution_start_time = Instant::now();
    let execution = prepared_run.execute_with_progress(backend, hooks, progress_reporter.as_ref())?;
    record_stage_duration(stage_timing_recorder, "native_run_execution", execution_start_time);

    if let Some(progress_reporter) = progress_reporter {
        progress_reporter.finish()?;
    }

    record_delivery_reports::<Backend::Error, Hooks::Error>(&execution.delivery_reports)?;
    complete_artifacts::<Backend::Error, Hooks::Error>(
        execution.completed_outputs,
        telemetry_session,
        thread_name,
        association_mode,
        phenotype_count,
        single_phenotype_name.as_deref(),
    )
}

fn record_delivery_reports<BackendError, HookError>(
    delivery_reports: &[AssociationDeliveryReport],
) -> Result<(), CoordinatedRunDetailError<BackendError, HookError>> {
    for (group_index, report) in delivery_reports.iter().enumerate() {
        let processed_chunk_count = i64::try_from(report.processed_chunk_count)
            .map_err(|_| CoordinatedRunDetailError::ProcessedChunkCountOutOfRange)?;
        g_runtime::emit_diagnostic_event(
            "debug",
            "native_dispatch_delivery_finished",
            "Association group delivery finished.",
            &DeliveryFinishedDiagnosticFields { group_index, processed_chunk_count },
        )?;
        for warning in &report.warnings {
            let nonconverged_count = u64::try_from(warning.nonconverged_count)
                .map_err(|_| CoordinatedRunDetailError::AssociationWarningCountOutOfRange)?;
            let total_fit_count = u64::try_from(warning.total_fit_count)
                .map_err(|_| CoordinatedRunDetailError::AssociationWarningCountOutOfRange)?;
            tracing::warn!(
                target: "g.engine",
                g_event = "association_delivery_warning",
                chromosome = warning.chromosome,
                nonconverged_count,
                total_fit_count,
                "{}",
                warning.message
            );
        }
    }
    Ok(())
}

fn complete_artifacts<BackendError, HookError>(
    completed_outputs: Vec<g_output::CompletedOutputRun>,
    telemetry_session: &TelemetryRunSession,
    thread_name: &str,
    association_mode: g_plan::AssociationMode,
    phenotype_count: i64,
    single_phenotype_name: Option<&str>,
) -> Result<Vec<PhenotypeRunArtifact>, CoordinatedRunDetailError<BackendError, HookError>> {
    let artifacts = completed_outputs
        .into_iter()
        .map(|run| PhenotypeRunArtifact {
            output_run_directory: run.run_directory.display().to_string(),
            parquet_dataset_directory: run.parts_directory.display().to_string(),
        })
        .collect::<Vec<_>>();
    if phenotype_count == 1 {
        let phenotype_name = single_phenotype_name.ok_or(CoordinatedRunDetailError::MissingPhenotypeOutput)?;
        let artifact = artifacts.first().ok_or(CoordinatedRunDetailError::MissingPhenotypeOutput)?;
        telemetry_session.emit_current_event(
            thread_name,
            WRITER_FINISHED_EVENT_NAME,
            "info",
            &PhenotypeWriterFinishedTelemetryFields {
                association_mode: association_mode.as_str(),
                phenotype: phenotype_name,
                parquet_dataset_path: &artifact.parquet_dataset_directory,
            },
        )?;
    } else {
        let parquet_dataset_paths =
            artifacts.iter().map(|artifact| artifact.parquet_dataset_directory.as_str()).collect::<Vec<_>>();
        telemetry_session.emit_current_event(
            thread_name,
            WRITER_FINISHED_EVENT_NAME,
            "info",
            &MultiPhenotypeWriterFinishedTelemetryFields {
                association_mode: association_mode.as_str(),
                phenotype_count,
                parquet_dataset_paths: &parquet_dataset_paths,
            },
        )?;
    }
    g_runtime::emit_diagnostic_event(
        "info",
        "runner_metadata_artifacts_completed",
        "Completed REGENIE run artifacts.",
        &ArtifactsCompletedDiagnosticFields { association_mode: association_mode.as_str(), phenotype_count },
    )?;
    Ok(artifacts)
}

fn record_stage_duration(recorder: Option<&mut StageTimingRecorder>, stage_name: &str, start_time: Instant) {
    if let Some(recorder) = recorder {
        recorder.add_stage_duration(stage_name, start_time.elapsed().as_secs_f64());
    }
}
