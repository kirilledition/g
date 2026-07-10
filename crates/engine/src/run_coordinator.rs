//! Coarse native run orchestration above the prepared engine.

use std::sync::Arc;
use std::time::Instant;

use g_runtime::{PhenotypeRunArtifacts, StageTimingRecorder, TelemetryRunError, TelemetryRunSession};

use crate::backend::AssociationBackend;
use crate::delivery_execution::AssociationDeliveryReport;
use crate::progress::{RunProgressError, RunProgressReporter};
use crate::run::{RunEngine, RunExecutionError, RunHooks, RunPreparationError};

#[derive(Debug, thiserror::Error)]
pub enum CoordinatedRunError<BackendError, HookError> {
    #[error("Native run preparation failed.")]
    Preparation(#[from] RunPreparationError),
    #[error("Native run execution failed.")]
    Execution(#[from] RunExecutionError<BackendError, HookError>),
    #[error("Native run telemetry failed.")]
    Telemetry(#[from] TelemetryRunError),
    #[error("Native run progress reporting failed.")]
    Progress(#[from] RunProgressError),
    #[error("Native run diagnostic serialization failed.")]
    Diagnostic(#[from] serde_json::Error),
    #[error("Phenotype count exceeds native int64 telemetry capacity.")]
    PhenotypeCountOutOfRange,
    #[error("Processed chunk count exceeds native int64 telemetry capacity.")]
    ProcessedChunkCountOutOfRange,
    #[error("Association warning count exceeds native uint64 telemetry capacity.")]
    AssociationWarningCountOutOfRange,
    #[error("Prepared run retained unresolved GPU genotype format.")]
    UnresolvedGpuGenotypeFormat,
    #[error("Native run completed without a phenotype output.")]
    MissingPhenotypeOutput,
}

/// Prepare, execute, observe, and describe one native association run.
///
/// # Errors
///
/// Returns a typed preparation, execution, telemetry, or diagnostic error.
#[allow(clippy::too_many_arguments)]
pub fn execute_coordinated_run<Backend, Hooks>(
    run_plan: g_plan::RunPlan,
    effective_config_toml: String,
    backend: Arc<Backend>,
    hooks: &mut Hooks,
    telemetry_session: &TelemetryRunSession,
    thread_name: &str,
    mut stage_timing_recorder: Option<&mut StageTimingRecorder>,
) -> Result<Vec<PhenotypeRunArtifacts>, CoordinatedRunError<Backend::Error, Hooks::Error>>
where
    Backend: AssociationBackend + 'static,
    Backend::ChromosomeState: 'static,
    Backend::DeviceResult: 'static,
    Hooks: RunHooks,
{
    let phenotype_count =
        i64::try_from(run_plan.phenotype_runs.len()).map_err(|_| CoordinatedRunError::PhenotypeCountOutOfRange)?;
    let association_mode = run_plan.association_mode;
    let trait_type = run_plan.analysis.trait_type;
    let chunk_size = run_plan.analysis.chunk_size;
    let variant_limit = run_plan.compute.variant_limit;
    let device = run_plan.compute.device;
    let single_phenotype_name =
        run_plan.phenotype_runs.first().filter(|_| phenotype_count == 1).map(|run| run.phenotype_name.clone());
    let progress_reporter = (run_plan.diagnostics.telemetry != g_plan::TelemetryMode::Off)
        .then(|| Arc::new(RunProgressReporter::new(telemetry_session.clone(), thread_name.to_string())));

    g_runtime::emit_run_diagnostic_event(&g_runtime::build_runner_execution_plan_build_started_diagnostic_payload())?;
    let preparation_start_time = Instant::now();
    let prepared_run = RunEngine::open(run_plan, effective_config_toml)?.prepare()?;
    let resolved_gpu_genotype_format = prepared_run.resolved_gpu_genotype_format();
    let association_backend_kind = match resolved_gpu_genotype_format {
        g_plan::GpuGenotypeFormat::Dosage => "jax_dosage",
        g_plan::GpuGenotypeFormat::Packed8 => "jax_packed8",
        g_plan::GpuGenotypeFormat::Auto => return Err(CoordinatedRunError::UnresolvedGpuGenotypeFormat),
    };
    telemetry_session.emit_execution_plan_prepared_event(
        thread_name,
        association_mode.as_str(),
        trait_type.as_str(),
        phenotype_count,
        i64::from(chunk_size),
        variant_limit.map(i64::from),
        device.as_str(),
    )?;
    g_runtime::emit_run_diagnostic_event(&g_runtime::build_runner_execution_plan_prepared_diagnostic_payload(
        association_mode.as_str(),
        phenotype_count,
        i64::from(chunk_size),
        variant_limit.map(i64::from),
        device.as_str(),
    ))?;
    telemetry_session.emit_association_backend_selected_event(
        thread_name,
        association_mode.as_str(),
        association_backend_kind,
        device.as_str(),
        resolved_gpu_genotype_format.as_str(),
        single_phenotype_name.as_deref(),
        (phenotype_count > 1).then_some(phenotype_count),
    )?;
    record_stage_duration(stage_timing_recorder.as_deref_mut(), "native_run_preparation", preparation_start_time);

    g_runtime::emit_run_diagnostic_event(&g_runtime::build_runner_execution_plan_dispatch_started_diagnostic_payload(
        phenotype_count,
        association_mode.as_str(),
    ))?;
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
) -> Result<(), CoordinatedRunError<BackendError, HookError>> {
    for (group_index, report) in delivery_reports.iter().enumerate() {
        let processed_chunk_count = i64::try_from(report.processed_chunk_count)
            .map_err(|_| CoordinatedRunError::ProcessedChunkCountOutOfRange)?;
        g_runtime::emit_run_diagnostic_event(&g_runtime::build_native_dispatch_delivery_finished_diagnostic_payload(
            &format!("association_group_{group_index}"),
            processed_chunk_count,
        ))?;
        for warning in &report.warnings {
            let nonconverged_count = u64::try_from(warning.nonconverged_count)
                .map_err(|_| CoordinatedRunError::AssociationWarningCountOutOfRange)?;
            let total_fit_count = u64::try_from(warning.total_fit_count)
                .map_err(|_| CoordinatedRunError::AssociationWarningCountOutOfRange)?;
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
) -> Result<Vec<PhenotypeRunArtifacts>, CoordinatedRunError<BackendError, HookError>> {
    let artifacts = completed_outputs
        .into_iter()
        .map(|run| PhenotypeRunArtifacts {
            output_run_directory: run.run_directory.display().to_string(),
            parquet_dataset_directory: run.parts_directory.display().to_string(),
        })
        .collect::<Vec<_>>();
    let parquet_dataset_paths =
        artifacts.iter().map(|artifact| artifact.parquet_dataset_directory.as_str()).collect::<Vec<_>>();
    if phenotype_count == 1 {
        let phenotype_name = single_phenotype_name.ok_or(CoordinatedRunError::MissingPhenotypeOutput)?;
        let artifact = artifacts.first().ok_or(CoordinatedRunError::MissingPhenotypeOutput)?;
        telemetry_session.emit_phenotype_writer_finished_event(
            thread_name,
            association_mode.as_str(),
            phenotype_name,
            &artifact.parquet_dataset_directory,
        )?;
    } else {
        telemetry_session.emit_multi_phenotype_writer_finished_event(
            thread_name,
            association_mode.as_str(),
            phenotype_count,
            &parquet_dataset_paths,
        )?;
    }
    g_runtime::emit_run_diagnostic_event(&g_runtime::build_runner_metadata_artifacts_completed_diagnostic_payload(
        association_mode.as_str(),
        phenotype_count,
    ))?;
    Ok(artifacts)
}

fn record_stage_duration(recorder: Option<&mut StageTimingRecorder>, stage_name: &str, start_time: Instant) {
    if let Some(recorder) = recorder {
        recorder.add_stage_duration(stage_name.to_string(), start_time.elapsed().as_secs_f64());
    }
}
