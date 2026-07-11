//! Prepared native association-run ownership.

use std::collections::BTreeSet;
use std::path::Path;
use std::sync::Arc;

use g_input::{AlignedPhenotypeGroup, PhenotypeGroupLoadRequest, SampleIdentifierData};
use g_output::{CompletedOutputRun, ManifestFileFingerprintCache, OutputDeliveryState, OutputManager};
use g_plan::{GpuGenotypeFormat, RunPlan};

use crate::backend::AssociationBackend;
use crate::delivery::{
    AssociationDeliveryRequest, AssociationDeliverySettings, GroupedUnionAssociationDeliveryRequest,
};
use crate::delivery_execution::{
    AssociationDeliveryReport, DeliveryError, run_association_delivery, run_grouped_union_association_delivery,
};
use crate::output_manifest::build_prediction_loco_file_fingerprints_with_cache;
use crate::pipeline::BgenRunEngine;
use crate::preflight::{PreflightError, validate_jax_index_capacity, validate_multi_trait_preflight_values};
use crate::preparation::{
    PipelineOutputPreparationError, RuntimeOutputGroupInput, RuntimeOutputPlan, build_runtime_output_initializations,
};
use crate::progress::{ProgressTotals, RunProgressReporter};
use crate::trusted_validation::TrustedBgenValidationError;

/// Failure while converting a run plan into a fully prepared native run.
#[derive(Debug, thiserror::Error)]
pub(crate) enum RunPreparationError {
    #[error(transparent)]
    Bgen(#[from] g_genotype::BgenError),
    #[error(transparent)]
    Input(#[from] g_input::InputError),
    #[error(transparent)]
    Output(#[from] g_output::OutputError),
    #[error(transparent)]
    OutputPreparation(#[from] PipelineOutputPreparationError),
    #[error(transparent)]
    Preflight(#[from] PreflightError),
    #[error(transparent)]
    TrustedValidation(#[from] TrustedBgenValidationError),
    #[error("The run plan contains no phenotype outputs.")]
    EmptyPhenotypePlan,
    #[error("Aligned input produced no phenotype groups.")]
    EmptyPhenotypeGroups,
    #[error("Resolved GPU genotype format cannot remain auto during run preparation.")]
    UnresolvedGpuGenotypeFormat,
    #[error("{field_name} must be positive.")]
    NonPositiveCapacity { field_name: &'static str },
    #[error("{field_name} overflowed the native usize representation.")]
    CapacityOverflow { field_name: &'static str },
    #[error("{field_name} exceeds the JAX int32 domain.")]
    JaxIntegerOverflow { field_name: &'static str },
}

/// Binding-owned hooks used at the native run boundary.
pub trait RunHooks {
    type Error: std::error::Error + Send + Sync + 'static;

    /// Check whether execution must stop.
    ///
    /// # Errors
    ///
    /// Returns the binding-specific interruption error.
    fn check_interruption(&mut self) -> Result<(), Self::Error>;

    /// Return the process signal name associated with an interruption error.
    fn interruption_signal_name(error: &Self::Error) -> Option<&str>;
}

/// Completed native execution and its output artifacts.
pub struct RunExecution {
    pub completed_outputs: Vec<CompletedOutputRun>,
    pub delivery_reports: Vec<AssociationDeliveryReport>,
}

/// Failure while executing a fully prepared run.
#[derive(Debug, thiserror::Error)]
pub enum RunExecutionError<BackendError, HookError> {
    #[error("Association delivery failed.")]
    Delivery(#[source] DeliveryError<BackendError, HookError>),
    #[error("Association delivery was interrupted.")]
    Interrupted(#[source] HookError),
    #[error("Association delivery was interrupted and queued output could not be flushed.")]
    InterruptedOutputFlush {
        interruption: HookError,
        #[source]
        output: g_output::OutputError,
    },
    #[error("Association delivery failed and output abort also failed.")]
    DeliveryAbort {
        delivery: Box<DeliveryError<BackendError, HookError>>,
        #[source]
        output: g_output::OutputError,
    },
    #[error("Output completion failed.")]
    OutputFinish(#[source] g_output::OutputError),
}

/// Unprepared native run owning its canonical plan and output lifecycle.
pub struct RunEngine {
    run_plan: Arc<RunPlan>,
    output_manager: OutputManager,
}

impl RunEngine {
    /// Open the output lifecycle for a canonical run plan.
    ///
    /// # Errors
    ///
    /// Returns an error when planned output paths cannot be opened.
    pub fn open(run_plan: RunPlan, effective_config_toml: String) -> Result<Self, RunPreparationError> {
        if run_plan.phenotype_runs.is_empty() {
            return Err(RunPreparationError::EmptyPhenotypePlan);
        }
        validate_jax_integer_domain(&run_plan)?;
        let decode_tile_variant_count = usize::try_from(run_plan.compute.bgen_decode_tile_variant_count)
            .map_err(|_| RunPreparationError::CapacityOverflow { field_name: "BGEN decode tile variant count" })?;
        g_genotype::set_bgen_decode_tile_variant_count(decode_tile_variant_count)?;
        let run_plan = Arc::new(run_plan);
        let output_manager = OutputManager::open(Arc::clone(&run_plan), effective_config_toml)?;
        Ok(Self { run_plan, output_manager })
    }

    /// Resolve genotype policy and prepare all native input and output state.
    ///
    /// # Errors
    ///
    /// Returns a typed error when BGEN preparation, input alignment, preflight,
    /// manifest construction, resume validation, or output initialization fails.
    pub fn prepare(self) -> Result<PreparedRun, RunPreparationError> {
        let Self { run_plan, mut output_manager } = self;
        let chunk_size = positive_u32_as_usize(run_plan.analysis.chunk_size, "analysis chunk size")?;
        let variant_limit = run_plan
            .compute
            .variant_limit
            .map(usize::try_from)
            .transpose()
            .map_err(|_| RunPreparationError::CapacityOverflow { field_name: "variant limit" })?;
        let (resolved_gpu_genotype_format, prepared_bgen_engine) =
            resolve_gpu_genotype_format(&run_plan, &output_manager, chunk_size, variant_limit)?;
        if resolved_gpu_genotype_format == GpuGenotypeFormat::Auto {
            return Err(RunPreparationError::UnresolvedGpuGenotypeFormat);
        }
        let effective_trusted_no_missing_diploid =
            run_plan.compute.trusted_no_missing_diploid || resolved_gpu_genotype_format == GpuGenotypeFormat::Packed8;
        let bgen_engine = match prepared_bgen_engine {
            Some(engine) => engine,
            None => open_bgen_engine(&run_plan, chunk_size, variant_limit, effective_trusted_no_missing_diploid)?,
        };
        validate_scanned_variants(&bgen_engine, variant_limit)?;
        let phenotype_names = run_plan
            .phenotype_runs
            .iter()
            .map(|phenotype_run| phenotype_run.phenotype_name.clone())
            .collect::<Vec<_>>();
        let prediction_loco_paths =
            g_input::resolve_prediction_loco_paths(Path::new(&run_plan.input.prediction_list_path), &phenotype_names)?;
        let groups = load_groups(&run_plan, &bgen_engine, &phenotype_names, &prediction_loco_paths)?;
        validate_groups(&run_plan, &groups)?;
        let prepared_groups = prepare_output_groups(
            &run_plan,
            groups,
            &prediction_loco_paths,
            &mut output_manager,
            resolved_gpu_genotype_format,
            effective_trusted_no_missing_diploid,
            bgen_engine.reader.variant_count(),
        )?;
        let staging_depth = positive_u32_as_usize(run_plan.compute.staging_depth, "association staging depth")?;
        let result_in_flight_limit = run_plan
            .compute
            .result_in_flight_limit
            .map(|value| positive_u32_as_usize(value, "association result in-flight limit"))
            .transpose()?
            .map_or_else(
                || {
                    staging_depth.checked_add(1).ok_or(RunPreparationError::CapacityOverflow {
                        field_name: "association result in-flight limit",
                    })
                },
                Ok,
            )?;
        Ok(PreparedRun {
            run_plan,
            resolved_gpu_genotype_format,
            effective_trusted_no_missing_diploid,
            bgen_engine,
            groups: prepared_groups,
            output_manager,
            staging_depth,
            result_in_flight_limit,
        })
    }
}

struct PreparedAssociationGroup {
    group: AlignedPhenotypeGroup,
    output: OutputDeliveryState,
}

/// Fully prepared run awaiting one association backend.
pub struct PreparedRun {
    run_plan: Arc<RunPlan>,
    resolved_gpu_genotype_format: GpuGenotypeFormat,
    effective_trusted_no_missing_diploid: bool,
    bgen_engine: BgenRunEngine,
    groups: Vec<PreparedAssociationGroup>,
    output_manager: OutputManager,
    staging_depth: usize,
    result_in_flight_limit: usize,
}

impl PreparedRun {
    /// Return the resolved association backend contract.
    #[must_use]
    pub const fn resolved_gpu_genotype_format(&self) -> GpuGenotypeFormat {
        self.resolved_gpu_genotype_format
    }

    /// Execute every prepared group with optional throttled progress reporting.
    ///
    /// # Errors
    ///
    /// Returns a typed backend, hook, delivery, or output completion error.
    pub fn execute_with_progress<Backend, Hooks>(
        self,
        backend: Arc<Backend>,
        hooks: &mut Hooks,
        progress_reporter: Option<&Arc<RunProgressReporter>>,
    ) -> Result<RunExecution, RunExecutionError<Backend::Error, Hooks::Error>>
    where
        Backend: AssociationBackend + 'static,
        Backend::ChromosomeState: 'static,
        Backend::DeviceResult: 'static,
        Hooks: RunHooks,
    {
        let PreparedRun {
            run_plan,
            resolved_gpu_genotype_format,
            effective_trusted_no_missing_diploid,
            bgen_engine,
            groups,
            output_manager,
            staging_depth,
            result_in_flight_limit,
        } = self;
        let grouped_union_sample_indices =
            grouped_union_sample_indices(&groups, resolved_gpu_genotype_format, effective_trusted_no_missing_diploid);
        let statistics_policy = match run_plan.association_mode {
            g_plan::AssociationMode::Regenie2Linear => g_genotype::ChunkStatisticsPolicy {
                retain_imputed_dosage_square_sum: true,
                collect_sparse_candidate_mask: false,
            },
            g_plan::AssociationMode::Regenie2Binary => g_genotype::ChunkStatisticsPolicy {
                retain_imputed_dosage_square_sum: false,
                collect_sparse_candidate_mask: run_plan.correction.method
                    == g_plan::BinaryFallbackMethod::FirthApproximate,
            },
        };
        let delivery_result: Result<Vec<AssociationDeliveryReport>, DeliveryError<Backend::Error, Hooks::Error>> =
            (|| {
                let progress_context = if let Some(reporter) = progress_reporter {
                    let all_chunk_specs = bgen_engine.plan_chunks(&BTreeSet::new())?;
                    Some((reporter, ProgressTotals::from_chunk_specs(&all_chunk_specs)?))
                } else {
                    None
                };
                let requests = groups
                    .into_iter()
                    .map(|prepared_group| {
                        let progress = progress_context
                            .map(|(reporter, totals)| {
                                reporter.register_delivery(
                                    prepared_group.group.phenotype_group.phenotype_names.join(","),
                                    totals,
                                )
                            })
                            .transpose()?;
                        Ok(AssociationDeliveryRequest {
                            group: prepared_group.group,
                            settings: AssociationDeliverySettings {
                                writer_sessions: prepared_group.output.writer_sessions,
                                committed_chunk_identifier_sets: prepared_group.output.committed_chunk_identifier_sets,
                                null_logistic_nonconvergence_policy: run_plan
                                    .compute
                                    .kernels
                                    .binary_null
                                    .nonconvergence_policy,
                                staging_depth,
                                result_in_flight_limit,
                                progress,
                                use_packed8: resolved_gpu_genotype_format == GpuGenotypeFormat::Packed8,
                                statistics_policy,
                            },
                        })
                    })
                    .collect::<Result<Vec<_>, DeliveryError<Backend::Error, Hooks::Error>>>()?;
                if let Some(union_sample_indices) = grouped_union_sample_indices {
                    run_grouped_union_association_delivery(
                        &bgen_engine,
                        &backend,
                        GroupedUnionAssociationDeliveryRequest { groups: requests, union_sample_indices },
                        || hooks.check_interruption(),
                    )
                    .map(|report| vec![report])
                } else {
                    let mut reports = Vec::with_capacity(requests.len());
                    let mut delivery_error = None;
                    for request in requests {
                        match run_association_delivery(&bgen_engine, &backend, request, || hooks.check_interruption()) {
                            Ok(report) => reports.push(report),
                            Err(error) => {
                                delivery_error = Some(error);
                                break;
                            }
                        }
                    }
                    delivery_error.map_or(Ok(reports), Err)
                }
            })();
        drop(backend);
        finish_execution::<Backend::Error, Hooks>(delivery_result, output_manager)
    }
}

fn positive_u32_as_usize(value: u32, field_name: &'static str) -> Result<usize, RunPreparationError> {
    if value == 0 {
        return Err(RunPreparationError::NonPositiveCapacity { field_name });
    }
    usize::try_from(value).map_err(|_| RunPreparationError::CapacityOverflow { field_name })
}

/// Validate every host-plan integer consumed by JAX.
///
/// # Errors
///
/// Returns an error when a required count is zero or exceeds the signed int32
/// domain used by JAX indices, reductions, and loop state.
pub(crate) fn validate_jax_integer_domain(run_plan: &RunPlan) -> Result<(), RunPreparationError> {
    let kernels = &run_plan.compute.kernels;
    let positive_values = [
        ("analysis chunk size", run_plan.analysis.chunk_size),
        ("binary null maximum iterations", kernels.binary_null.maximum_iterations),
        ("Firth batch size", kernels.firth.batch_size),
        ("Firth candidate capacity", kernels.firth.candidate_capacity),
        ("Firth maximum iterations", kernels.firth.maximum_iterations),
        ("Firth pseudo maximum iterations", kernels.firth.pseudo_maximum_iterations),
        ("Firth pseudo inner maximum iterations", kernels.firth.pseudo_inner_maximum_iterations),
        ("Firth Newton-Raphson zero-start iterations", kernels.firth.newton_raphson_zero_start_iterations),
        ("Firth line-search maximum attempts", kernels.firth.line_search_maximum_attempts),
        ("Firth step-halving maximum attempts", kernels.firth.step_halving_maximum_attempts),
        ("null Firth maximum iterations", kernels.null_firth.maximum_iterations),
        ("null Firth fallback iteration multiplier", kernels.null_firth.fallback_iteration_multiplier),
        ("null Firth line-search maximum attempts", kernels.null_firth.line_search_maximum_attempts),
    ];
    for (field_name, value) in positive_values {
        if value == 0 {
            return Err(RunPreparationError::NonPositiveCapacity { field_name });
        }
        i32::try_from(value).map_err(|_| RunPreparationError::JaxIntegerOverflow { field_name })?;
    }
    let fallback_iteration_limit = u64::from(kernels.null_firth.maximum_iterations)
        .checked_mul(u64::from(kernels.null_firth.fallback_iteration_multiplier))
        .ok_or(RunPreparationError::JaxIntegerOverflow { field_name: "null Firth fallback iteration limit" })?;
    let maximum_jax_integer = u64::from(i32::MAX.unsigned_abs());
    if fallback_iteration_limit > maximum_jax_integer {
        return Err(RunPreparationError::JaxIntegerOverflow { field_name: "null Firth fallback iteration limit" });
    }
    Ok(())
}

fn open_bgen_engine(
    run_plan: &RunPlan,
    chunk_size: usize,
    variant_limit: Option<usize>,
    trusted_no_missing_diploid: bool,
) -> Result<BgenRunEngine, RunPreparationError> {
    let engine = BgenRunEngine::open(
        Path::new(&run_plan.input.bgen_path),
        chunk_size,
        variant_limit,
        trusted_no_missing_diploid,
    )?;
    if trusted_no_missing_diploid {
        let cache_directory = crate::trusted_validation::default_cache_directory()?;
        crate::trusted_validation::validate_trusted_no_missing_diploid_with_cache_directory(
            &engine.reader,
            Path::new(&run_plan.input.bgen_path),
            run_plan.compute.trusted_bgen_validation_mode,
            &cache_directory,
        )?;
    }
    Ok(engine)
}

fn resolve_gpu_genotype_format(
    run_plan: &RunPlan,
    output_manager: &OutputManager,
    chunk_size: usize,
    variant_limit: Option<usize>,
) -> Result<(GpuGenotypeFormat, Option<BgenRunEngine>), RunPreparationError> {
    if run_plan.compute.requested_gpu_genotype_format != GpuGenotypeFormat::Auto {
        return Ok((run_plan.compute.requested_gpu_genotype_format, None));
    }
    let single_binary =
        run_plan.phenotype_runs.len() == 1 && run_plan.analysis.trait_type == g_plan::RegenieTraitType::Binary;
    if !single_binary {
        return Ok((GpuGenotypeFormat::Dosage, None));
    }
    if let Some(manifest_format) = resumed_manifest_gpu_genotype_format(run_plan, output_manager)? {
        return Ok((manifest_format, None));
    }
    if run_plan.compute.device != g_plan::Device::Gpu {
        return Ok((GpuGenotypeFormat::Dosage, None));
    }
    match open_bgen_engine(run_plan, chunk_size, variant_limit, true) {
        Ok(engine) => Ok((GpuGenotypeFormat::Packed8, Some(engine))),
        Err(_) => Ok((GpuGenotypeFormat::Dosage, None)),
    }
}

fn resumed_manifest_gpu_genotype_format(
    run_plan: &RunPlan,
    output_manager: &OutputManager,
) -> Result<Option<GpuGenotypeFormat>, RunPreparationError> {
    if !run_plan.output.resume {
        return Ok(None);
    }
    let phenotype_name =
        &run_plan.phenotype_runs.first().ok_or(RunPreparationError::EmptyPhenotypePlan)?.phenotype_name;
    output_manager.existing_manifest_gpu_genotype_format(phenotype_name).map_err(Into::into)
}

fn validate_scanned_variants(bgen_engine: &BgenRunEngine, variant_limit: Option<usize>) -> Result<(), PreflightError> {
    let variant_count = bgen_engine.reader.variant_count();
    if variant_count == 0 {
        return Err(PreflightError::EmptyBgenInput);
    }
    if variant_limit.is_some_and(|limit| limit.min(variant_count) == 0) {
        return Err(PreflightError::EmptyBgenScan);
    }
    Ok(())
}

fn load_groups(
    run_plan: &RunPlan,
    bgen_engine: &BgenRunEngine,
    phenotype_names: &[String],
    prediction_loco_paths: &[g_input::PredictionLocoPath],
) -> Result<Vec<AlignedPhenotypeGroup>, RunPreparationError> {
    let sample_identifiers = load_sample_identifiers(run_plan, bgen_engine)?;
    let groups = g_input::load_aligned_phenotype_groups(&PhenotypeGroupLoadRequest {
        sample_identifiers: &sample_identifiers,
        phenotype_path: &run_plan.input.phenotype_path,
        prediction_list_path: &run_plan.input.prediction_list_path,
        prediction_loco_paths,
        phenotype_names,
        covariate_path: run_plan.input.covariate_path.as_deref(),
        covariate_names: Some(&run_plan.input.covariate_names),
        is_binary_trait: run_plan.analysis.trait_type == g_plan::RegenieTraitType::Binary,
        sample_key_mode: run_plan.input.sample_key_mode,
        sample_mode: run_plan.compute.multi_phenotype_sample_mode,
    })?;
    if groups.is_empty() {
        return Err(RunPreparationError::EmptyPhenotypeGroups);
    }
    Ok(groups)
}

fn load_sample_identifiers(
    run_plan: &RunPlan,
    bgen_engine: &BgenRunEngine,
) -> Result<SampleIdentifierData, RunPreparationError> {
    Ok(g_input::load_sample_identifier_data_from_sample_file(
        Path::new(&run_plan.input.sample_path),
        bgen_engine.reader.sample_count(),
    )?)
}

fn validate_groups(run_plan: &RunPlan, groups: &[AlignedPhenotypeGroup]) -> Result<(), RunPreparationError> {
    let is_binary_trait = run_plan.analysis.trait_type == g_plan::RegenieTraitType::Binary;
    let chunk_size = positive_u32_as_usize(run_plan.analysis.chunk_size, "analysis chunk size")?;
    let firth_candidate_capacity =
        positive_u32_as_usize(run_plan.compute.kernels.firth.candidate_capacity, "Firth candidate capacity")?;
    let firth_batch_size = positive_u32_as_usize(run_plan.compute.kernels.firth.batch_size, "Firth batch size")?;
    for group in groups {
        let trait_count = group.phenotype_group.phenotype_names.len();
        let sample_count = group.sample_indices.len();
        validate_jax_index_capacity(
            trait_count,
            sample_count,
            chunk_size,
            firth_candidate_capacity,
            firth_batch_size,
            is_binary_trait,
        )?;
        validate_multi_trait_preflight_values(
            trait_count,
            sample_count,
            &group.phenotype_values,
            sample_count,
            group.covariate_names.len(),
            &group.covariate_values,
            is_binary_trait,
        )?;
    }
    Ok(())
}

fn prepare_output_groups(
    run_plan: &RunPlan,
    groups: Vec<AlignedPhenotypeGroup>,
    prediction_loco_paths: &[g_input::PredictionLocoPath],
    output_manager: &mut OutputManager,
    resolved_gpu_genotype_format: GpuGenotypeFormat,
    effective_trusted_no_missing_diploid: bool,
    variant_count: usize,
) -> Result<Vec<PreparedAssociationGroup>, RunPreparationError> {
    let runtime_output_plan =
        RuntimeOutputPlan { variant_count, effective_trusted_no_missing_diploid, resolved_gpu_genotype_format };
    let mut fingerprint_cache = ManifestFileFingerprintCache::default();
    let all_prediction_loco_files: Arc<[g_output::PredictionLocoFileFingerprint]> =
        build_prediction_loco_file_fingerprints_with_cache(prediction_loco_paths, &mut fingerprint_cache)?.into();
    let mut run_initializations = Vec::with_capacity(run_plan.phenotype_runs.len());
    for group in &groups {
        run_initializations.extend(build_runtime_output_initializations(
            &RuntimeOutputGroupInput {
                phenotype_group: &group.phenotype_group,
                covariate_names: &group.covariate_names,
                sample_count: group.sample_indices.len(),
                output_sample_mode: group.phenotype_group.sample_mode,
            },
            &runtime_output_plan,
            &all_prediction_loco_files,
        )?);
    }
    let collect_stage_timings = run_plan.diagnostics.stage_timings_path.is_some()
        || run_plan.diagnostics.profile_summary_path.is_some()
        || matches!(run_plan.diagnostics.telemetry, g_plan::TelemetryMode::Profile);
    output_manager.initialize(run_initializations, collect_stage_timings)?;
    groups
        .into_iter()
        .map(|group| {
            let output = output_manager.delivery_state_for_phenotypes(&group.phenotype_group.phenotype_names)?;
            Ok(PreparedAssociationGroup { group, output })
        })
        .collect()
}

fn grouped_union_sample_indices(
    groups: &[PreparedAssociationGroup],
    genotype_format: GpuGenotypeFormat,
    effective_trusted_no_missing_diploid: bool,
) -> Option<Vec<usize>> {
    if groups.len() <= 1 || genotype_format == GpuGenotypeFormat::Packed8 || !effective_trusted_no_missing_diploid {
        return None;
    }
    let union_sample_indices =
        g_input::build_union_sample_indices(groups.iter().map(|group| group.group.sample_indices.as_slice()));
    let grouped_sample_count = groups.iter().map(|group| group.group.sample_indices.len()).sum::<usize>();
    (union_sample_indices.len() < grouped_sample_count).then_some(union_sample_indices)
}

fn finish_execution<BackendError, Hooks>(
    delivery_result: Result<Vec<AssociationDeliveryReport>, DeliveryError<BackendError, Hooks::Error>>,
    output_manager: OutputManager,
) -> Result<RunExecution, RunExecutionError<BackendError, Hooks::Error>>
where
    Hooks: RunHooks,
{
    match delivery_result {
        Ok(delivery_reports) => output_manager
            .finish()
            .map(|completed_outputs| RunExecution { completed_outputs, delivery_reports })
            .map_err(RunExecutionError::OutputFinish),
        Err(DeliveryError::Interrupted(interruption)) => {
            let signal_name = Hooks::interruption_signal_name(&interruption).unwrap_or("unknown");
            match output_manager.finish_interrupted(signal_name) {
                Ok(()) => Err(RunExecutionError::Interrupted(interruption)),
                Err(output) => Err(RunExecutionError::InterruptedOutputFlush { interruption, output }),
            }
        }
        Err(delivery) => match output_manager.abort() {
            Ok(()) => Err(RunExecutionError::Delivery(delivery)),
            Err(output) => Err(RunExecutionError::DeliveryAbort { delivery: Box::new(delivery), output }),
        },
    }
}
