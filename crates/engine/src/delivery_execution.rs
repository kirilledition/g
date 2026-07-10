//! Association delivery execution independent of Python.

use std::collections::BTreeSet;
use std::sync::Arc;

use g_genotype::{BgenError, GenotypeError};
use g_input::{InputError, PredictionError};
use g_output::OutputError;

use crate::association_scheduler::{
    AssociationBatchPipeline, CompletedAssociationBatch, OwnedGenotypeBuffer, ScheduledAssociationBatch, SchedulerError,
};
use crate::backend::{
    AssociationBackend, GroupPreparationInput, SampleMajorCovariateMatrixView, TraitMajorPhenotypeMatrixView,
    TraitMajorPredictionMatrixView,
};
use crate::delivery::{
    AssociationDeliveryRequest, AssociationDeliverySettings, GroupedUnionAssociationDeliveryRequest,
};
use crate::genotype_buffer::{
    GenotypeBufferPool, decode_genotype_buffer, homogeneous_chunk_chromosome, project_variant_major_dosages,
};
use crate::null_logistic_policy::{
    NullLogisticNonconvergenceAction, NullLogisticPolicyError, plan_null_logistic_nonconvergence,
};
use crate::output_schedule::{active_trait_indices_for_chunk, intersect_committed_chunk_identifier_sets};
use crate::output_write::write_host_association_batch;
use crate::pipeline::BgenRunEngine;
use crate::progress::RunProgressError;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DeliveryWarning {
    pub chromosome: String,
    pub message: String,
    pub nonconverged_count: usize,
    pub total_fit_count: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AssociationDeliveryReport {
    pub processed_chunk_count: usize,
    pub warnings: Vec<DeliveryWarning>,
}

#[derive(Debug, thiserror::Error)]
pub enum DeliveryError<BackendError, InterruptionError> {
    #[error("association backend failed during {stage}: {source}")]
    Backend {
        stage: &'static str,
        #[source]
        source: BackendError,
    },
    #[error(transparent)]
    Bgen(#[from] BgenError),
    #[error(transparent)]
    Genotype(#[from] GenotypeError),
    #[error(transparent)]
    Prediction(#[from] PredictionError),
    #[error(transparent)]
    Input(#[from] InputError),
    #[error(transparent)]
    Output(#[from] OutputError),
    #[error(transparent)]
    Progress(#[from] RunProgressError),
    #[error(transparent)]
    NullLogisticPolicy(#[from] NullLogisticPolicyError),
    #[error("association delivery was interrupted: {0}")]
    Interrupted(InterruptionError),
    #[error(transparent)]
    Scheduler(#[from] SchedulerError<BackendError>),
    #[error("invalid association delivery: {0}")]
    InvalidInput(String),
    #[error("binary null logistic model did not converge: {0}")]
    NullLogisticNonconvergence(String),
}

type DeliveryResult<Value, BackendError, InterruptionError> =
    Result<Value, DeliveryError<BackendError, InterruptionError>>;

/// Execute one aligned phenotype group through decode, compute, and output.
///
/// # Errors
///
/// Returns a typed error when input validation, interruption, BGEN delivery,
/// backend execution, scheduling, or output fails.
pub(crate) fn run_association_delivery<Backend, CheckInterruption, InterruptionError>(
    engine: &BgenRunEngine,
    backend: &Arc<Backend>,
    request: &AssociationDeliveryRequest,
    mut check_interruption: CheckInterruption,
) -> DeliveryResult<AssociationDeliveryReport, Backend::Error, InterruptionError>
where
    Backend: AssociationBackend + 'static,
    Backend::ChromosomeState: 'static,
    Backend::DeviceResult: 'static,
    CheckInterruption: FnMut() -> Result<(), InterruptionError>,
{
    validate_delivery_request::<Backend::Error, InterruptionError>(request)?;
    engine.reader.prepare_sample_selection(&request.group.sample_indices)?;
    let delivery_result = run_prepared_association_delivery(engine, backend, request, &mut check_interruption);
    let clear_result = engine.reader.clear_prepared_sample_selection();
    match (delivery_result, clear_result) {
        (Err(error), _) => Err(error),
        (Ok(_), Err(error)) => Err(error.into()),
        (Ok(report), Ok(())) => Ok(report),
    }
}

fn run_prepared_association_delivery<Backend, CheckInterruption, InterruptionError>(
    engine: &BgenRunEngine,
    backend: &Arc<Backend>,
    request: &AssociationDeliveryRequest,
    check_interruption: &mut CheckInterruption,
) -> DeliveryResult<AssociationDeliveryReport, Backend::Error, InterruptionError>
where
    Backend: AssociationBackend + 'static,
    Backend::ChromosomeState: 'static,
    Backend::DeviceResult: 'static,
    CheckInterruption: FnMut() -> Result<(), InterruptionError>,
{
    let group_state = backend
        .prepare_group(group_preparation_input(&request.group))
        .map_err(|source| DeliveryError::Backend { stage: "prepare_group", source })?;
    let committed_chunk_identifiers =
        intersect_committed_chunk_identifier_sets(&request.settings.committed_chunk_identifier_sets);
    let chunk_specs = engine.plan_chunks(&committed_chunk_identifiers)?;
    if let Some(progress) = request.settings.progress.as_ref() {
        let all_chunk_specs = engine.plan_chunks(&BTreeSet::new())?;
        progress.initialize(&all_chunk_specs, &chunk_specs)?;
    }
    let mut buffer_pool = GenotypeBufferPool::default();
    let mut current_chromosome = None;
    let mut pipeline: Option<AssociationBatchPipeline<Backend>> = None;
    let mut processed_chunk_count = 0_usize;
    let mut warnings = Vec::new();

    for chunk_spec in chunk_specs {
        check_interruption().map_err(DeliveryError::Interrupted)?;
        let variant_count = chunk_spec
            .variant_stop_index
            .checked_sub(chunk_spec.variant_start_index)
            .ok_or_else(|| DeliveryError::InvalidInput("BGEN chunk stop precedes its start.".to_string()))?;
        let metadata =
            engine.reader.variant_metadata_slice(chunk_spec.variant_start_index, chunk_spec.variant_stop_index)?;
        let chromosome = homogeneous_chunk_chromosome(&metadata, variant_count)?;
        if current_chromosome.as_deref() != Some(chromosome.as_str()) {
            if let Some(mut previous_pipeline) = pipeline.take() {
                finish_and_drain_pipeline(&mut previous_pipeline, &request.settings, &mut buffer_pool)?;
            }
            pipeline = Some(prepare_chromosome_pipeline(
                backend,
                &group_state,
                &request.group,
                &request.settings,
                &chromosome,
                &mut warnings,
            )?);
            current_chromosome = Some(chromosome);
        }

        let active_trait_indices = active_trait_indices_for_chunk(
            request.settings.writer_sessions.len(),
            chunk_spec.variant_start_index,
            &request.settings.committed_chunk_identifier_sets,
        )
        .map_err(DeliveryError::InvalidInput)?;
        if active_trait_indices.is_empty() {
            return Err(DeliveryError::InvalidInput(
                "planned a chunk already committed by every output writer".to_string(),
            ));
        }
        let sample_count = request.group.sample_indices.len();
        let genotype_value_count = variant_count
            .checked_mul(sample_count)
            .ok_or_else(|| DeliveryError::InvalidInput("genotype batch dimensions overflow usize".to_string()))?;
        let mut genotype_buffer = buffer_pool.acquire(genotype_value_count, request.settings.use_packed8)?;
        let statistics = decode_genotype_buffer(
            &engine.reader,
            chunk_spec.variant_start_index,
            chunk_spec.variant_stop_index,
            &mut genotype_buffer,
        )?;
        let scheduled_batch = ScheduledAssociationBatch {
            variant_start_index: chunk_spec.variant_start_index,
            variant_count,
            sample_count,
            metadata,
            statistics,
            genotype_buffer,
            active_trait_indices,
        };
        let active_pipeline = pipeline
            .as_ref()
            .ok_or_else(|| DeliveryError::InvalidInput("association pipeline was not initialized".to_string()))?;
        active_pipeline.submit(scheduled_batch)?;
        processed_chunk_count += 1;
        drain_available_batches(active_pipeline, &request.settings, &mut buffer_pool)?;
    }

    if let Some(mut final_pipeline) = pipeline {
        finish_and_drain_pipeline(&mut final_pipeline, &request.settings, &mut buffer_pool)?;
    }
    check_interruption().map_err(DeliveryError::Interrupted)?;
    Ok(AssociationDeliveryReport { processed_chunk_count, warnings })
}

fn group_preparation_input(group: &g_input::AlignedPhenotypeGroup) -> GroupPreparationInput<'_> {
    let sample_count = group.sample_indices.len();
    GroupPreparationInput {
        phenotypes: TraitMajorPhenotypeMatrixView {
            values: &group.phenotype_values,
            trait_count: group.phenotype_group.phenotype_names.len(),
            sample_count,
        },
        covariates: SampleMajorCovariateMatrixView {
            values: &group.covariate_values,
            sample_count,
            covariate_count: group.covariate_names.len(),
        },
    }
}

fn prepare_chromosome_pipeline<Backend, InterruptionError>(
    backend: &Arc<Backend>,
    group_state: &Backend::GroupState,
    group: &g_input::AlignedPhenotypeGroup,
    settings: &AssociationDeliverySettings,
    chromosome: &str,
    warnings: &mut Vec<DeliveryWarning>,
) -> DeliveryResult<AssociationBatchPipeline<Backend>, Backend::Error, InterruptionError>
where
    Backend: AssociationBackend + 'static,
    Backend::ChromosomeState: 'static,
    Backend::DeviceResult: 'static,
{
    let predictions = group.chromosome_prediction_matrix(chromosome)?;
    let prepared_chromosome = backend
        .prepare_chromosome(
            group_state,
            TraitMajorPredictionMatrixView {
                values: &predictions.prediction_values,
                trait_count: predictions.trait_count,
                sample_count: predictions.sample_count,
            },
        )
        .map_err(|source| DeliveryError::Backend { stage: "prepare_chromosome", source })?;
    if let Some(logistic_converged) = prepared_chromosome.null_logistic_converged.as_ref() {
        enforce_null_logistic_policy::<Backend::Error, InterruptionError>(
            chromosome,
            logistic_converged,
            &group.phenotype_group.phenotype_names,
            settings.null_logistic_nonconvergence_policy,
            warnings,
        )?;
    }
    Ok(AssociationBatchPipeline::new(
        Arc::clone(backend),
        prepared_chromosome.state,
        settings.staging_depth,
        settings.result_in_flight_limit,
    )?)
}

fn enforce_null_logistic_policy<BackendError, InterruptionError>(
    chromosome: &str,
    convergence_flags: &[bool],
    phenotype_names: &[String],
    policy: g_plan::NullLogisticNonconvergencePolicy,
    warnings: &mut Vec<DeliveryWarning>,
) -> DeliveryResult<(), BackendError, InterruptionError> {
    let plan = plan_null_logistic_nonconvergence(
        chromosome,
        convergence_flags,
        false,
        Some(phenotype_names),
        policy.as_str(),
    )?;
    match plan.action {
        NullLogisticNonconvergenceAction::Continue => Ok(()),
        NullLogisticNonconvergenceAction::Fail => Err(DeliveryError::NullLogisticNonconvergence(
            plan.message.unwrap_or_else(|| "unknown null-logistic failure".to_string()),
        )),
        NullLogisticNonconvergenceAction::Warn => {
            warnings.push(DeliveryWarning {
                chromosome: chromosome.to_string(),
                message: plan.warning_message.unwrap_or_else(|| {
                    "Binary null logistic model did not converge; continuing under warning policy.".to_string()
                }),
                nonconverged_count: plan.nonconverged_count,
                total_fit_count: plan.total_fit_count,
            });
            Ok(())
        }
    }
}

fn validate_delivery_request<BackendError, InterruptionError>(
    request: &AssociationDeliveryRequest,
) -> DeliveryResult<(), BackendError, InterruptionError> {
    let trait_count = request.group.phenotype_group.phenotype_names.len();
    if trait_count == 0 {
        return Err(DeliveryError::InvalidInput("phenotype group contains no traits".to_string()));
    }
    if request.group.sample_indices.is_empty() {
        return Err(DeliveryError::InvalidInput("phenotype group contains no aligned samples".to_string()));
    }
    if request.settings.writer_sessions.len() != trait_count {
        return Err(DeliveryError::InvalidInput(format!(
            "output writer count {} does not match trait count {trait_count}",
            request.settings.writer_sessions.len()
        )));
    }
    if request.settings.committed_chunk_identifier_sets.len() != trait_count {
        return Err(DeliveryError::InvalidInput(format!(
            "committed chunk set count {} does not match trait count {trait_count}",
            request.settings.committed_chunk_identifier_sets.len()
        )));
    }
    if request.settings.staging_depth == 0 || request.settings.result_in_flight_limit == 0 {
        return Err(DeliveryError::InvalidInput("association queue capacities must be positive".to_string()));
    }
    Ok(())
}

fn drain_available_batches<Backend, InterruptionError>(
    pipeline: &AssociationBatchPipeline<Backend>,
    settings: &AssociationDeliverySettings,
    buffer_pool: &mut GenotypeBufferPool,
) -> DeliveryResult<(), Backend::Error, InterruptionError>
where
    Backend: AssociationBackend + 'static,
    Backend::ChromosomeState: 'static,
    Backend::DeviceResult: 'static,
{
    while let Some(completed_batch) = pipeline.try_receive()? {
        write_completed_batch::<Backend::Error, InterruptionError>(completed_batch, settings, buffer_pool)?;
    }
    Ok(())
}

fn finish_and_drain_pipeline<Backend, InterruptionError>(
    pipeline: &mut AssociationBatchPipeline<Backend>,
    settings: &AssociationDeliverySettings,
    buffer_pool: &mut GenotypeBufferPool,
) -> DeliveryResult<(), Backend::Error, InterruptionError>
where
    Backend: AssociationBackend + 'static,
    Backend::ChromosomeState: 'static,
    Backend::DeviceResult: 'static,
{
    pipeline.close_submission();
    while let Some(completed_batch) = pipeline.receive()? {
        write_completed_batch::<Backend::Error, InterruptionError>(completed_batch, settings, buffer_pool)?;
    }
    pipeline.join()?;
    Ok(())
}

fn write_completed_batch<BackendError, InterruptionError>(
    completed_batch: CompletedAssociationBatch,
    settings: &AssociationDeliverySettings,
    buffer_pool: &mut GenotypeBufferPool,
) -> DeliveryResult<(), BackendError, InterruptionError> {
    let CompletedAssociationBatch {
        variant_start_index,
        metadata,
        statistics,
        genotype_buffer,
        active_trait_indices,
        result,
        ..
    } = completed_batch;
    let variant_count = metadata.variant_identifier.len();
    write_host_association_batch(
        &settings.writer_sessions,
        &active_trait_indices,
        variant_start_index,
        metadata,
        statistics,
        result,
    )?;
    if let Some(progress) = settings.progress.as_ref() {
        progress.record_writer_accepted(variant_count)?;
    }
    buffer_pool.release(genotype_buffer);
    Ok(())
}

struct GroupedUnionGroupRuntime<Backend>
where
    Backend: AssociationBackend + 'static,
    Backend::ChromosomeState: 'static,
    Backend::DeviceResult: 'static,
{
    request: AssociationDeliveryRequest,
    group_state: Backend::GroupState,
    sample_positions: Vec<usize>,
    pipeline: Option<AssociationBatchPipeline<Backend>>,
    buffer_pool: GenotypeBufferPool,
}

/// Decode each union-sample chunk once and execute every aligned group.
///
/// # Errors
///
/// Returns a typed error when union validation, interruption, BGEN delivery,
/// backend execution, scheduling, projection, or output fails.
pub(crate) fn run_grouped_union_association_delivery<Backend, CheckInterruption, InterruptionError>(
    engine: &BgenRunEngine,
    backend: &Arc<Backend>,
    request: GroupedUnionAssociationDeliveryRequest,
    mut check_interruption: CheckInterruption,
) -> DeliveryResult<AssociationDeliveryReport, Backend::Error, InterruptionError>
where
    Backend: AssociationBackend + 'static,
    Backend::ChromosomeState: 'static,
    Backend::DeviceResult: 'static,
    CheckInterruption: FnMut() -> Result<(), InterruptionError>,
{
    validate_grouped_union_request::<Backend::Error, InterruptionError>(&request)?;
    engine.reader.prepare_sample_selection(&request.union_sample_indices)?;
    let delivery_result =
        run_prepared_grouped_union_association_delivery(engine, backend, request, &mut check_interruption);
    let clear_result = engine.reader.clear_prepared_sample_selection();
    match (delivery_result, clear_result) {
        (Err(error), _) => Err(error),
        (Ok(_), Err(error)) => Err(error.into()),
        (Ok(report), Ok(())) => Ok(report),
    }
}

fn run_prepared_grouped_union_association_delivery<Backend, CheckInterruption, InterruptionError>(
    engine: &BgenRunEngine,
    backend: &Arc<Backend>,
    request: GroupedUnionAssociationDeliveryRequest,
    check_interruption: &mut CheckInterruption,
) -> DeliveryResult<AssociationDeliveryReport, Backend::Error, InterruptionError>
where
    Backend: AssociationBackend + 'static,
    Backend::ChromosomeState: 'static,
    Backend::DeviceResult: 'static,
    CheckInterruption: FnMut() -> Result<(), InterruptionError>,
{
    let GroupedUnionAssociationDeliveryRequest { groups, union_sample_indices } = request;
    let committed_chunk_identifier_sets = groups
        .iter()
        .flat_map(|group| group.settings.committed_chunk_identifier_sets.iter().cloned())
        .collect::<Vec<_>>();
    let shared_committed_chunk_identifiers =
        intersect_committed_chunk_identifier_sets(&committed_chunk_identifier_sets);
    let chunk_specs = engine.plan_chunks(&shared_committed_chunk_identifiers)?;
    let mut group_runtimes =
        prepare_grouped_union_runtimes::<Backend, InterruptionError>(backend.as_ref(), groups, &union_sample_indices)?;
    let all_chunk_specs = engine.plan_chunks(&BTreeSet::new())?;
    for group_runtime in &group_runtimes {
        let committed_chunk_identifiers =
            intersect_committed_chunk_identifier_sets(&group_runtime.request.settings.committed_chunk_identifier_sets);
        let pending_chunk_specs = engine.plan_chunks(&committed_chunk_identifiers)?;
        if let Some(progress) = group_runtime.request.settings.progress.as_ref() {
            progress.initialize(&all_chunk_specs, &pending_chunk_specs)?;
        }
    }
    let union_sample_count = union_sample_indices.len();
    let mut union_buffer_pool = GenotypeBufferPool::default();
    let mut warnings = Vec::new();

    let delivery_result = (|| {
        let mut current_chromosome = None;
        let mut processed_chunk_count = 0_usize;
        for chunk_spec in chunk_specs {
            check_interruption().map_err(DeliveryError::Interrupted)?;
            let variant_count = chunk_spec
                .variant_stop_index
                .checked_sub(chunk_spec.variant_start_index)
                .ok_or_else(|| DeliveryError::InvalidInput("BGEN chunk stop precedes its start.".to_string()))?;
            let metadata =
                engine.reader.variant_metadata_slice(chunk_spec.variant_start_index, chunk_spec.variant_stop_index)?;
            let chromosome = homogeneous_chunk_chromosome(&metadata, variant_count)?;
            if current_chromosome.as_deref() != Some(chromosome.as_str()) {
                finish_grouped_union_pipelines::<Backend, InterruptionError>(&mut group_runtimes)?;
                start_grouped_union_chromosome_pipelines::<Backend, InterruptionError>(
                    backend,
                    &mut group_runtimes,
                    &chromosome,
                    &mut warnings,
                )?;
                current_chromosome = Some(chromosome);
            }

            let union_value_count = variant_count
                .checked_mul(union_sample_count)
                .ok_or_else(|| DeliveryError::InvalidInput("union genotype dimensions overflow usize".to_string()))?;
            let mut union_buffer = union_buffer_pool.acquire(union_value_count, false)?;
            decode_genotype_buffer(
                &engine.reader,
                chunk_spec.variant_start_index,
                chunk_spec.variant_stop_index,
                &mut union_buffer,
            )?;
            let OwnedGenotypeBuffer::Dosage(union_dosages) = union_buffer else {
                return Err(DeliveryError::InvalidInput(
                    "grouped-union delivery acquired a packed genotype buffer".to_string(),
                ));
            };
            let submission_result = submit_grouped_union_chunk::<Backend, InterruptionError>(
                &mut group_runtimes,
                &union_dosages,
                union_sample_count,
                variant_count,
                chunk_spec.variant_start_index,
                &metadata,
            );
            union_buffer_pool.release(OwnedGenotypeBuffer::Dosage(union_dosages));
            submission_result?;
            processed_chunk_count += 1;
            drain_grouped_union_pipelines::<Backend, InterruptionError>(&mut group_runtimes)?;
        }
        finish_grouped_union_pipelines::<Backend, InterruptionError>(&mut group_runtimes)?;
        check_interruption().map_err(DeliveryError::Interrupted)?;
        Ok(AssociationDeliveryReport { processed_chunk_count, warnings })
    })();

    if delivery_result.is_err() {
        abort_grouped_union_pipelines(&mut group_runtimes);
    }
    delivery_result
}

fn prepare_grouped_union_runtimes<Backend, InterruptionError>(
    backend: &Backend,
    groups: Vec<AssociationDeliveryRequest>,
    union_sample_indices: &[usize],
) -> DeliveryResult<Vec<GroupedUnionGroupRuntime<Backend>>, Backend::Error, InterruptionError>
where
    Backend: AssociationBackend + 'static,
    Backend::ChromosomeState: 'static,
    Backend::DeviceResult: 'static,
{
    groups
        .into_iter()
        .map(|request| {
            let sample_positions =
                g_input::build_group_sample_position_array(union_sample_indices, &request.group.sample_indices)?
                    .into_iter()
                    .map(|position| {
                        usize::try_from(position).map_err(|_| {
                            DeliveryError::InvalidInput("grouped-union sample position must be nonnegative".to_string())
                        })
                    })
                    .collect::<DeliveryResult<Vec<_>, Backend::Error, InterruptionError>>()?;
            let group_state = backend
                .prepare_group(group_preparation_input(&request.group))
                .map_err(|source| DeliveryError::Backend { stage: "prepare_group", source })?;
            Ok(GroupedUnionGroupRuntime {
                request,
                group_state,
                sample_positions,
                pipeline: None,
                buffer_pool: GenotypeBufferPool::default(),
            })
        })
        .collect()
}

fn start_grouped_union_chromosome_pipelines<Backend, InterruptionError>(
    backend: &Arc<Backend>,
    group_runtimes: &mut [GroupedUnionGroupRuntime<Backend>],
    chromosome: &str,
    warnings: &mut Vec<DeliveryWarning>,
) -> DeliveryResult<(), Backend::Error, InterruptionError>
where
    Backend: AssociationBackend + 'static,
    Backend::ChromosomeState: 'static,
    Backend::DeviceResult: 'static,
{
    for group_runtime in group_runtimes {
        group_runtime.pipeline = Some(prepare_chromosome_pipeline(
            backend,
            &group_runtime.group_state,
            &group_runtime.request.group,
            &group_runtime.request.settings,
            chromosome,
            warnings,
        )?);
    }
    Ok(())
}

fn submit_grouped_union_chunk<Backend, InterruptionError>(
    group_runtimes: &mut [GroupedUnionGroupRuntime<Backend>],
    union_dosages: &[f32],
    union_sample_count: usize,
    variant_count: usize,
    variant_start_index: usize,
    metadata: &g_genotype::VariantMetadataColumns,
) -> DeliveryResult<(), Backend::Error, InterruptionError>
where
    Backend: AssociationBackend + 'static,
    Backend::ChromosomeState: 'static,
    Backend::DeviceResult: 'static,
{
    for group_runtime in group_runtimes {
        let settings = &group_runtime.request.settings;
        let active_trait_indices = active_trait_indices_for_chunk(
            settings.writer_sessions.len(),
            variant_start_index,
            &settings.committed_chunk_identifier_sets,
        )
        .map_err(DeliveryError::InvalidInput)?;
        if active_trait_indices.is_empty() {
            continue;
        }
        let sample_count = group_runtime.sample_positions.len();
        let genotype_value_count = variant_count
            .checked_mul(sample_count)
            .ok_or_else(|| DeliveryError::InvalidInput("projected genotype dimensions overflow usize".to_string()))?;
        let genotype_buffer = group_runtime.buffer_pool.acquire(genotype_value_count, false)?;
        let OwnedGenotypeBuffer::Dosage(mut group_dosages) = genotype_buffer else {
            return Err(DeliveryError::InvalidInput(
                "grouped-union delivery acquired a packed group buffer".to_string(),
            ));
        };
        project_variant_major_dosages(
            union_dosages,
            union_sample_count,
            variant_count,
            &group_runtime.sample_positions,
            &mut group_dosages,
        )?;
        let statistics =
            g_genotype::summarize_variant_major_dosage_matrix(&group_dosages, sample_count, variant_count)?;
        let scheduled_batch = ScheduledAssociationBatch {
            variant_start_index,
            variant_count,
            sample_count,
            metadata: metadata.clone(),
            statistics,
            genotype_buffer: OwnedGenotypeBuffer::Dosage(group_dosages),
            active_trait_indices,
        };
        let pipeline = group_runtime
            .pipeline
            .as_ref()
            .ok_or_else(|| DeliveryError::InvalidInput("grouped-union pipeline was not initialized".to_string()))?;
        pipeline.submit(scheduled_batch)?;
    }
    Ok(())
}

fn drain_grouped_union_pipelines<Backend, InterruptionError>(
    group_runtimes: &mut [GroupedUnionGroupRuntime<Backend>],
) -> DeliveryResult<(), Backend::Error, InterruptionError>
where
    Backend: AssociationBackend + 'static,
    Backend::ChromosomeState: 'static,
    Backend::DeviceResult: 'static,
{
    for group_runtime in group_runtimes {
        let pipeline = group_runtime
            .pipeline
            .as_ref()
            .ok_or_else(|| DeliveryError::InvalidInput("grouped-union pipeline was not initialized".to_string()))?;
        drain_available_batches(pipeline, &group_runtime.request.settings, &mut group_runtime.buffer_pool)?;
    }
    Ok(())
}

fn finish_grouped_union_pipelines<Backend, InterruptionError>(
    group_runtimes: &mut [GroupedUnionGroupRuntime<Backend>],
) -> DeliveryResult<(), Backend::Error, InterruptionError>
where
    Backend: AssociationBackend + 'static,
    Backend::ChromosomeState: 'static,
    Backend::DeviceResult: 'static,
{
    for group_runtime in group_runtimes {
        if let Some(mut pipeline) = group_runtime.pipeline.take() {
            finish_and_drain_pipeline(&mut pipeline, &group_runtime.request.settings, &mut group_runtime.buffer_pool)?;
        }
    }
    Ok(())
}

fn abort_grouped_union_pipelines<Backend>(group_runtimes: &mut [GroupedUnionGroupRuntime<Backend>])
where
    Backend: AssociationBackend + 'static,
    Backend::ChromosomeState: 'static,
    Backend::DeviceResult: 'static,
{
    for group_runtime in group_runtimes {
        if let Some(pipeline) = group_runtime.pipeline.as_mut() {
            let _ = pipeline.abort();
        }
    }
}

fn validate_grouped_union_request<BackendError, InterruptionError>(
    request: &GroupedUnionAssociationDeliveryRequest,
) -> DeliveryResult<(), BackendError, InterruptionError> {
    if request.groups.is_empty() {
        return Err(DeliveryError::InvalidInput(
            "grouped-union delivery requires at least one phenotype group".to_string(),
        ));
    }
    if request.union_sample_indices.is_empty() {
        return Err(DeliveryError::InvalidInput(
            "grouped-union delivery requires at least one union sample".to_string(),
        ));
    }
    let sample_indices_by_group =
        request.groups.iter().map(|group| group.group.sample_indices.clone()).collect::<Vec<_>>();
    if request.union_sample_indices != g_input::build_union_sample_indices(&sample_indices_by_group) {
        return Err(DeliveryError::InvalidInput(
            "union sample indices do not match the ordered group union".to_string(),
        ));
    }
    for group in &request.groups {
        validate_delivery_request::<BackendError, InterruptionError>(group)?;
        if group.settings.use_packed8 {
            return Err(DeliveryError::InvalidInput(
                "grouped-union delivery supports dosage genotypes only".to_string(),
            ));
        }
    }
    Ok(())
}
