//! Association delivery execution independent of Python.

use std::sync::Arc;

use g_genotype::{BgenError, GenotypeError};
use g_input::{InputError, PredictionError};
use g_output::{NativeVariantMetadataHandle, OutputError};

use crate::association_scheduler::{
    AssociationBatchOutput, AssociationBatchPipeline, CompletedAssociationBatch, ScheduledAssociationBatch,
    SchedulerError,
};
use crate::backend::{
    AssociationBackend, GroupPreparationInput, OwnedGenotypeBuffer, SampleMajorCovariateMatrix, TraitMajorMatrix,
};
use crate::delivery::{
    AssociationDeliveryRequest, AssociationDeliverySettings, GroupedUnionAssociationDeliveryRequest,
};
use crate::genotype_buffer::{
    GenotypeBufferPool, allocate_genotype_buffer, decode_genotype_buffer, homogeneous_chunk_chromosome,
    project_variant_major_dosages,
};
use crate::null_logistic_policy::{
    NullLogisticNonconvergenceAction, NullLogisticPolicyError, plan_null_logistic_nonconvergence,
};
use crate::output_schedule::{
    ActiveTraitSelection, active_trait_selection_for_chunk, intersect_committed_chunk_identifier_sets,
};
use crate::output_write::write_host_association_batch;
use crate::pipeline::BgenRunEngine;
use crate::progress::RunProgressError;

#[derive(Debug, Eq, PartialEq)]
pub(crate) struct DeliveryWarning {
    pub(crate) chromosome: String,
    pub(crate) message: String,
    pub(crate) nonconverged_count: usize,
    pub(crate) total_fit_count: usize,
}

#[derive(Debug, Eq, PartialEq)]
pub(crate) struct AssociationDeliveryReport {
    pub(crate) processed_chunk_count: usize,
    pub(crate) warnings: Vec<DeliveryWarning>,
}

#[derive(Debug, thiserror::Error)]
pub(crate) enum DeliveryError<BackendError, InterruptionError> {
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

struct PlannedAssociationDelivery {
    chunk_specs: Vec<g_genotype::ChunkSpec>,
    chromosome_blocks: Vec<Arc<str>>,
}

/// Execute one aligned phenotype group through decode, compute, and output.
///
/// # Errors
///
/// Returns a typed error when input validation, interruption, BGEN delivery,
/// backend execution, scheduling, or output fails.
pub(crate) fn run_association_delivery<Backend, CheckInterruption, InterruptionError>(
    engine: &BgenRunEngine,
    backend: &Arc<Backend>,
    mut request: AssociationDeliveryRequest,
    mut check_interruption: CheckInterruption,
) -> DeliveryResult<AssociationDeliveryReport, Backend::Error, InterruptionError>
where
    Backend: AssociationBackend + 'static,
    Backend::ChromosomeState: 'static,
    Backend::DeviceResult: 'static,
    CheckInterruption: FnMut() -> Result<(), InterruptionError>,
{
    validate_delivery_request::<Backend::Error, InterruptionError>(&request)?;
    let planned_delivery = plan_association_delivery(engine, &mut request)?;
    if planned_delivery.chunk_specs.is_empty() {
        check_interruption().map_err(DeliveryError::Interrupted)?;
        return Ok(AssociationDeliveryReport { processed_chunk_count: 0, warnings: Vec::new() });
    }
    run_planned_association_delivery(engine, backend, request, planned_delivery.chunk_specs, &mut check_interruption)
}

fn run_planned_association_delivery<Backend, CheckInterruption, InterruptionError>(
    engine: &BgenRunEngine,
    backend: &Arc<Backend>,
    mut request: AssociationDeliveryRequest,
    chunk_specs: Vec<g_genotype::ChunkSpec>,
    check_interruption: &mut CheckInterruption,
) -> DeliveryResult<AssociationDeliveryReport, Backend::Error, InterruptionError>
where
    Backend: AssociationBackend + 'static,
    Backend::ChromosomeState: 'static,
    Backend::DeviceResult: 'static,
    CheckInterruption: FnMut() -> Result<(), InterruptionError>,
{
    engine.reader.prepare_sample_selection(&request.group.sample_indices)?;
    let delivery_result =
        run_prepared_association_delivery(engine, backend, &mut request, chunk_specs, check_interruption);
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
    request: &mut AssociationDeliveryRequest,
    chunk_specs: Vec<g_genotype::ChunkSpec>,
    check_interruption: &mut CheckInterruption,
) -> DeliveryResult<AssociationDeliveryReport, Backend::Error, InterruptionError>
where
    Backend: AssociationBackend + 'static,
    Backend::ChromosomeState: 'static,
    Backend::DeviceResult: 'static,
    CheckInterruption: FnMut() -> Result<(), InterruptionError>,
{
    let group_state = backend
        .prepare_group(group_preparation_input(&mut request.group))
        .map_err(|source| DeliveryError::Backend { stage: "prepare_group", source })?;
    let delivery_result: DeliveryResult<AssociationDeliveryReport, Backend::Error, InterruptionError> = (|| {
        let mut pipeline = AssociationBatchPipeline::new(
            Arc::clone(backend),
            request.settings.staging_depth,
            request.settings.result_in_flight_limit,
        )?;
        let group_index = pipeline.register_group()?;
        let mut current_chromosome = None;
        let mut processed_chunk_count = 0_usize;
        let mut warnings = Vec::new();
        let packed8_compute_variant_count = request
            .settings
            .use_packed8
            .then(|| engine.chunk_size.min(engine.variant_limit.unwrap_or_else(|| engine.reader.variant_count())));

        for chunk_spec in chunk_specs {
            check_interruption().map_err(DeliveryError::Interrupted)?;
            let variant_count = chunk_spec
                .variant_stop_index
                .checked_sub(chunk_spec.variant_start_index)
                .ok_or_else(|| DeliveryError::InvalidInput("BGEN chunk stop precedes its start.".to_string()))?;
            let metadata =
                engine.reader.variant_metadata_slice(chunk_spec.variant_start_index, chunk_spec.variant_stop_index)?;
            let chromosome = homogeneous_chunk_chromosome(&metadata, variant_count)?;
            if current_chromosome.as_deref() != Some(chromosome.as_ref()) {
                let mut write_completed = |completed_batch: CompletedAssociationBatch| {
                    debug_assert_eq!(completed_batch.group_index, group_index);
                    write_completed_batch::<Backend::Error, InterruptionError>(completed_batch, &request.settings)
                };
                drain_pending_batches(&mut pipeline, &mut write_completed)?;
                pipeline.release_chromosome(group_index)?;
                let chromosome_state = prepare_chromosome_state(
                    backend.as_ref(),
                    &group_state,
                    &mut request.group,
                    &request.settings,
                    &chromosome,
                    &mut warnings,
                )?;
                pipeline.prepare_chromosome(group_index, chromosome_state)?;
                current_chromosome = Some(chromosome);
            }

            let active_trait_selection = active_trait_selection_for_chunk(
                request.settings.writer_sessions.len(),
                chunk_spec.variant_start_index,
                &request.settings.committed_chunk_identifier_sets,
            )
            .map_err(DeliveryError::InvalidInput)?;
            if matches!(&active_trait_selection, ActiveTraitSelection::Indices(indices) if indices.is_empty()) {
                return Err(DeliveryError::InvalidInput(
                    "planned a chunk already committed by every output writer".to_string(),
                ));
            }
            let sample_count = request.group.sample_indices.len();
            let compute_variant_count = packed8_compute_variant_count.unwrap_or(variant_count);
            let mut genotype_buffer = allocate_genotype_buffer(
                variant_count,
                compute_variant_count,
                sample_count,
                request.settings.use_packed8,
            )?;
            let statistics = decode_genotype_buffer(
                &engine.reader,
                chunk_spec.variant_start_index,
                chunk_spec.variant_stop_index,
                &mut genotype_buffer,
                request.settings.statistics_policy,
                compute_variant_count,
                sample_count,
            )?;
            let scheduled_batch = ScheduledAssociationBatch {
                variant_start_index: chunk_spec.variant_start_index,
                compute_variant_count,
                sample_count,
                metadata: NativeVariantMetadataHandle::try_new(&metadata)?,
                statistics,
                genotype_buffer,
                active_trait_selection,
            };
            let mut write_completed = |completed_batch: CompletedAssociationBatch| {
                debug_assert_eq!(completed_batch.group_index, group_index);
                write_completed_batch::<Backend::Error, InterruptionError>(completed_batch, &request.settings)
            };
            submit_batch(&mut pipeline, group_index, scheduled_batch, &mut write_completed)?;
            processed_chunk_count += 1;
            drain_available_batches(&mut pipeline, &mut write_completed)?;
        }

        let mut write_completed = |completed_batch: CompletedAssociationBatch| {
            debug_assert_eq!(completed_batch.group_index, group_index);
            write_completed_batch::<Backend::Error, InterruptionError>(completed_batch, &request.settings)
        };
        finish_and_drain_pipeline(&mut pipeline, &mut write_completed)?;
        check_interruption().map_err(DeliveryError::Interrupted)?;
        Ok(AssociationDeliveryReport { processed_chunk_count, warnings })
    })();
    backend.release_group(group_state);
    delivery_result
}

fn plan_association_delivery<BackendError, InterruptionError>(
    engine: &BgenRunEngine,
    request: &mut AssociationDeliveryRequest,
) -> DeliveryResult<PlannedAssociationDelivery, BackendError, InterruptionError> {
    let committed_chunk_identifiers =
        intersect_committed_chunk_identifier_sets(&request.settings.committed_chunk_identifier_sets);
    let chunk_specs = engine.plan_chunks(&committed_chunk_identifiers)?;
    let chromosome_blocks = planned_chromosome_blocks(engine, &chunk_specs)?;
    request.group.plan_prediction_uses(&chromosome_blocks)?;
    if let Some(progress) = request.settings.progress.as_ref() {
        progress.initialize(&chunk_specs)?;
    }
    Ok(PlannedAssociationDelivery { chunk_specs, chromosome_blocks })
}

fn group_preparation_input(group: &mut g_input::AlignedPhenotypeGroup) -> GroupPreparationInput {
    let sample_count = group.sample_indices.len();
    GroupPreparationInput {
        phenotypes: TraitMajorMatrix {
            values: std::mem::take(&mut group.phenotype_values),
            trait_count: group.phenotype_group.phenotype_names.len(),
            sample_count,
        },
        covariates: SampleMajorCovariateMatrix {
            values: std::mem::take(&mut group.covariate_values),
            sample_count,
            covariate_count: group.covariate_names.len(),
        },
    }
}

fn planned_chromosome_blocks(
    engine: &BgenRunEngine,
    chunk_specs: &[g_genotype::ChunkSpec],
) -> Result<Vec<Arc<str>>, BgenError> {
    let mut chromosome_blocks = Vec::new();
    for chunk_spec in chunk_specs {
        if chunk_spec.variant_start_index >= chunk_spec.variant_stop_index {
            return Err(BgenError::Range("Planned association chunk is empty.".to_string()));
        }
        let metadata =
            engine.reader.variant_metadata_slice(chunk_spec.variant_start_index, chunk_spec.variant_start_index + 1)?;
        let chromosome = metadata
            .shared_chromosome(0)
            .ok_or_else(|| BgenError::Range("Planned association chunk has no chromosome label.".to_string()))?;
        if chromosome_blocks.last().is_none_or(|previous: &Arc<str>| previous.as_ref() != chromosome.as_ref()) {
            chromosome_blocks.push(chromosome);
        }
    }
    Ok(chromosome_blocks)
}

fn prepare_chromosome_state<Backend, InterruptionError>(
    backend: &Backend,
    group_state: &Backend::GroupState,
    group: &mut g_input::AlignedPhenotypeGroup,
    settings: &AssociationDeliverySettings,
    chromosome: &str,
    warnings: &mut Vec<DeliveryWarning>,
) -> DeliveryResult<Backend::ChromosomeState, Backend::Error, InterruptionError>
where
    Backend: AssociationBackend + 'static,
    Backend::ChromosomeState: 'static,
    Backend::DeviceResult: 'static,
{
    let predictions = group.take_chromosome_prediction_matrix(chromosome)?;
    let prepared_chromosome = backend
        .prepare_chromosome(
            group_state,
            TraitMajorMatrix {
                values: predictions.prediction_values,
                trait_count: predictions.trait_count,
                sample_count: predictions.sample_count,
            },
        )
        .map_err(|source| DeliveryError::Backend { stage: "prepare_chromosome", source })?;
    if let Some(logistic_converged) = prepared_chromosome.null_logistic_converged.as_ref()
        && let Err(error) = enforce_null_logistic_policy::<Backend::Error, InterruptionError>(
            chromosome,
            logistic_converged,
            &group.phenotype_group.phenotype_names,
            settings.null_logistic_nonconvergence_policy,
            warnings,
        )
    {
        backend.release_chromosome(prepared_chromosome.state);
        return Err(error);
    }
    Ok(prepared_chromosome.state)
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

fn drain_available_batches<Backend, InterruptionError, WriteCompleted>(
    pipeline: &mut AssociationBatchPipeline<Backend>,
    write_completed: &mut WriteCompleted,
) -> DeliveryResult<(), Backend::Error, InterruptionError>
where
    Backend: AssociationBackend + 'static,
    Backend::ChromosomeState: 'static,
    Backend::DeviceResult: 'static,
    WriteCompleted: FnMut(CompletedAssociationBatch) -> DeliveryResult<(), Backend::Error, InterruptionError>,
{
    while let Some(completed_batch) = pipeline.try_receive()? {
        write_completed(completed_batch)?;
    }
    Ok(())
}

fn submit_batch<Backend, InterruptionError, WriteCompleted>(
    pipeline: &mut AssociationBatchPipeline<Backend>,
    group_index: usize,
    scheduled_batch: ScheduledAssociationBatch,
    write_completed: &mut WriteCompleted,
) -> DeliveryResult<(), Backend::Error, InterruptionError>
where
    Backend: AssociationBackend + 'static,
    Backend::ChromosomeState: 'static,
    Backend::DeviceResult: 'static,
    WriteCompleted: FnMut(CompletedAssociationBatch) -> DeliveryResult<(), Backend::Error, InterruptionError>,
{
    let mut pending_batch = scheduled_batch;
    loop {
        match pipeline.try_submit(group_index, pending_batch)? {
            None => return Ok(()),
            Some(returned_batch) => {
                write_completed(pipeline.receive()?)?;
                pending_batch = returned_batch;
            }
        }
    }
}

fn drain_pending_batches<Backend, InterruptionError, WriteCompleted>(
    pipeline: &mut AssociationBatchPipeline<Backend>,
    write_completed: &mut WriteCompleted,
) -> DeliveryResult<(), Backend::Error, InterruptionError>
where
    Backend: AssociationBackend + 'static,
    Backend::ChromosomeState: 'static,
    Backend::DeviceResult: 'static,
    WriteCompleted: FnMut(CompletedAssociationBatch) -> DeliveryResult<(), Backend::Error, InterruptionError>,
{
    while !pipeline.is_drained() {
        write_completed(pipeline.receive()?)?;
    }
    Ok(())
}

fn finish_and_drain_pipeline<Backend, InterruptionError, WriteCompleted>(
    pipeline: &mut AssociationBatchPipeline<Backend>,
    write_completed: &mut WriteCompleted,
) -> DeliveryResult<(), Backend::Error, InterruptionError>
where
    Backend: AssociationBackend + 'static,
    Backend::ChromosomeState: 'static,
    Backend::DeviceResult: 'static,
    WriteCompleted: FnMut(CompletedAssociationBatch) -> DeliveryResult<(), Backend::Error, InterruptionError>,
{
    drain_pending_batches(pipeline, write_completed)?;
    pipeline.release_all_chromosomes()?;
    pipeline.close_submission();
    pipeline.join()?;
    Ok(())
}

fn write_completed_batch<BackendError, InterruptionError>(
    completed_batch: CompletedAssociationBatch,
    settings: &AssociationDeliverySettings,
) -> DeliveryResult<(), BackendError, InterruptionError> {
    let CompletedAssociationBatch { group_index: _, output, result } = completed_batch;
    let AssociationBatchOutput { variant_start_index, metadata, statistics, active_trait_selection } = output;
    let active_trait_indices = match &active_trait_selection {
        ActiveTraitSelection::All => None,
        ActiveTraitSelection::Indices(indices) => Some(indices.as_slice()),
    };
    let variant_count = metadata.row_count();
    write_host_association_batch(
        &settings.writer_sessions,
        active_trait_indices,
        variant_start_index,
        metadata,
        statistics,
        result,
    )?;
    if let Some(progress) = settings.progress.as_ref() {
        progress.record_writer_accepted(variant_count)?;
    }
    Ok(())
}

struct GroupedUnionGroupRuntime<Backend>
where
    Backend: AssociationBackend + 'static,
    Backend::ChromosomeState: 'static,
    Backend::DeviceResult: 'static,
{
    group_index: usize,
    request: AssociationDeliveryRequest,
    group_state: Option<Backend::GroupState>,
    sample_positions: Vec<usize>,
    chromosome_blocks: Vec<Arc<str>>,
    next_chromosome_block_index: usize,
    final_chunk_identifier: usize,
}

struct PendingGroupedUnionGroup {
    request: AssociationDeliveryRequest,
    chunk_specs: Vec<g_genotype::ChunkSpec>,
    chromosome_blocks: Vec<Arc<str>>,
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
    let GroupedUnionAssociationDeliveryRequest { groups, union_sample_indices: _ } = request;
    let pending_groups = plan_grouped_union_requests::<Backend::Error, InterruptionError>(engine, groups)?;
    if pending_groups.is_empty() {
        check_interruption().map_err(DeliveryError::Interrupted)?;
        return Ok(AssociationDeliveryReport { processed_chunk_count: 0, warnings: Vec::new() });
    }
    let active_union_sample_indices = g_input::build_union_sample_indices(
        pending_groups.iter().map(|group| group.request.group.sample_indices.as_slice()),
    );
    let grouped_sample_count = pending_groups.iter().try_fold(0_usize, |sample_count, group| {
        sample_count
            .checked_add(group.request.group.sample_indices.len())
            .ok_or_else(|| DeliveryError::InvalidInput("pending grouped sample count overflowed usize".to_string()))
    })?;
    if pending_groups.len() == 1 || active_union_sample_indices.len() >= grouped_sample_count {
        return run_pending_groups_direct(engine, backend, pending_groups, &mut check_interruption);
    }

    let committed_chunk_identifier_sets = pending_groups
        .iter()
        .flat_map(|group| group.request.settings.committed_chunk_identifier_sets.iter().cloned())
        .collect::<Vec<_>>();
    let shared_committed_chunk_identifiers =
        intersect_committed_chunk_identifier_sets(&committed_chunk_identifier_sets);
    let chunk_specs = engine.plan_chunks(&shared_committed_chunk_identifiers)?;
    engine.reader.prepare_sample_selection(&active_union_sample_indices)?;
    let delivery_result = run_prepared_grouped_union_association_delivery(
        engine,
        backend,
        pending_groups,
        &active_union_sample_indices,
        chunk_specs,
        &mut check_interruption,
    );
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
    pending_groups: Vec<PendingGroupedUnionGroup>,
    union_sample_indices: &[usize],
    chunk_specs: Vec<g_genotype::ChunkSpec>,
    check_interruption: &mut CheckInterruption,
) -> DeliveryResult<AssociationDeliveryReport, Backend::Error, InterruptionError>
where
    Backend: AssociationBackend + 'static,
    Backend::ChromosomeState: 'static,
    Backend::DeviceResult: 'static,
    CheckInterruption: FnMut() -> Result<(), InterruptionError>,
{
    let first_settings =
        &pending_groups.first().expect("grouped execution requires at least two pending groups").request.settings;
    let mut pipeline = AssociationBatchPipeline::new(
        Arc::clone(backend),
        first_settings.staging_depth,
        first_settings.result_in_flight_limit,
    )?;
    let mut group_runtimes = prepare_grouped_union_runtimes::<Backend, InterruptionError>(
        &mut pipeline,
        pending_groups,
        union_sample_indices,
    )?;
    let delivery_result = execute_grouped_union_chunks(
        engine,
        backend.as_ref(),
        &mut pipeline,
        &mut group_runtimes,
        union_sample_indices.len(),
        chunk_specs,
        check_interruption,
    );

    if delivery_result.is_err() {
        let _ = pipeline.abort();
    }
    for group_runtime in &mut group_runtimes {
        if let Some(group_state) = group_runtime.group_state.take() {
            backend.release_group(group_state);
        }
    }
    delivery_result
}

fn execute_grouped_union_chunks<Backend, CheckInterruption, InterruptionError>(
    engine: &BgenRunEngine,
    backend: &Backend,
    pipeline: &mut AssociationBatchPipeline<Backend>,
    group_runtimes: &mut [GroupedUnionGroupRuntime<Backend>],
    union_sample_count: usize,
    chunk_specs: Vec<g_genotype::ChunkSpec>,
    check_interruption: &mut CheckInterruption,
) -> DeliveryResult<AssociationDeliveryReport, Backend::Error, InterruptionError>
where
    Backend: AssociationBackend + 'static,
    Backend::ChromosomeState: 'static,
    Backend::DeviceResult: 'static,
    CheckInterruption: FnMut() -> Result<(), InterruptionError>,
{
    let mut union_buffer_pool = GenotypeBufferPool::default();
    let mut current_chromosome = None;
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
        if current_chromosome.as_deref() != Some(chromosome.as_ref()) {
            {
                let mut write_completed = |completed_batch: CompletedAssociationBatch| {
                    write_grouped_union_completed_batch::<Backend, InterruptionError>(completed_batch, group_runtimes)
                };
                drain_pending_batches(pipeline, &mut write_completed)?;
            }
            for group_runtime in &*group_runtimes {
                let starts_next_chromosome_block = group_runtime
                    .chromosome_blocks
                    .get(group_runtime.next_chromosome_block_index)
                    .is_some_and(|planned_chromosome| planned_chromosome.as_ref() == chromosome.as_ref());
                let completed_before_transition = group_runtime.final_chunk_identifier < chunk_spec.variant_start_index;
                if starts_next_chromosome_block || completed_before_transition {
                    pipeline.release_chromosome(group_runtime.group_index)?;
                }
            }
            prepare_grouped_union_chromosome_states::<Backend, InterruptionError>(
                backend,
                pipeline,
                group_runtimes,
                &chromosome,
                &mut warnings,
            )?;
            current_chromosome = Some(chromosome);
        }

        let union_value_count = variant_count
            .checked_mul(union_sample_count)
            .ok_or_else(|| DeliveryError::InvalidInput("union genotype dimensions overflow usize".to_string()))?;
        let union_buffer = union_buffer_pool.acquire(union_value_count, false)?;
        let OwnedGenotypeBuffer::Dosage(mut union_dosages) = union_buffer else {
            return Err(DeliveryError::InvalidInput(
                "grouped-union delivery acquired a packed genotype buffer".to_string(),
            ));
        };
        engine.reader.read_trusted_variant_major_dosage_f32_into_address_prepared(
            chunk_spec.variant_start_index,
            chunk_spec.variant_stop_index,
            g_genotype::OutputBufferAddress::from_mut_ptr(union_dosages.as_mut_ptr()),
            g_genotype::OutputValueCount::new(union_dosages.len()),
        )?;
        let submission_result = submit_grouped_union_chunk::<Backend, InterruptionError>(
            pipeline,
            group_runtimes,
            &union_dosages,
            union_sample_count,
            variant_count,
            chunk_spec.variant_start_index,
            &metadata,
        );
        union_buffer_pool.release(OwnedGenotypeBuffer::Dosage(union_dosages));
        submission_result?;
        processed_chunk_count += 1;
        let mut write_completed = |completed_batch: CompletedAssociationBatch| {
            write_grouped_union_completed_batch::<Backend, InterruptionError>(completed_batch, group_runtimes)
        };
        drain_available_batches(pipeline, &mut write_completed)?;
    }
    let mut write_completed = |completed_batch: CompletedAssociationBatch| {
        write_grouped_union_completed_batch::<Backend, InterruptionError>(completed_batch, group_runtimes)
    };
    finish_and_drain_pipeline(pipeline, &mut write_completed)?;
    check_interruption().map_err(DeliveryError::Interrupted)?;
    Ok(AssociationDeliveryReport { processed_chunk_count, warnings })
}

fn plan_grouped_union_requests<BackendError, InterruptionError>(
    engine: &BgenRunEngine,
    groups: Vec<AssociationDeliveryRequest>,
) -> DeliveryResult<Vec<PendingGroupedUnionGroup>, BackendError, InterruptionError> {
    let mut pending_groups = Vec::with_capacity(groups.len());
    for mut request in groups {
        let planned_delivery = plan_association_delivery(engine, &mut request)?;
        if !planned_delivery.chunk_specs.is_empty() {
            pending_groups.push(PendingGroupedUnionGroup {
                request,
                chunk_specs: planned_delivery.chunk_specs,
                chromosome_blocks: planned_delivery.chromosome_blocks,
            });
        }
    }
    Ok(pending_groups)
}

fn run_pending_groups_direct<Backend, CheckInterruption, InterruptionError>(
    engine: &BgenRunEngine,
    backend: &Arc<Backend>,
    pending_groups: Vec<PendingGroupedUnionGroup>,
    check_interruption: &mut CheckInterruption,
) -> DeliveryResult<AssociationDeliveryReport, Backend::Error, InterruptionError>
where
    Backend: AssociationBackend + 'static,
    Backend::ChromosomeState: 'static,
    Backend::DeviceResult: 'static,
    CheckInterruption: FnMut() -> Result<(), InterruptionError>,
{
    let mut processed_chunk_count = 0_usize;
    let mut warnings = Vec::new();
    for pending_group in pending_groups {
        let report = run_planned_association_delivery(
            engine,
            backend,
            pending_group.request,
            pending_group.chunk_specs,
            check_interruption,
        )?;
        processed_chunk_count = processed_chunk_count.checked_add(report.processed_chunk_count).ok_or_else(|| {
            DeliveryError::InvalidInput("direct resumed group chunk count overflowed usize".to_string())
        })?;
        warnings.extend(report.warnings);
    }
    Ok(AssociationDeliveryReport { processed_chunk_count, warnings })
}

fn prepare_grouped_union_runtimes<Backend, InterruptionError>(
    pipeline: &mut AssociationBatchPipeline<Backend>,
    pending_groups: Vec<PendingGroupedUnionGroup>,
    union_sample_indices: &[usize],
) -> DeliveryResult<Vec<GroupedUnionGroupRuntime<Backend>>, Backend::Error, InterruptionError>
where
    Backend: AssociationBackend + 'static,
    Backend::ChromosomeState: 'static,
    Backend::DeviceResult: 'static,
{
    let sample_positions_by_group = g_input::build_group_sample_position_arrays(
        union_sample_indices,
        pending_groups.iter().map(|group| group.request.group.sample_indices.as_slice()),
    )?;
    pending_groups
        .into_iter()
        .zip(sample_positions_by_group)
        .map(|(pending_group, sample_positions)| {
            let final_chunk_identifier = pending_group
                .chunk_specs
                .last()
                .expect("pending grouped delivery has at least one chunk")
                .variant_start_index;
            let group_index = pipeline.register_group()?;
            Ok(GroupedUnionGroupRuntime {
                group_index,
                request: pending_group.request,
                group_state: None,
                sample_positions,
                chromosome_blocks: pending_group.chromosome_blocks,
                next_chromosome_block_index: 0,
                final_chunk_identifier,
            })
        })
        .collect()
}

fn prepare_grouped_union_chromosome_states<Backend, InterruptionError>(
    backend: &Backend,
    pipeline: &mut AssociationBatchPipeline<Backend>,
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
        let Some(planned_chromosome) = group_runtime.chromosome_blocks.get(group_runtime.next_chromosome_block_index)
        else {
            continue;
        };
        if planned_chromosome.as_ref() != chromosome {
            continue;
        }
        if group_runtime.group_state.is_none() {
            group_runtime.group_state = Some(
                backend
                    .prepare_group(group_preparation_input(&mut group_runtime.request.group))
                    .map_err(|source| DeliveryError::Backend { stage: "prepare_group", source })?,
            );
        }
        let chromosome_state = prepare_chromosome_state(
            backend,
            group_runtime.group_state.as_ref().expect("group state was prepared immediately above"),
            &mut group_runtime.request.group,
            &group_runtime.request.settings,
            chromosome,
            warnings,
        )?;
        pipeline.prepare_chromosome(group_runtime.group_index, chromosome_state)?;
        group_runtime.next_chromosome_block_index += 1;
        if group_runtime.next_chromosome_block_index == group_runtime.chromosome_blocks.len() {
            backend.release_group(group_runtime.group_state.take().expect("final chromosome retained group state"));
        }
    }
    Ok(())
}

fn submit_grouped_union_chunk<Backend, InterruptionError>(
    pipeline: &mut AssociationBatchPipeline<Backend>,
    group_runtimes: &[GroupedUnionGroupRuntime<Backend>],
    union_dosages: &[f32],
    union_sample_count: usize,
    variant_count: usize,
    variant_start_index: usize,
    metadata: &g_genotype_contracts::VariantMetadataColumns,
) -> DeliveryResult<(), Backend::Error, InterruptionError>
where
    Backend: AssociationBackend + 'static,
    Backend::ChromosomeState: 'static,
    Backend::DeviceResult: 'static,
{
    let output_metadata = NativeVariantMetadataHandle::try_new(metadata)?;
    for group_runtime in group_runtimes {
        let settings = &group_runtime.request.settings;
        let active_trait_selection = active_trait_selection_for_chunk(
            settings.writer_sessions.len(),
            variant_start_index,
            &settings.committed_chunk_identifier_sets,
        )
        .map_err(DeliveryError::InvalidInput)?;
        if matches!(&active_trait_selection, ActiveTraitSelection::Indices(indices) if indices.is_empty()) {
            continue;
        }
        let sample_count = group_runtime.sample_positions.len();
        let genotype_buffer = allocate_genotype_buffer(variant_count, variant_count, sample_count, false)?;
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
        let statistics = g_genotype::summarize_variant_major_dosage_matrix(
            &group_dosages,
            sample_count,
            variant_count,
            settings.statistics_policy,
        )?;
        let scheduled_batch = ScheduledAssociationBatch {
            variant_start_index,
            compute_variant_count: variant_count,
            sample_count,
            metadata: output_metadata.clone(),
            statistics,
            genotype_buffer: OwnedGenotypeBuffer::Dosage(group_dosages),
            active_trait_selection,
        };
        let mut write_completed = |completed_batch: CompletedAssociationBatch| {
            write_grouped_union_completed_batch::<Backend, InterruptionError>(completed_batch, group_runtimes)
        };
        submit_batch(pipeline, group_runtime.group_index, scheduled_batch, &mut write_completed)?;
    }
    Ok(())
}

fn write_grouped_union_completed_batch<Backend, InterruptionError>(
    completed_batch: CompletedAssociationBatch,
    group_runtimes: &[GroupedUnionGroupRuntime<Backend>],
) -> DeliveryResult<(), Backend::Error, InterruptionError>
where
    Backend: AssociationBackend + 'static,
    Backend::ChromosomeState: 'static,
    Backend::DeviceResult: 'static,
{
    let group_index = completed_batch.group_index;
    let group_runtime = group_runtimes.get(group_index).ok_or_else(|| {
        DeliveryError::InvalidInput(format!("shared pipeline returned unknown group index {group_index}"))
    })?;
    if group_runtime.group_index != group_index {
        return Err(DeliveryError::InvalidInput(format!(
            "shared pipeline group routing mismatch: slot {group_index} stores group {}",
            group_runtime.group_index
        )));
    }
    write_completed_batch(completed_batch, &group_runtime.request.settings)
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
    if request.union_sample_indices
        != g_input::build_union_sample_indices(request.groups.iter().map(|group| group.group.sample_indices.as_slice()))
    {
        return Err(DeliveryError::InvalidInput(
            "union sample indices do not match the ordered group union".to_string(),
        ));
    }
    let first_settings = &request.groups[0].settings;
    for group in &request.groups {
        validate_delivery_request::<BackendError, InterruptionError>(group)?;
        if group.settings.use_packed8 {
            return Err(DeliveryError::InvalidInput(
                "grouped-union delivery supports dosage genotypes only".to_string(),
            ));
        }
        if group.settings.staging_depth != first_settings.staging_depth
            || group.settings.result_in_flight_limit != first_settings.result_in_flight_limit
        {
            return Err(DeliveryError::InvalidInput(
                "grouped-union delivery requires one shared queue-capacity policy".to_string(),
            ));
        }
    }
    Ok(())
}
