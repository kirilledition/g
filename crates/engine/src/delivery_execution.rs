//! Association delivery execution independent of Python.

use std::sync::Arc;

use g_genotype::{BgenError, GenotypeError};
use g_input::PredictionError;
use g_output::{NativeVariantMetadataHandle, OutputError};

use crate::association_scheduler::{
    AssociationBatchOutput, AssociationBatchPipeline, CompletedAssociationBatch, ScheduledAssociationBatch,
    SchedulerError,
};
use crate::backend::{AssociationBackend, GroupPreparationInput, SampleMajorCovariateMatrix, TraitMajorMatrix};
use crate::delivery::{AssociationDeliveryRequest, AssociationDeliverySettings, PreparedGenotypeInput};
use crate::genotype_buffer::homogeneous_chunk_chromosome;
use crate::null_logistic_policy::{
    NullLogisticNonconvergenceAction, NullLogisticPolicyError, plan_null_logistic_nonconvergence,
};
use crate::output_schedule::{
    ActiveTraitSelection, active_trait_selection_for_chunk, intersect_committed_chunk_identifier_sets,
};
use crate::output_write::write_host_association_batch;
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
    genotype_input: &PreparedGenotypeInput,
    backend: &Arc<Backend>,
    mut request: AssociationDeliveryRequest,
    mut check_interruption: CheckInterruption,
) -> DeliveryResult<AssociationDeliveryReport, Backend::Error, InterruptionError>
where
    Backend: AssociationBackend + 'static,
    CheckInterruption: FnMut() -> Result<(), InterruptionError>,
{
    validate_delivery_request::<Backend::Error, InterruptionError>(&request)?;
    let chunk_specs = plan_association_delivery(genotype_input, &mut request)?;
    if chunk_specs.is_empty() {
        check_interruption().map_err(DeliveryError::Interrupted)?;
        return Ok(AssociationDeliveryReport { processed_chunk_count: 0, warnings: Vec::new() });
    }
    let read_session = genotype_input.reader.read_session(&request.group.sample_indices)?;
    let delivery_result = run_prepared_association_delivery(
        genotype_input,
        &read_session,
        backend,
        &mut request,
        chunk_specs,
        &mut check_interruption,
    );
    let source_result = read_session.finish().map_err(DeliveryError::Bgen);
    match delivery_result {
        Err(error) => Err(error),
        Ok(report) => source_result.map(|()| report),
    }
}

fn run_prepared_association_delivery<Backend, CheckInterruption, InterruptionError>(
    genotype_input: &PreparedGenotypeInput,
    read_session: &g_genotype::BgenReadSession<'_>,
    backend: &Arc<Backend>,
    request: &mut AssociationDeliveryRequest,
    chunk_specs: Vec<g_genotype::ChunkSpec>,
    check_interruption: &mut CheckInterruption,
) -> DeliveryResult<AssociationDeliveryReport, Backend::Error, InterruptionError>
where
    Backend: AssociationBackend + 'static,
    CheckInterruption: FnMut() -> Result<(), InterruptionError>,
{
    let group_state = backend
        .prepare_group(group_preparation_input(&mut request.group))
        .map_err(|source| DeliveryError::Backend { stage: "prepare_group", source })?;
    let delivery_result: DeliveryResult<AssociationDeliveryReport, Backend::Error, InterruptionError> = (|| {
        let mut pipeline = AssociationBatchPipeline::new(Arc::clone(backend))?;
        let mut current_chromosome = None;
        let mut processed_chunk_count = 0_usize;
        let mut warnings = Vec::new();
        let compute_variant_count = genotype_input.chunk_size.min(genotype_input.reader.variant_count());

        for chunk_spec in chunk_specs {
            check_interruption().map_err(DeliveryError::Interrupted)?;
            let variant_count = chunk_spec
                .variant_stop_index
                .checked_sub(chunk_spec.variant_start_index)
                .ok_or_else(|| DeliveryError::InvalidInput("BGEN chunk stop precedes its start.".to_string()))?;
            let metadata = genotype_input
                .reader
                .variant_metadata_slice(chunk_spec.variant_start_index, chunk_spec.variant_stop_index)?;
            let chromosome = homogeneous_chunk_chromosome(&metadata, variant_count)?;
            if current_chromosome.as_deref() != Some(chromosome.as_ref()) {
                drain_pending_batches(&mut pipeline, &request.settings)?;
                pipeline.release_chromosome()?;
                let chromosome_state = prepare_chromosome_state(
                    backend.as_ref(),
                    &group_state,
                    &mut request.group,
                    &request.settings,
                    &chromosome,
                    &mut warnings,
                )?;
                pipeline.prepare_chromosome(chromosome_state)?;
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
            let decoded = read_session.decode_variant_major_batch(
                chunk_spec.variant_start_index,
                chunk_spec.variant_stop_index,
                compute_variant_count,
                request.settings.use_packed8,
                request.settings.statistics_policy,
            )?;
            debug_assert_eq!(decoded.sample_count, sample_count);
            let scheduled_batch = ScheduledAssociationBatch {
                decoded,
                metadata: NativeVariantMetadataHandle::try_new(&metadata)?,
                active_trait_selection,
            };
            submit_batch(&mut pipeline, scheduled_batch, &request.settings)?;
            processed_chunk_count += 1;
            drain_available_batches(&mut pipeline, &request.settings)?;
        }

        finish_and_drain_pipeline(&mut pipeline, &request.settings)?;
        check_interruption().map_err(DeliveryError::Interrupted)?;
        Ok(AssociationDeliveryReport { processed_chunk_count, warnings })
    })();
    backend.release_group(group_state);
    delivery_result
}

fn plan_association_delivery<BackendError, InterruptionError>(
    genotype_input: &PreparedGenotypeInput,
    request: &mut AssociationDeliveryRequest,
) -> DeliveryResult<Vec<g_genotype::ChunkSpec>, BackendError, InterruptionError> {
    let committed_chunk_identifiers =
        intersect_committed_chunk_identifier_sets(&request.settings.committed_chunk_identifier_sets);
    let chunk_specs = genotype_input
        .reader
        .plan_chromosome_homogeneous_chunks(genotype_input.chunk_size, &committed_chunk_identifiers)?;
    let chromosome_blocks = planned_chromosome_blocks(genotype_input, &chunk_specs)?;
    request.group.plan_prediction_uses(&chromosome_blocks)?;
    if let Some(progress) = request.settings.progress.as_ref() {
        progress.initialize(&chunk_specs)?;
    }
    Ok(chunk_specs)
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
    genotype_input: &PreparedGenotypeInput,
    chunk_specs: &[g_genotype::ChunkSpec],
) -> Result<Vec<Arc<str>>, BgenError> {
    let mut chromosome_blocks = Vec::new();
    for chunk_spec in chunk_specs {
        if chunk_spec.variant_start_index >= chunk_spec.variant_stop_index {
            return Err(BgenError::Range("Planned association chunk is empty.".to_string()));
        }
        let metadata = genotype_input
            .reader
            .variant_metadata_slice(chunk_spec.variant_start_index, chunk_spec.variant_start_index + 1)?;
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
{
    let predictions = group.take_chromosome_prediction_matrix(chromosome)?;
    let prepared_chromosome = backend
        .prepare_chromosome(group_state, predictions)
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
    Ok(())
}

fn drain_available_batches<Backend, InterruptionError>(
    pipeline: &mut AssociationBatchPipeline<Backend>,
    settings: &AssociationDeliverySettings,
) -> DeliveryResult<(), Backend::Error, InterruptionError>
where
    Backend: AssociationBackend + 'static,
{
    while let Some(completed_batch) = pipeline.try_receive()? {
        write_completed_batch(completed_batch, settings)?;
    }
    Ok(())
}

fn submit_batch<Backend, InterruptionError>(
    pipeline: &mut AssociationBatchPipeline<Backend>,
    scheduled_batch: ScheduledAssociationBatch,
    settings: &AssociationDeliverySettings,
) -> DeliveryResult<(), Backend::Error, InterruptionError>
where
    Backend: AssociationBackend + 'static,
{
    let mut pending_batch = scheduled_batch;
    loop {
        match pipeline.try_submit(pending_batch)? {
            None => return Ok(()),
            Some(returned_batch) => {
                write_completed_batch(pipeline.receive()?, settings)?;
                pending_batch = returned_batch;
            }
        }
    }
}

fn drain_pending_batches<Backend, InterruptionError>(
    pipeline: &mut AssociationBatchPipeline<Backend>,
    settings: &AssociationDeliverySettings,
) -> DeliveryResult<(), Backend::Error, InterruptionError>
where
    Backend: AssociationBackend + 'static,
{
    while !pipeline.is_drained() {
        write_completed_batch(pipeline.receive()?, settings)?;
    }
    Ok(())
}

fn finish_and_drain_pipeline<Backend, InterruptionError>(
    pipeline: &mut AssociationBatchPipeline<Backend>,
    settings: &AssociationDeliverySettings,
) -> DeliveryResult<(), Backend::Error, InterruptionError>
where
    Backend: AssociationBackend + 'static,
{
    drain_pending_batches(pipeline, settings)?;
    pipeline.release_chromosome()?;
    pipeline.close_submission();
    pipeline.join()?;
    Ok(())
}

fn write_completed_batch<BackendError, InterruptionError>(
    completed_batch: CompletedAssociationBatch,
    settings: &AssociationDeliverySettings,
) -> DeliveryResult<(), BackendError, InterruptionError> {
    let CompletedAssociationBatch { output, result } = completed_batch;
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
