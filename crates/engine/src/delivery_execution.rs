//! Association delivery execution independent of Python.

use std::sync::Arc;

use g_genotype::{BgenError, GenotypeBatch, GenotypeBatchPayload, GenotypeError};
use g_input::PredictionError;
use g_output::{NativeVariantMetadataHandle, OutputError};

use crate::association_scheduler::{
    AssociationBatchPipeline, CompletedAssociationBatch, ScheduledAssociationBatch, SchedulerError,
};
use crate::backend::{
    AssociationBackend, GenotypeDeliveryCapability, GenotypeTransferPreparation, GroupPreparationInput,
    SampleMajorCovariateMatrix, TraitMajorMatrix,
};
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

enum PlannedGenotypeDelivery {
    Host,
    CompressedPacked8(g_genotype::CompressedPacked8BatchLayout),
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
    let genotype_delivery = plan_genotype_delivery(genotype_input, backend.as_ref(), &request.settings, &chunk_specs)?;
    let genotype_transfer = match &genotype_delivery {
        PlannedGenotypeDelivery::Host => GenotypeTransferPreparation::Host,
        PlannedGenotypeDelivery::CompressedPacked8(_) => {
            GenotypeTransferPreparation::CompressedPacked8(read_session.compressed_packed8_transfer().clone())
        }
    };
    let group_state = backend
        .prepare_group(group_preparation_input(&mut request.group, genotype_transfer))
        .map_err(|source| DeliveryError::Backend { stage: "prepare_group", source })?;
    run_with_explicit_group_release(
        group_state,
        |shared_group_state| {
            let mut pipeline = AssociationBatchPipeline::new(Arc::clone(backend), shared_group_state)?;
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
                    prepare_chromosome_state(
                        &mut pipeline,
                        &mut request.group,
                        &request.settings,
                        &chromosome,
                        &mut warnings,
                    )?;
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
                let genotypes = match &genotype_delivery {
                    PlannedGenotypeDelivery::Host => read_session.decode_variant_major_batch(
                        chunk_spec.variant_start_index,
                        chunk_spec.variant_stop_index,
                        compute_variant_count,
                        request.settings.gpu_genotype_format == g_plan::GpuGenotypeFormat::Packed8,
                        request.settings.statistics_policy,
                    )?,
                    PlannedGenotypeDelivery::CompressedPacked8(layout) => {
                        let logical_variant_count = chunk_spec.variant_stop_index - chunk_spec.variant_start_index;
                        GenotypeBatch {
                            variant_start_index: chunk_spec.variant_start_index,
                            logical_variant_count,
                            compute_variant_count,
                            sample_count,
                            payload: GenotypeBatchPayload::CompressedPacked8(
                                read_session.pack_compressed_packed8_batch(
                                    layout,
                                    chunk_spec.variant_start_index,
                                    chunk_spec.variant_stop_index,
                                )?,
                            ),
                        }
                    }
                };
                debug_assert_eq!(genotypes.sample_count, sample_count);
                let scheduled_batch = ScheduledAssociationBatch {
                    genotypes,
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
        },
        |group_state| backend.release_group(group_state),
        |cleanup_stage, panic_message| {
            tracing::error!(cleanup_stage, panic_message, "association backend cleanup panicked");
        },
    )
}

fn run_with_explicit_group_release<GroupState, Value, Error, RunDelivery, ReleaseGroup, ObserveCleanupPanic>(
    group_state: GroupState,
    run_delivery: RunDelivery,
    release_group: ReleaseGroup,
    mut observe_cleanup_panic: ObserveCleanupPanic,
) -> Result<Value, Error>
where
    GroupState: Send + Sync,
    RunDelivery: FnOnce(Arc<GroupState>) -> Result<Value, Error>,
    ReleaseGroup: FnOnce(GroupState),
    ObserveCleanupPanic: FnMut(&'static str, &str),
{
    let shared_group_state = Arc::new(group_state);
    let delivery_outcome =
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| run_delivery(Arc::clone(&shared_group_state))));
    let Ok(group_state) = Arc::try_unwrap(shared_group_state) else {
        panic!("association scheduler retained group state after joining its workers");
    };
    let release_outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| release_group(group_state)));
    match (delivery_outcome, release_outcome) {
        (Ok(Ok(value)), Ok(())) => Ok(value),
        (Ok(Ok(_)), Err(release_panic)) => {
            observe_cleanup_panic("release_group", panic_payload_message(release_panic.as_ref()).as_str());
            std::panic::resume_unwind(release_panic);
        }
        (Ok(Err(error)), Ok(())) => Err(error),
        (Ok(Err(error)), Err(release_panic)) => {
            observe_cleanup_panic("release_group", panic_payload_message(release_panic.as_ref()).as_str());
            Err(error)
        }
        (Err(delivery_panic), Ok(())) => std::panic::resume_unwind(delivery_panic),
        (Err(delivery_panic), Err(release_panic)) => {
            observe_cleanup_panic("release_group", panic_payload_message(release_panic.as_ref()).as_str());
            std::panic::resume_unwind(delivery_panic);
        }
    }
}

fn panic_payload_message(payload: &(dyn std::any::Any + Send)) -> String {
    payload.downcast_ref::<&str>().map_or_else(
        || payload.downcast_ref::<String>().cloned().unwrap_or_else(|| "unknown panic payload".to_string()),
        |message| (*message).to_string(),
    )
}

fn plan_genotype_delivery<Backend, InterruptionError>(
    genotype_input: &PreparedGenotypeInput,
    backend: &Backend,
    settings: &AssociationDeliverySettings,
    chunk_specs: &[g_genotype::ChunkSpec],
) -> DeliveryResult<PlannedGenotypeDelivery, Backend::Error, InterruptionError>
where
    Backend: AssociationBackend + 'static,
{
    if chunk_specs.is_empty()
        || settings.gpu_genotype_format != g_plan::GpuGenotypeFormat::Packed8
        || backend.genotype_delivery_capability() != GenotypeDeliveryCapability::RawDeflatePacked8
    {
        return Ok(PlannedGenotypeDelivery::Host);
    }
    Ok(match genotype_input.reader.plan_compressed_packed8_batch_layout(chunk_specs)? {
        Some(layout) => PlannedGenotypeDelivery::CompressedPacked8(layout),
        None => PlannedGenotypeDelivery::Host,
    })
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

fn group_preparation_input(
    group: &mut g_input::AlignedPhenotypeGroup,
    genotype_transfer: GenotypeTransferPreparation,
) -> GroupPreparationInput {
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
        genotype_transfer,
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
    pipeline: &mut AssociationBatchPipeline<Backend>,
    group: &mut g_input::AlignedPhenotypeGroup,
    settings: &AssociationDeliverySettings,
    chromosome: &str,
    warnings: &mut Vec<DeliveryWarning>,
) -> DeliveryResult<(), Backend::Error, InterruptionError>
where
    Backend: AssociationBackend + 'static,
{
    let predictions = group.take_chromosome_prediction_matrix(chromosome)?;
    let null_logistic_converged = pipeline.prepare_chromosome(predictions)?;
    if let Some(logistic_converged) = null_logistic_converged.as_ref()
        && let Err(policy_error) = enforce_null_logistic_policy::<Backend::Error, InterruptionError>(
            chromosome,
            logistic_converged,
            &group.phenotype_group.phenotype_names,
            settings.null_logistic_nonconvergence_policy,
            warnings,
        )
    {
        if let Err(release_error) = pipeline.release_chromosome() {
            tracing::error!(
                cleanup_stage = "release_chromosome_after_null_policy_rejection",
                cleanup_error = %release_error,
                "association backend cleanup failed"
            );
        }
        return Err(policy_error);
    }
    Ok(())
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
    let CompletedAssociationBatch { context, statistics, result } = completed_batch;
    let crate::association_scheduler::AssociationBatchContext { variant_start_index, metadata, active_trait_selection } =
        context;
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

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, Eq, PartialEq, thiserror::Error)]
    #[error("test backend failure")]
    struct TestBackendError;

    #[derive(Debug, Eq, PartialEq, thiserror::Error)]
    #[error("test interruption")]
    struct TestInterruption;

    #[test]
    fn delivery_policy_continues_when_every_null_model_converges() {
        let mut warnings = Vec::new();
        enforce_null_logistic_policy::<TestBackendError, TestInterruption>(
            "22",
            &[true, true],
            &["trait-a".to_string(), "trait-b".to_string()],
            g_plan::NullLogisticNonconvergencePolicy::Fail,
            &mut warnings,
        )
        .expect("converged models continue");
        assert!(warnings.is_empty());
    }

    #[test]
    fn delivery_policy_records_structured_warning_counts() {
        let mut warnings = Vec::new();
        enforce_null_logistic_policy::<TestBackendError, TestInterruption>(
            "7",
            &[false, true, false],
            &["trait-a".to_string(), "trait-b".to_string(), "trait-c".to_string()],
            g_plan::NullLogisticNonconvergencePolicy::Warn,
            &mut warnings,
        )
        .expect("warning policy continues");
        assert_eq!(
            warnings,
            vec![DeliveryWarning {
                chromosome: "7".to_string(),
                message: "Binary null logistic model did not converge for chromosome 7: trait-a, trait-c. Continuing because --null_logistic_nonconvergence_policy=warn.".to_string(),
                nonconverged_count: 2,
                total_fit_count: 3,
            }]
        );
    }

    #[test]
    fn delivery_policy_stops_without_recording_a_warning_under_fail_policy() {
        let mut warnings = Vec::new();
        let error = enforce_null_logistic_policy::<TestBackendError, TestInterruption>(
            "1",
            &[true, false],
            &["trait-a".to_string(), "trait-b".to_string()],
            g_plan::NullLogisticNonconvergencePolicy::Fail,
            &mut warnings,
        )
        .expect_err("fail policy stops the delivery");
        assert!(matches!(
            error,
            DeliveryError::NullLogisticNonconvergence(message)
                if message.contains("chromosome 1: trait-b")
        ));
        assert!(warnings.is_empty());
    }

    #[test]
    fn group_release_runs_once_after_delivery_panic() {
        let release_count = std::sync::atomic::AtomicUsize::new(0);
        let cleanup_panic_count = std::sync::atomic::AtomicUsize::new(0);
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            run_with_explicit_group_release(
                (),
                |_group_state| -> Result<(), TestInterruption> {
                    panic!("intentional delivery panic");
                },
                |()| {
                    release_count.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                },
                |_cleanup_stage, _panic_message| {
                    cleanup_panic_count.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                },
            )
        }));

        let panic_payload = outcome.expect_err("delivery panic is resumed after group release");
        let panic_message =
            panic_payload.downcast_ref::<&str>().copied().expect("test panic payload is a string literal");
        assert_eq!(panic_message, "intentional delivery panic");
        assert_eq!(release_count.load(std::sync::atomic::Ordering::SeqCst), 1);
        assert_eq!(cleanup_panic_count.load(std::sync::atomic::Ordering::SeqCst), 0);
    }

    #[test]
    fn delivery_error_remains_primary_when_group_release_panics() {
        let release_count = std::sync::atomic::AtomicUsize::new(0);
        let mut cleanup_panics = Vec::new();
        let delivery_result = run_with_explicit_group_release(
            (),
            |_group_state| Err::<(), _>(TestInterruption),
            |()| {
                release_count.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                panic!("intentional group-release panic");
            },
            |cleanup_stage, panic_message| {
                cleanup_panics.push((cleanup_stage.to_string(), panic_message.to_string()));
            },
        );

        assert_eq!(delivery_result, Err(TestInterruption));
        assert_eq!(release_count.load(std::sync::atomic::Ordering::SeqCst), 1);
        assert_eq!(cleanup_panics, vec![("release_group".to_string(), "intentional group-release panic".to_string())]);
    }

    #[test]
    fn delivery_panic_remains_primary_and_observes_group_release_panic() {
        let release_count = std::sync::atomic::AtomicUsize::new(0);
        let mut cleanup_panics = Vec::new();
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            run_with_explicit_group_release(
                (),
                |_group_state| -> Result<(), TestInterruption> {
                    panic!("intentional delivery panic");
                },
                |()| {
                    release_count.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                    panic!("intentional group-release panic");
                },
                |cleanup_stage, panic_message| {
                    cleanup_panics.push((cleanup_stage.to_string(), panic_message.to_string()));
                },
            )
        }));

        let panic_payload = outcome.expect_err("primary delivery panic is resumed");
        let panic_message =
            panic_payload.downcast_ref::<&str>().copied().expect("test panic payload is a string literal");
        assert_eq!(panic_message, "intentional delivery panic");
        assert_eq!(release_count.load(std::sync::atomic::Ordering::SeqCst), 1);
        assert_eq!(cleanup_panics, vec![("release_group".to_string(), "intentional group-release panic".to_string())]);
    }
}
