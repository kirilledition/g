//! Bounded two-group reuse of one immutable GPU genotype source chunk.

use std::sync::Arc;

use g_genotype::{BgenReadSession, ChunkSpec, GenotypeBatch, GenotypeBatchPayload};
use g_output::NativeVariantMetadataHandle;

use crate::association_scheduler::{AssociationBatchContext, AssociationBatchPipeline, ScheduledAssociationBatch};
use crate::backend::{AssociationBackend, GenotypeDeliveryCapability, GenotypeTransferPreparation};
use crate::delivery::{AssociationDeliveryRequest, AssociationDeliverySettings, PreparedGenotypeInput};
use crate::delivery_execution::{
    AssociationDeliveryReport, DeliveryError, DeliveryResult, drain_pending_batches, finish_and_drain_pipeline,
    group_preparation_input, plan_association_delivery, prepare_chromosome_state, run_association_delivery,
    run_planned_association_delivery, submit_batch, validate_delivery_request,
};
use crate::genotype_buffer::homogeneous_chunk_chromosome;
use crate::output_schedule::{ActiveTraitSelection, active_trait_selection_for_chunk};
use crate::tiled_delivery_plan::{
    TiledGroupGeometry, has_shared_pending_chunk, merge_pending_chunks, within_retained_memory_limits,
};

struct PlannedDelivery {
    request: AssociationDeliveryRequest,
    chunks: Vec<ChunkSpec>,
    current_chromosome: Option<Arc<str>>,
    report: AssociationDeliveryReport,
}

struct GroupStatesGuard<'backend, Backend: AssociationBackend> {
    backend: &'backend Backend,
    states: Vec<Backend::GroupState>,
}

impl<Backend: AssociationBackend> Drop for GroupStatesGuard<'_, Backend> {
    fn drop(&mut self) {
        for state in self.states.drain(..) {
            self.backend.release_group(state);
        }
    }
}

struct SharedSourceGuard<'backend, Backend: AssociationBackend> {
    backend: &'backend Backend,
    source: Option<Backend::SharedSourceBatch>,
}

impl<Backend: AssociationBackend> SharedSourceGuard<'_, Backend> {
    fn prepare<InterruptionError>(
        &mut self,
        input: GenotypeBatch,
    ) -> DeliveryResult<(), Backend::Error, InterruptionError> {
        if self.source.is_some() {
            return Err(DeliveryError::InvalidInput("Previous shared source was not released.".to_string()));
        }
        self.source = self
            .backend
            .prepare_shared_source(input)
            .map_err(|source| DeliveryError::Backend { stage: "prepare_shared_source", source })?;
        if self.source.is_none() {
            return Err(DeliveryError::InvalidInput(
                "Backend advertised shared-source support but did not prepare a source.".to_string(),
            ));
        }
        Ok(())
    }

    fn release(&mut self) {
        if let Some(source) = self.source.take() {
            self.backend.release_shared_source(source);
        }
    }
}

impl<Backend: AssociationBackend> Drop for SharedSourceGuard<'_, Backend> {
    fn drop(&mut self) {
        self.release();
    }
}

/// Execute at most two groups, sharing only admitted compressed linear input.
///
/// All eligibility and fallback decisions precede backend state preparation.
/// Once execution starts, backend failures remain failures instead of replaying
/// work that may already have reached an output writer.
///
/// # Errors
///
/// Returns the same typed delivery failures as ordinary group execution.
pub(crate) fn run_association_delivery_tile<Backend, CheckInterruption, InterruptionError>(
    genotype_input: &PreparedGenotypeInput,
    backend: Option<&Arc<Backend>>,
    requests: Vec<AssociationDeliveryRequest>,
    mut check_interruption: CheckInterruption,
) -> DeliveryResult<Vec<AssociationDeliveryReport>, Backend::Error, InterruptionError>
where
    Backend: AssociationBackend + 'static,
    CheckInterruption: FnMut() -> Result<(), InterruptionError>,
{
    let compute_variant_count = genotype_input.chunk_size.min(genotype_input.reader.variant_count());
    let geometries = requests
        .iter()
        .map(|request| TiledGroupGeometry {
            samples: request.group.sample_indices.len(),
            covariates: request.group.covariate_names.len(),
            traits: request.group.phenotype_group.phenotype_names.len(),
        })
        .collect::<Vec<_>>();
    let eligible_backend = backend.filter(|backend| {
        backend.supports_shared_source_batches()
            && backend.genotype_delivery_capability() == GenotypeDeliveryCapability::RawDeflatePacked8
    });
    let eligible_requests = requests.iter().all(|request| {
        request.settings.gpu_genotype_format == g_plan::GpuGenotypeFormat::Packed8
            && request.settings.statistics_policy.retain_imputed_dosage_square_sum
            && !request.settings.statistics_policy.collect_sparse_candidate_mask
    });
    let Some(shared_backend) = eligible_backend.filter(|_| {
        eligible_requests
            && within_retained_memory_limits(compute_variant_count, genotype_input.reader.sample_count(), &geometries)
    }) else {
        return requests
            .into_iter()
            .map(|request| run_association_delivery(genotype_input, backend, request, &mut check_interruption))
            .collect();
    };

    let mut deliveries = Vec::with_capacity(2);
    for mut request in requests {
        validate_delivery_request::<Backend::Error, InterruptionError>(&request)?;
        let chunks = plan_association_delivery(genotype_input, &mut request)?;
        deliveries.push(PlannedDelivery {
            request,
            chunks,
            current_chromosome: None,
            report: AssociationDeliveryReport { processed_chunk_count: 0, warnings: Vec::new() },
        });
    }
    if !has_shared_pending_chunk(&deliveries[0].chunks, &deliveries[1].chunks) {
        return run_planned_sequential(genotype_input, backend, deliveries, &mut check_interruption);
    }
    let chunks =
        merge_pending_chunks(&deliveries[0].chunks, &deliveries[1].chunks).map_err(DeliveryError::InvalidInput)?;
    let Some(layout) = genotype_input.compressed_layout_for_chunks(&chunks)? else {
        return run_planned_sequential(genotype_input, backend, deliveries, &mut check_interruption);
    };
    let sessions = deliveries
        .iter()
        .map(|delivery| genotype_input.reader.read_session(&delivery.request.group.sample_indices))
        .collect::<Result<Vec<_>, _>>()?;
    let delivery_result = run_prepared_tile(
        genotype_input,
        shared_backend,
        &mut deliveries,
        &sessions,
        &layout,
        &chunks,
        &mut check_interruption,
    );
    // Check every opened session even on failure, preserving the ordinary
    // delivery error's priority over a subsequently detected source change.
    let mut source_result = Ok(());
    for session in sessions {
        if let Err(error) = session.finish()
            && source_result.is_ok()
        {
            source_result = Err(DeliveryError::Bgen(error));
        }
    }
    delivery_result?;
    source_result?;
    Ok(deliveries.into_iter().map(|delivery| delivery.report).collect())
}

fn run_planned_sequential<Backend, CheckInterruption, InterruptionError>(
    genotype_input: &PreparedGenotypeInput,
    backend: Option<&Arc<Backend>>,
    deliveries: Vec<PlannedDelivery>,
    check_interruption: &mut CheckInterruption,
) -> DeliveryResult<Vec<AssociationDeliveryReport>, Backend::Error, InterruptionError>
where
    Backend: AssociationBackend + 'static,
    CheckInterruption: FnMut() -> Result<(), InterruptionError>,
{
    deliveries
        .into_iter()
        .map(|delivery| {
            run_planned_association_delivery(
                genotype_input,
                backend,
                delivery.request,
                delivery.chunks,
                check_interruption,
            )
        })
        .collect()
}

fn run_prepared_tile<Backend, CheckInterruption, InterruptionError>(
    genotype_input: &PreparedGenotypeInput,
    backend: &Arc<Backend>,
    deliveries: &mut [PlannedDelivery],
    sessions: &[BgenReadSession<'_>],
    layout: &g_genotype::CompressedPacked8BatchLayout,
    chunks: &[ChunkSpec],
    check_interruption: &mut CheckInterruption,
) -> DeliveryResult<(), Backend::Error, InterruptionError>
where
    Backend: AssociationBackend + 'static,
    CheckInterruption: FnMut() -> Result<(), InterruptionError>,
{
    let mut groups = GroupStatesGuard { backend: backend.as_ref(), states: Vec::with_capacity(2) };
    for (delivery, session) in deliveries.iter_mut().zip(sessions) {
        check_interruption().map_err(DeliveryError::Interrupted)?;
        let transfer = GenotypeTransferPreparation::CompressedPacked8(session.compressed_packed8_transfer().clone());
        let state = backend
            .prepare_group(group_preparation_input(&mut delivery.request.group, transfer))
            .map_err(|source| DeliveryError::Backend { stage: "prepare_group", source })?;
        groups.states.push(state);
    }
    // Declaration order is intentional: all pipeline workers abort/join before
    // the shared source is released, and group state is released last. This
    // also covers a failure while starting the second pipeline.
    let mut source = SharedSourceGuard { backend: backend.as_ref(), source: None };
    let mut pipelines = Vec::with_capacity(2);
    for state in &groups.states {
        pipelines.push(AssociationBatchPipeline::new(Arc::clone(backend), state)?);
    }
    let compute_variant_count = genotype_input.chunk_size.min(genotype_input.reader.variant_count());
    for chunk in chunks {
        check_interruption().map_err(DeliveryError::Interrupted)?;
        let active_groups = deliveries
            .iter()
            .map(|delivery| {
                delivery
                    .chunks
                    .binary_search_by_key(&chunk.variant_start_index, |chunk| chunk.variant_start_index)
                    .is_ok()
            })
            .collect::<Vec<_>>();
        let metadata =
            genotype_input.reader.variant_metadata_slice(chunk.variant_start_index, chunk.variant_stop_index)?;
        let logical_variant_count = chunk.variant_stop_index - chunk.variant_start_index;
        let chromosome = homogeneous_chunk_chromosome(&metadata, logical_variant_count)?;
        prepare_tile_chromosomes(
            backend.as_ref(),
            &groups.states,
            &mut pipelines,
            deliveries,
            &active_groups,
            &chromosome,
            check_interruption,
        )?;
        let shared = active_groups.iter().all(|active| *active);
        if shared {
            // Compressed members do not depend on a session's sample selection.
            // Decode them with the full file count, then select each group from
            // the immutable source; never pass the first group's selected count.
            let input = compressed_batch(
                &sessions[0],
                layout,
                chunk,
                compute_variant_count,
                genotype_input.reader.sample_count(),
            )?;
            source.prepare(input)?;
        }
        for (group_index, delivery) in deliveries.iter_mut().enumerate() {
            if !active_groups[group_index] {
                continue;
            }
            check_interruption().map_err(DeliveryError::Interrupted)?;
            let settings = &delivery.request.settings;
            let active_trait_selection = active_traits_for_pending_chunk(settings, chunk.variant_start_index)
                .map_err(DeliveryError::InvalidInput)?;
            let output_metadata = NativeVariantMetadataHandle::try_new(&metadata)?;
            let pipeline = &mut pipelines[group_index];
            if let Some(shared_source) = source.source.as_ref() {
                submit_shared_batch(
                    backend.as_ref(),
                    &groups.states[group_index],
                    shared_source,
                    pipeline,
                    AssociationBatchContext {
                        variant_start_index: chunk.variant_start_index,
                        metadata: output_metadata,
                        active_trait_selection,
                    },
                    compute_variant_count,
                )?;
            } else {
                let genotypes = compressed_batch(
                    &sessions[group_index],
                    layout,
                    chunk,
                    compute_variant_count,
                    delivery.request.group.sample_indices.len(),
                )?;
                submit_batch(
                    pipeline,
                    ScheduledAssociationBatch { genotypes, metadata: output_metadata, active_trait_selection },
                    settings,
                )?;
            }
        }
        drain_tile_batches(&mut pipelines, deliveries, &active_groups, check_interruption)?;
        source.release();
    }
    for (pipeline, delivery) in pipelines.iter_mut().zip(deliveries.iter()) {
        finish_and_drain_pipeline(pipeline, &delivery.request.settings)?;
    }
    check_interruption().map_err(DeliveryError::Interrupted)?;
    Ok(())
}

fn drain_tile_batches<Backend, CheckInterruption, InterruptionError>(
    pipelines: &mut [AssociationBatchPipeline<'_, Backend>],
    deliveries: &mut [PlannedDelivery],
    active_groups: &[bool],
    check_interruption: &mut CheckInterruption,
) -> DeliveryResult<(), Backend::Error, InterruptionError>
where
    Backend: AssociationBackend + 'static,
    CheckInterruption: FnMut() -> Result<(), InterruptionError>,
{
    let mut drained_active_group = false;
    for (group_index, delivery) in deliveries.iter_mut().enumerate() {
        if !active_groups[group_index] {
            continue;
        }
        // Keep accepted output when interrupted between consumers; remaining
        // pipelines still join before the shared source and group owners drop.
        if drained_active_group {
            check_interruption().map_err(DeliveryError::Interrupted)?;
        }
        drain_pending_batches(&mut pipelines[group_index], &delivery.request.settings)?;
        delivery.report.processed_chunk_count += 1;
        drained_active_group = true;
    }
    Ok(())
}

fn active_traits_for_pending_chunk(
    settings: &AssociationDeliverySettings,
    variant_start_index: usize,
) -> Result<ActiveTraitSelection, String> {
    let selection = active_trait_selection_for_chunk(
        settings.writer_sessions.len(),
        variant_start_index,
        &settings.committed_chunk_identifier_sets,
    )?;
    if matches!(&selection, ActiveTraitSelection::Indices(indices) if indices.is_empty()) {
        return Err("planned a chunk already committed by every output writer".to_string());
    }
    Ok(selection)
}

fn prepare_tile_chromosomes<Backend, CheckInterruption, InterruptionError>(
    backend: &Backend,
    groups: &[Backend::GroupState],
    pipelines: &mut [AssociationBatchPipeline<'_, Backend>],
    deliveries: &mut [PlannedDelivery],
    active_groups: &[bool],
    chromosome: &Arc<str>,
    check_interruption: &mut CheckInterruption,
) -> DeliveryResult<(), Backend::Error, InterruptionError>
where
    Backend: AssociationBackend + 'static,
    CheckInterruption: FnMut() -> Result<(), InterruptionError>,
{
    for (group_index, delivery) in deliveries.iter_mut().enumerate() {
        if !active_groups[group_index] {
            continue;
        }
        check_interruption().map_err(DeliveryError::Interrupted)?;
        if delivery.current_chromosome.as_deref() != Some(chromosome.as_ref()) {
            let pipeline = &mut pipelines[group_index];
            pipeline.release_chromosome()?;
            let chromosome_state = prepare_chromosome_state(
                backend,
                &groups[group_index],
                &mut delivery.request.group,
                &delivery.request.settings,
                chromosome,
                &mut delivery.report.warnings,
            )?;
            pipeline.prepare_chromosome(chromosome_state)?;
            delivery.current_chromosome = Some(Arc::clone(chromosome));
        }
    }
    Ok(())
}

fn submit_shared_batch<Backend, InterruptionError>(
    backend: &Backend,
    group: &Backend::GroupState,
    source: &Backend::SharedSourceBatch,
    pipeline: &mut AssociationBatchPipeline<'_, Backend>,
    context: AssociationBatchContext,
    compute_variant_count: usize,
) -> DeliveryResult<(), Backend::Error, InterruptionError>
where
    Backend: AssociationBackend + 'static,
{
    let input = backend
        .select_shared_source(group, source)
        .map_err(|source| DeliveryError::Backend { stage: "select_shared_source", source })?
        .ok_or_else(|| {
            DeliveryError::InvalidInput(
                "Backend advertised shared-source support but did not select a batch.".to_string(),
            )
        })?;
    pipeline.submit_pretransferred(context, input, compute_variant_count)?;
    Ok(())
}

fn compressed_batch(
    session: &BgenReadSession<'_>,
    layout: &g_genotype::CompressedPacked8BatchLayout,
    chunk: &ChunkSpec,
    compute_variant_count: usize,
    sample_count: usize,
) -> Result<GenotypeBatch, g_genotype::BgenError> {
    Ok(GenotypeBatch {
        variant_start_index: chunk.variant_start_index,
        logical_variant_count: chunk.variant_stop_index - chunk.variant_start_index,
        compute_variant_count,
        sample_count,
        payload: GenotypeBatchPayload::CompressedPacked8(session.pack_compressed_packed8_batch(
            layout,
            chunk.variant_start_index,
            chunk.variant_stop_index,
        )?),
    })
}
