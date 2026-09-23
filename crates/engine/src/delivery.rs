//! Python-free association delivery inputs.

use std::cell::RefCell;
use std::sync::Arc;

use g_genotype::ChunkStatisticsPolicy;
use g_input::AlignedPhenotypeGroup;
use g_output::OutputWriterSession;
use g_plan::{GpuGenotypeFormat, NullLogisticNonconvergencePolicy};

use crate::progress::DeliveryProgress;

/// Prepared genotype input shared by one or more association deliveries.
pub(crate) struct PreparedGenotypeInput {
    pub(crate) reader: g_genotype::BgenReaderCore,
    pub(crate) chunk_size: usize,
    compressed_layout: RefCell<Option<PreparedCompressedLayout>>,
}

struct PreparedCompressedLayout {
    chunk_specs: Vec<g_genotype::ChunkSpec>,
    layout: Arc<g_genotype::CompressedPacked8BatchLayout>,
}

impl PreparedGenotypeInput {
    pub(crate) fn new(reader: g_genotype::BgenReaderCore, chunk_size: usize) -> Self {
        Self { reader, chunk_size, compressed_layout: RefCell::new(None) }
    }

    /// Reuse compressed geometry only for the exact previous pending chunk plan.
    ///
    /// The single-entry cache retains no genotype data or sample-dependent state.
    /// A different resume plan replaces the entry, preserving the slab shape
    /// derived from that delivery's remaining chunks.
    ///
    /// # Errors
    ///
    /// Returns the reader's layout validation error when a new plan is invalid.
    pub(crate) fn compressed_layout_for_chunks(
        &self,
        chunk_specs: &[g_genotype::ChunkSpec],
    ) -> Result<Option<Arc<g_genotype::CompressedPacked8BatchLayout>>, g_genotype::BgenError> {
        let mut cached_layout = self.compressed_layout.borrow_mut();
        if let Some(cached) = cached_layout.as_ref()
            && cached.chunk_specs == chunk_specs
        {
            return Ok(Some(Arc::clone(&cached.layout)));
        }
        let Some(layout) = self.reader.plan_compressed_packed8_batch_layout(chunk_specs)? else {
            return Ok(None);
        };
        let layout = Arc::new(layout);
        *cached_layout = Some(PreparedCompressedLayout {
            chunk_specs: chunk_specs
                .iter()
                .map(|chunk| g_genotype::ChunkSpec {
                    variant_start_index: chunk.variant_start_index,
                    variant_stop_index: chunk.variant_stop_index,
                })
                .collect(),
            layout: Arc::clone(&layout),
        });
        Ok(Some(layout))
    }
}

/// Runtime controls and output state for one aligned phenotype group.
pub(crate) struct AssociationDeliverySettings {
    pub writer_sessions: Vec<Arc<OutputWriterSession>>,
    pub committed_chunk_identifier_sets: Vec<Arc<std::collections::BTreeSet<usize>>>,
    pub null_logistic_nonconvergence_policy: NullLogisticNonconvergencePolicy,
    pub progress: Option<DeliveryProgress>,
    pub gpu_genotype_format: GpuGenotypeFormat,
    pub statistics_policy: ChunkStatisticsPolicy,
}

/// One native delivery request for a trait-major phenotype group.
pub(crate) struct AssociationDeliveryRequest {
    pub group: AlignedPhenotypeGroup,
    pub settings: AssociationDeliverySettings,
}
