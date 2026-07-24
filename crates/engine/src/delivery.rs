//! Python-free association delivery inputs.

use g_genotype::ChunkStatisticsPolicy;
use g_input::AlignedPhenotypeGroup;
use g_output::OutputDeliveryToken;
use g_plan::{GpuGenotypeFormat, NullLogisticNonconvergencePolicy};

use crate::progress::DeliveryProgress;

/// Prepared genotype input shared by one or more association deliveries.
pub(crate) struct PreparedGenotypeInput {
    pub(crate) reader: g_genotype::BgenReaderCore,
    pub(crate) chunk_size: usize,
}

/// Runtime controls and output state for one aligned phenotype group.
pub(crate) struct AssociationDeliverySettings {
    pub output: OutputDeliveryToken,
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
