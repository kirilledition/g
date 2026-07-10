//! Python-free association delivery inputs.

use std::sync::Arc;

use g_input::AlignedPhenotypeGroup;
use g_output::OutputWriterSession;
use g_plan::{FloatingPointDtype, NullLogisticNonconvergencePolicy};

/// Runtime controls and output state for one aligned phenotype group.
pub(crate) struct AssociationDeliverySettings {
    pub writer_sessions: Vec<Arc<OutputWriterSession>>,
    pub committed_chunk_identifier_sets: Vec<Arc<std::collections::BTreeSet<usize>>>,
    pub null_logistic_nonconvergence_policy: NullLogisticNonconvergencePolicy,
    pub staging_depth: usize,
    pub result_in_flight_limit: usize,
    pub output_statistic_dtype: FloatingPointDtype,
    pub use_packed8: bool,
}

/// One native delivery request for a trait-major phenotype group.
pub(crate) struct AssociationDeliveryRequest {
    pub group: AlignedPhenotypeGroup,
    pub settings: AssociationDeliverySettings,
}

/// Delivery requests sharing one decoded union sample set.
pub(crate) struct GroupedUnionAssociationDeliveryRequest {
    pub groups: Vec<AssociationDeliveryRequest>,
    pub union_sample_indices: Vec<usize>,
}
