//! Existing output authority shared with resume planning.

use crate::association_implementation::AssociationImplementationCompatibility;

/// Content and implementation state required by an existing output plan.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ExistingOutputResumeAgreement {
    /// Authoritative fingerprint shared by every materialized phenotype manifest.
    pub bgen_content_fingerprint: g_genotype_contracts::BgenContentFingerprint,
    /// GPU genotype representation shared by every materialized phenotype manifest.
    pub gpu_genotype_format: g_plan::GpuGenotypeFormat,
    /// Runtime association implementation shared by every materialized phenotype manifest.
    pub association_implementation: AssociationImplementationCompatibility,
}
