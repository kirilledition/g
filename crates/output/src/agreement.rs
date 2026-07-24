//! Existing output authority shared with resume planning.

/// BGEN content and GPU representation required by an existing output plan.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ExistingOutputResumeAgreement {
    /// Authoritative fingerprint shared by every materialized phenotype manifest.
    pub bgen_content_fingerprint: g_genotype_contracts::BgenContentFingerprint,
    /// GPU genotype representation shared by every materialized phenotype manifest.
    pub gpu_genotype_format: g_plan::GpuGenotypeFormat,
}
