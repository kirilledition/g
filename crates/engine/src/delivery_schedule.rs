//! GPU genotype-format resolution policy.

mod genotype_format;

pub use genotype_format::{
    GpuGenotypeFormatResolutionPlan, plan_auto_gpu_genotype_format_after_trusted_validation,
    plan_gpu_genotype_format_auto_to_dosage, plan_single_trait_binary_gpu_genotype_format_resolution,
    resolve_effective_trusted_no_missing_diploid, resolve_manifest_gpu_genotype_format,
};
